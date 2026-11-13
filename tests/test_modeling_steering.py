"""Human steering is durable and cannot freeze ordinary proposal work."""

import asyncio
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from shinka.core.async_runner import ShinkaEvolveRunner
from shinka.core.config import EvolutionConfig
from shinka.modeling.steering import GuidanceInbox


def test_guidance_survives_restart_supersession_expiry_and_bad_input(tmp_path, caplog):
    inbox = GuidanceInbox(tmp_path)
    inbox.publish("direction", "Study poor time extrapolation.", message_id="early")
    before = inbox.snapshot()
    inbox.publish("direction", "Study subject variation.", message_id="next", supersedes=["early"], evidence_ids=["round-3"])
    inbox.publish("avoid", "Old numerical issue.", message_id="expired", expires_at="2000-01-01T00:00:00Z")
    (tmp_path / "inbox" / "bad.json").write_text("{broken")
    (tmp_path / "inbox" / "contract.json").write_text(json.dumps({
        "schema_version": 1, "id": "contract", "kind": "contract_change",
        "text": "Change validation split.", "created_at": "2026-01-01T00:00:00Z",
    }))
    after = GuidanceInbox(tmp_path).snapshot()
    assert [m.id for m in after.messages] == ["next"]
    assert [m.id for m in before.messages] == ["early"]
    assert after.version > before.version
    assert after.sha256 != before.sha256
    assert "round-3" in after.prompt_text
    assert "Ignored steering message" in caplog.text
    assert GuidanceInbox(tmp_path).snapshot() == after
    events = inbox.events()
    assert sum(e["event"] == "invalid" for e in events) == 2
    assert {(e["message_id"], e["reason"]) for e in events if e["event"] == "skipped"} == {
        ("early", "superseded"), ("expired", "expired"),
    }
    with pytest.raises(FrozenInstanceError):
        before.messages[0].text = "edited"


def test_mutated_file_does_not_replace_last_valid_guidance(tmp_path):
    inbox = GuidanceInbox(tmp_path)
    inbox.publish("preference", "Use interpretable models when possible.", message_id="m1")
    original = inbox.snapshot()
    path = next((tmp_path / "inbox").glob("*.json"))
    message = json.loads(path.read_text())
    message["text"] = "Edited in place"
    path.write_text(json.dumps(message))
    assert inbox.snapshot() == original
    assert any(e["event"] == "invalid" and "processed message" in e["reason"] for e in inbox.events())
    with pytest.raises(ValueError, match="duplicate"):
        inbox.publish("direction", "Duplicate.", message_id="m1")
    with pytest.raises(ValueError, match="contract"):
        inbox.publish("contract_change", "Change held-out scoring.")


def test_questions_never_wait_and_pause_resume_and_stop_work(tmp_path):
    async def run():
        inbox = GuidanceInbox(tmp_path)
        inbox.publish("question", "Can we identify the clearance rate?", message_id="question")
        assert not (await asyncio.wait_for(inbox.wait_if_paused(), 1)).paused
        inbox.publish("pause", "Review before the next proposal.")
        waiting = asyncio.create_task(inbox.wait_if_paused(poll_interval=0.01))
        await asyncio.sleep(0.03)
        assert not waiting.done()
        inbox.publish("resume", "Continue with the current contract.")
        assert not (await asyncio.wait_for(waiting, 1)).paused
        inbox.publish("pause", "Review again.")
        stop = asyncio.Event()
        waiting = asyncio.create_task(inbox.wait_if_paused(stop_event=stop, poll_interval=0.01))
        stop.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, 1)
    asyncio.run(run())


def test_initial_review_checkpoint_requires_explicit_resume(tmp_path):
    inbox = GuidanceInbox(tmp_path, start_paused=True)
    assert inbox.snapshot().paused
    inbox.publish("question", "What should be learned next?")
    assert inbox.snapshot().paused
    inbox.publish("resume", "Proceed.")
    assert not GuidanceInbox(tmp_path, start_paused=True).snapshot().paused


def _runner(tmp_path, query):
    instance = object.__new__(ShinkaEvolveRunner)
    instance.evo_config = EvolutionConfig(steering_dir=str(tmp_path), steering_poll_interval=0.01)
    instance.guidance_inbox = GuidanceInbox(tmp_path)
    instance.should_stop = asyncio.Event()
    instance.verbose = False
    instance.llm_selection = None
    instance.llm = SimpleNamespace(get_kwargs=lambda **kwargs: {}, query=query)

    class Sampler:
        task_sys_msg = "Original prompt"

        def sample(self, **kwargs):
            return self.task_sys_msg + "\nFORMAT INSTRUCTIONS", "Parent and research history", "full"

        def sample_fix(self, **kwargs):
            return self.task_sys_msg, "Repair the program", "full"

        def initial_program_prompt(self):
            return self.task_sys_msg, "Create the initial program"

    instance.prompt_sampler = Sampler()
    instance._get_current_system_prompt = lambda: ("EVOLVED PROMPT", "prompt-evolved")

    async def save(**kwargs):
        pass

    instance._save_patch_attempt_async = save
    return instance


def test_runner_pauses_new_requests_but_existing_work_can_finish(tmp_path):
    async def run():
        requests = []

        async def query(**kwargs):
            requests.append(kwargs)
            return None

        runner = _runner(tmp_path, query)
        runner.guidance_inbox.publish("pause", "Please pause new proposals.")
        task = asyncio.create_task(runner._run_patch_async(None, [], [], 1))
        completed_evaluation = asyncio.Event()

        async def evaluation():
            await asyncio.sleep(0.02)
            completed_evaluation.set()

        await evaluation()
        assert completed_evaluation.is_set()
        assert not requests and not task.done()
        runner.guidance_inbox.publish("resume", "Continue.")
        result = await asyncio.wait_for(task, 2)
        assert len(requests) == 1
        assert result[1]["guidance_version"] > 0
    asyncio.run(run())


def test_runner_snapshots_each_proposal_without_changing_shared_sampler(tmp_path):
    async def run():
        first_started = asyncio.Event()
        finish_first = asyncio.Event()
        requests = []

        async def query(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                first_started.set()
                await finish_first.wait()
            return None

        runner = _runner(tmp_path, query)
        runner.guidance_inbox.publish("direction", "Look for distinct mechanisms.", message_id="explore")
        first = asyncio.create_task(runner._run_patch_async(None, [], [], 1))
        await asyncio.wait_for(first_started.wait(), 2)
        runner.guidance_inbox.publish("direction", "Revisit promising mechanisms.", message_id="exploit", supersedes=["explore"])
        second_result = await runner._run_patch_async(None, [], [], 2)
        finish_first.set()
        first_result = await first
        assert runner.prompt_sampler.task_sys_msg == "Original prompt"
        assert "Look for distinct mechanisms." in requests[0]["system_msg"]
        assert "Revisit promising mechanisms." not in requests[0]["system_msg"]
        assert "Revisit promising mechanisms." in requests[1]["system_msg"]
        assert requests[1]["system_msg"].index("Human scientific guidance") > requests[1]["system_msg"].index("FORMAT INSTRUCTIONS")
        assert first_result[1]["guidance_sha256"] != second_result[1]["guidance_sha256"]
        receipts = [json.loads(p.read_text()) for p in (tmp_path / "proposals").glob("*.json")]
        assert {r["generation"] for r in receipts} == {1, 2}
        assert {r["system_prompt"] for r in receipts} == {r["system_msg"] for r in requests}
        assert all("prompt_sha256" in r for r in receipts)
        assert {e["message_id"] for e in runner.guidance_inbox.events() if e["event"] == "applied"} == {"explore", "exploit"}
    asyncio.run(run())


def test_default_configuration_is_backward_compatible(tmp_path):
    async def run():
        runner = object.__new__(ShinkaEvolveRunner)
        assert EvolutionConfig().steering_dir is None
        assert not EvolutionConfig().steering_start_paused
        assert await runner._proposal_guidance() is None
        assert runner._append_guidance("original", None) == "original"
        assert await runner._record_guidance_request(None, generation=1, system_prompt="a", user_prompt="b") == {}
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["initial", "repair"])
def test_guidance_reaches_initial_and_repair_requests(tmp_path, mode):
    async def run():
        requests = []

        async def query(**kwargs):
            requests.append(kwargs)
            return None

        runner = _runner(tmp_path, query)
        runner.total_api_cost = 0
        runner.guidance_inbox.publish("preference", "Prefer a clear scientific explanation.")
        if mode == "initial":
            async def reconcile(*args):
                pass

            runner._reconcile_initial_costs = reconcile
            with pytest.raises(RuntimeError, match="failed to generate"):
                await runner._generate_initial_program()
        else:
            await runner._run_fix_patch_async(None, [], 1)
        assert len(requests) == 1
        assert "Prefer a clear scientific explanation." in requests[0]["system_msg"]
        assert len(list((tmp_path / "proposals").glob("*.json"))) == 1
    asyncio.run(run())
