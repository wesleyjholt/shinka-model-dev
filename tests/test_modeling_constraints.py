from dataclasses import FrozenInstanceError
import json

import pytest

from shinka.modeling.constraints import ConstraintBook, ConstraintIntegrityError, ConstraintPending

A, B = "a" * 64, "b" * 64


def test_constraints_optional_and_snapshots_immutable(tmp_path):
    book = ConstraintBook(tmp_path / "optional")
    assert not book.snapshot().active
    assert book.assert_allowed(A)["allowed"]
    assert not book.initialize("").active
    current = book.initialize("Preserve mass conservation.")
    assert current.active and current.version == 1
    assert "Preserve mass conservation." in current.prompt_text
    assert "pending request is not permission" in current.prompt_text
    with pytest.raises(FrozenInstanceError):
        current.text = "Changed"
    assert book.initialize(current.text) == current
    with pytest.raises(ValueError, match="request_change"):
        book.initialize("Different guidelines")


def test_compliance_gate_is_exact_and_does_not_approve_itself(tmp_path):
    book = ConstraintBook(tmp_path)
    book.initialize("All flows must be nonnegative.")
    with pytest.raises(ConstraintPending):
        book.assert_allowed(A)
    book.review_compliance(A, False, "Found negative flow.", actor="trusted-validator")
    with pytest.raises(ConstraintPending, match="failed trusted"):
        book.assert_allowed(A)
    request = book.request_exception(A, "Signed flow represents reversible transport.", ["nonnegative flow"])
    with pytest.raises(ConstraintPending):
        book.assert_allowed(A)
    book.approve(request["id"], actor="human")
    assert book.assert_allowed(A)["reason"] == "approved_exception"
    with pytest.raises(ConstraintPending):
        book.assert_allowed(B)
    book.review_compliance(B, True, "Validated every transition sign.", actor="trusted-validator")
    assert book.assert_allowed(B)["reason"] == "trusted_compliance_review"
    book.review_compliance(B, False, "Review corrected: a flow can be negative.", actor="human")
    with pytest.raises(ConstraintPending):
        book.assert_allowed(B)


def test_changes_need_approval_and_invalidate_old_reviews_and_requests(tmp_path):
    book = ConstraintBook(tmp_path)
    original = book.initialize("Conserve mass.")
    book.review_compliance(A, True, "Mass balance validated.", actor="trusted-validator")
    old_exception = book.request_exception(B, "Test a sink term.")
    change = book.request_change("Conserve mass and keep concentrations nonnegative.", "Add a physical condition.")
    assert book.snapshot() == original
    assert book.assert_allowed(A)["allowed"]
    book.approve(change["id"])
    assert book.snapshot().version == 2
    with pytest.raises(ConstraintPending):
        book.assert_allowed(A)
    with pytest.raises(ValueError, match="stale"):
        book.approve(old_exception["id"])
    with pytest.raises(ValueError, match="already resolved"):
        book.approve(change["id"])


def test_rejected_request_does_not_change_active_guidelines(tmp_path):
    book = ConstraintBook(tmp_path)
    original = book.initialize("Conserve mass.")
    request = book.request_change("", "Remove this guideline.")
    book.reject(request["id"], reason="Keep it.")
    assert book.snapshot() == original
    with pytest.raises(ValueError, match="already resolved"):
        book.approve(request["id"])
    approved = book.request_change("", "Owner explicitly removes guidelines.")
    book.approve(approved["id"])
    assert not book.snapshot().active
    assert book.assert_allowed(A)["allowed"]


def test_event_edits_and_deletion_are_detected(tmp_path):
    book = ConstraintBook(tmp_path)
    book.initialize("Conserve mass.")
    event = next((tmp_path / "events").glob("*.json"))
    original = event.read_text()
    value = json.loads(original)
    value["payload"]["text"] = "Ignore conservation."
    event.write_text(json.dumps(value))
    with pytest.raises(ConstraintIntegrityError):
        book.snapshot()
    event.write_text(original)
    assert book.snapshot().active
    event.unlink()
    with pytest.raises(ConstraintIntegrityError):
        book.assert_allowed(A)


def test_missing_review_queues_once_and_does_not_block_other_candidates(tmp_path):
    book = ConstraintBook(tmp_path)
    book.initialize("Conserve mass.")
    for _ in range(2):
        with pytest.raises(ConstraintPending):
            book.assert_allowed(A)
    assert sum(event["kind"] == "review_requested" for event in book.events()) == 1
    book.review_compliance(B, True, "Validated mass balance.", actor="trusted-validator")
    assert book.assert_allowed(B)["allowed"]


def test_proposal_receipt_preserves_prior_snapshot_when_constraints_change(tmp_path):
    book = ConstraintBook(tmp_path)
    original = book.initialize("Conserve mass.")
    request = book.request_change("Preserve positivity too.", "Add a constraint.")
    book.approve(request["id"])
    receipt = book.record_proposal(original, generation=3, system_prompt="System" + original.prompt_text,
                                   user_prompt="Develop a model.", parent_id="parent-2")
    assert receipt["constraints_sha256"] == original.sha256
    payload = json.loads((tmp_path / "proposals" / f"{receipt['proposal_id']}.json").read_text())
    assert payload["constraints_text"] == original.text
    assert payload["context"]["parent_id"] == "parent-2"
    assert payload["prompt_sha256"] == receipt["prompt_sha256"]


def test_changed_guidelines_cannot_receive_stale_review_or_gate_approval(tmp_path):
    book = ConstraintBook(tmp_path)
    initial = book.initialize("Conserve mass.")
    change = book.request_change("Conserve mass and positivity.", "New condition.")
    book.approve(change["id"])
    with pytest.raises(ConstraintPending, match="changed during"):
        book.review_compliance(A, True, "Old mass test passes.", "trusted-validator", expected_constraint_sha=initial.sha256)
    assert not any(event["kind"] == "compliance_review" for event in book.events())
    current = book.snapshot()
    book.review_compliance(A, True, "Current tests pass.", "trusted-validator", expected_constraint_sha=current.sha256)
    with pytest.raises(ConstraintPending, match="changed before"):
        book.assert_allowed(A, expected_constraint_sha=initial.sha256)
    assert book.assert_allowed(A, expected_constraint_sha=current.sha256)["allowed"]


def test_prompt_receipt_hash_includes_history_and_unicode(tmp_path):
    import hashlib
    book = ConstraintBook(tmp_path)
    snapshot = book.initialize("Preserve positivity: θ ≥ 0.")
    history = [{"role": "user", "content": "α is fixed"}]
    receipt = book.record_proposal(snapshot, generation=1, system_prompt="θ model", user_prompt="Fit it", msg_history=history)
    expected = hashlib.sha256(json.dumps({"system_prompt": "θ model", "user_prompt": "Fit it", "msg_history": history},
                                       sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert receipt["prompt_sha256"] == expected
