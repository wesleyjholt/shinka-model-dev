---
name: shinka-model-development
description: Interview a scientist and specify a Shinka model-development or calibration campaign before launch, then guide execution under the approved objective and constraints. Use for scientific model search; not routine program-speed optimization or inspection of completed results.
---

# Scientific model development

Before a new scientific campaign, conduct a short, adaptive interview and obtain
the scientist's approval of a concrete campaign specification. The scientist need
not know an objective function or calibration algorithm. Translate their intended
use into a proposed evaluation design and explain the important trade-offs.

## Interview: adaptive questions, required information

Inspect supplied context and previous answers first. Start with **3–5 short
questions at most**, using fewer when the information is already available. Ask in
small groups. Do not present a long form or hide a long interview inside compound
questions. An existing conversation can supply the interview evidence; do not
repeat answered questions just to complete a checklist.

**Generate the questions and answer choices for this project. Do not use a fixed
question bank.** Prefer the platform's structured question tool with a few short,
concrete choices (usually 2–4, within the tool's limit) and free-text input. If no
such tool is available, present lettered choices in chat. Offer an uncertainty or
delegation choice, such as “Unsure—recommend an approach,” when it is useful. If
the interface already permits free text, do not add a redundant “Other” choice.
For compatible aims, allow combinations rather than forcing a false choice.

Use plain language about the scientist's problem. Avoid asking them to choose
technical scores, samplers, or software before their purpose is clear. Choices
must not narrow the model space or imply that a proposed default is approved.
Never treat a preselected option, silence, or elapsed time as an answer.

Regardless of the wording or number of questions, **account for these five fields**:

| Required field | Information to recover |
| --- | --- |
| Purpose and intended use | Scientific question, decision, or use case; relevant conditions and expected generalization. |
| Prediction priorities and useful improvement | Important outputs or features; what useful progress means; consequential errors or trade-offs when relevant. |
| Evidence and starting point | Available data and uncertainty information, existing models, prior results, papers, or expert knowledge. Record missing evidence too. |
| Optional required constraints | Requirements that must hold, distinguished from preferences. “No additional constraints” is valid; unknown is not equivalent to none. |
| Practical limits | Relevant time, compute, cost, data-access, or execution-location limits; who can authorize unresolved resource choices. |

Record each field in `campaign-spec.md` with the scientist's answer or existing
context reference, its status, and any assumption. Valid statuses are **known**,
**explicitly none/not applicable**, **scientist unsure**, and **agent proposed**.
An uncertain answer is useful information; propose a reasonable plan instead of
repeating the question. Do not silently fill gaps with supposed user preferences.

Check coverage before moving on. Ask a targeted follow-up only if a remaining gap
would materially change the objective, feasibility, model boundary, or evaluation
plan. Otherwise state the assumption in the proposal for review. Keep the initial
question budget small; technical detail belongs in the proposed plan.

## Turn the answers into an approvable campaign

Write a concise specification that connects the intended use to:

- The objective, score direction, important outputs, and any weighting or trade-offs.
  Explain why the proposed measure represents useful progress. Offer a small
  number of alternatives only when the choice matters.
- Training and development evaluation, generalization cases, and an independent
  final assessment where available. State measurement/noise assumptions and any
  uncertainty requirement. Repeatedly searched validation data are not an
  untouched final test.
- Calibration methods and budgets for screening, refinement, and confirmation;
  method-specific reliability checks; promotion and stopping rules. Recommend
  these from the use case rather than making the scientist invent them. Compare
  finalists under the same declared protocol; MAP is a point estimate.
- Optional mandatory model constraints and ordinary research preferences. Keep
  model structure open otherwise. Papers inform hypotheses but do not define an
  allowed model list; novel, unpublished hypotheses need no citation to qualify.
  Reserve exploration effort as well as refinement of successful models.
- Execution location, finite resource allocations, permitted data exposure,
  reports, and asynchronous steering. Remote execution requires queue-aware,
  durable job handling; it is not a long-running local callback.

Expose material unresolved assumptions and ask for approval of **this concrete
specification**, not another questionnaire. Record the approved specification
version and the scientist's actual approval reference. A numerical campaign must
not launch until the coverage check is complete and the specification is approved.
If approval is already explicit for this exact specification, do not ask again.
Read-only preparation can continue while an answer is pending.

## Execute and steer

Read the fork's [model-development guide](../../MODEL_DEVELOPMENT.md) for the
calibration APIs, records, reports, paper library, and constraint gate. For cluster
work, inspect the available remote execution adapter and its profile workflow;
do not assume a cluster is commissioned merely because the adapter is installed.
Use the scientist's approved author account/model route.

Ordinary guidance and paper updates take effect between proposals without waiting
for a reply. Mandatory constraints stay active until the scientist explicitly
approves a change or a candidate-specific exception. Submit that permission
request and continue other compliant work; authors cannot self-approve.

Resume an approved campaign without a new interview when its objective,
evaluation contract, constraints, and resource authorization remain applicable.
For a material change, ask only about the changed information and obtain approval
of the revised specification. Do not silently change the comparison contract or
require a new interview for an ordinary direction update within it.
