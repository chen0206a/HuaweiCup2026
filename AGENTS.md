# Agent Instructions

## Startup workflow
Before formal work:
1. Run `git status`.
2. If a remote exists, fetch/pull the latest version when safe and when local changes will not be overwritten.
3. Read `AGENTS.md`, `CONTEXT.md`, `STATUS.md`, and `DECISIONS.md`.
4. For a question-specific task, also read the matching `qX/HANDOFF.md`.
5. Read only files needed for the assigned task.
6. Do not scan the whole repository just to understand it. Do not read all historical experiments, CSVs, logs, or `archive/` by default. Investigate those only when a concrete task requires it.

## Shared project record
- Confirmed global decisions in `DECISIONS.md` are authoritative.
- `CONTEXT.md` is the current concise project snapshot. Keep it current after major changes; move obsolete detail to `archive/` instead of appending indefinitely.
- `qX/HANDOFF.md` is the preferred source for a question's agreed interpretation, current method, results, outputs, risks, and next steps.
- Never assume an experiment result from chat memory. Paper numbers and claims must come from actual result files.
- Never invent experiments, metrics, references, or conclusions.
- Do not change the shared validation protocol casually. If a decision has a serious problem, record the reason and propose a change before changing the protocol.
- Do not add model complexity just to appear innovative; complexity must serve the problem.

## Validation and data
For any machine-learning, prediction, or classification work, read `shared/validation_protocol.md` first. Check sample units, grouping, time/spatial structure, duplicate sources, and leakage before choosing a split. Update the protocol only through a reviewed global decision.

## Formal experiments
A formal experiment lives in `experiments/exp_XXX_name/` and contains:
- `config.yaml`: model, data version, features, split, random seed, and main hyperparameters at minimum.
- `metrics.json`: machine-readable results.
- `notes.md`: why it was run, result, whether it belongs in the paper, and next action.
- `figures/` when needed.
Do not register every temporary debugging run as a formal experiment.

## Writing and reproducibility
- Keep scripts lightweight and runnable from the repository root.
- Generate tables and summaries from committed, real result files.
- Do not put fabricated content into the paper template.
- Record data and code assumptions needed to reproduce an important result.

## Git collaboration
Before work, run `git status`; run `git pull` when applicable and safe. After a meaningful milestone, inspect `git diff`, stage only intended files, commit with a clear message, and push when applicable. Never overwrite another question's confirmed work without review. Do not resolve complex merge conflicts automatically. Prefer focused commits such as `Q2: add baseline`, `Paper: update Q1 results`, or `Docs: update validation protocol`.

## Default account roles
Roles are defaults and may be reassigned to match the competition workload.
- **Account A — Overall / Modeling Lead:** whole-problem interpretation, main direction, validation protocol, Q1 or core modeling, `DECISIONS.md`, final consistency.
- **Account B — Experiment Lead:** Python, baselines, features, batch experiments, ablations, robustness, experiment summaries.
- **Account C — Paper / Secondary Modeling Lead:** other questions, LaTeX, figures, tables, paper integration, final manuscript review.

## Task allocation guidance
Use high-reasoning effort for reading and decomposing the problem, comparing modeling routes, designing validation, identifying leakage, deciding after initial experiments, and final paper review. Use ordinary engineering effort for project setup, Python implementation, baselines, data processing, LaTeX, plots, tables, and routine fixes. Do not bind work to a particular AI model.
