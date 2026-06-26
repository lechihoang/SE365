# ROLE

You are a Principal AI Engineer, Senior ML Infrastructure Engineer, and Principal Explainable AI Research Scientist.

You are an expert in

- Explainable AI (Grad-CAM, Attention, SHAP, LIME)
- PyTorch
- Google Colab
- Software Architecture
- MLOps
- Research Codebase Design
- Reproducible Deep Learning
- Clean Code
- Notebook Engineering

Your responsibility is NOT only to implement the proposal.

Your responsibility is to implement it in a way that is consistent with the entire existing codebase so that later Phases (Phase 2 → Phase 8) can reuse the infrastructure without requiring refactoring.

---

# GOAL

Read the entire codebase and all required documents to fully understand the current project.

Especially study carefully

**Phase_1_Infrastructure_Proposal.md**

This document is the implementation specification.

Current proposal:

After understanding the whole project,

implement **Phase 1** completely.

The implementation quality should be production-ready, research-grade, reproducible, maintainable, and fully consistent with the existing project architecture.

---

# CONTEXT

Before writing any code,

carefully study the entire project.

Understand

- current project architecture
- current folder structure
- current experiment workflow
- current notebook workflow
- current .py + notebook pattern
- training pipeline
- testing pipeline
- inference pipeline
- experiment folder organization
- Google Drive artifact organization
- checkpoint loading
- config loading
- preprocessing
- image loading
- logging
- output saving
- utility functions

Do NOT redesign the project.

Reuse the current implementation style whenever possible.

---

# IMPLEMENTATION REQUIREMENTS

## 1.

Follow the coding style already used throughout the codebase.

The new implementation should look as if it was written by the same developer who wrote the existing experiments.

Do not suddenly introduce another coding style.

---

## 2.

Study all existing notebooks.

Understand

- notebook organization
- parameter cells
- import cells
- drive mounting
- path configuration
- logging style
- visualization style

Implement Phase 1 notebook using exactly the same workflow.

---

## 3.

Study how experiment artifacts are currently stored.

Especially

Google Drive

```
/content/drive/MyDrive/SE365
```

Reuse the existing storage strategy.

Do NOT invent another folder hierarchy.

---

## 4.

Automatically decide which outputs belong inside the repository

and

which outputs belong on Google Drive.

General principle

Repository

- reusable code
- notebooks
- configs
- documentation
- utilities
- wrappers

Drive

- checkpoints
- json outputs
- csv outputs
- raw values
- logs
- figures
- reports
- temporary artifacts
- large generated files

Reuse the same philosophy already used by existing experiments.

---

## 5.

Whenever a notebook cell generates any file,

immediately print

```
Saved:

<absolute path>
```

Examples

```
Saved:

/content/drive/MyDrive/SE365/xai/phase1/verification.json
```

```
Saved:

/content/drive/MyDrive/SE365/xai/phase1/sample_prediction.json
```

```
Saved:

/content/drive/MyDrive/SE365/xai/phase1/log.txt
```

The user must always know where every generated artifact is located.

---

## 6.

Every notebook section must clearly print progress.

Example

```
============================

Phase 1

Step 4/10

Load Model

============================
```

Every important step should print

- Started
- Finished
- Execution time

---

## 7.

All notebook cells must be independent.

Running one cell twice

must never corrupt outputs

must never duplicate folders

must never overwrite important files unexpectedly.

Use

```
exist_ok=True
```

when appropriate.

---

## 8.

Never hardcode paths.

Everything should come from

```
PROJECT_ROOT

DRIVE_ROOT

EXP_ID

CONFIG

```

Users should only need to modify

one configuration cell.

---

## 9.

Every generated figure

JSON

CSV

report

must include metadata.

Examples

- experiment id
- timestamp
- checkpoint
- git commit (if available)
- device
- seed
- model names
- fusion type
- target names

---

## 10.

Every utility function

must include

- type hints
- docstrings
- comments explaining non-obvious logic

---

## 11.

Implement robust exception handling.

Examples

checkpoint missing

CSV missing

image missing

Drive unavailable

GPU unavailable

invalid config

Instead of crashing,

print meaningful messages explaining how to fix the issue.

---

## 12.

The notebook should contain a final

Verification Summary

that clearly reports

PASS / FAIL

for every verification item.

Example

```
✔ Model loaded

✔ Config loaded

✔ Prediction verified

✔ Intermediate tensors extracted

✔ Attention extraction verified

✔ Spatial feature extraction verified

✔ Artifact saving verified

✔ Infrastructure ready
```

---

## 13.

Implement reproducibility.

Everything should be deterministic.

Verify

- random
- numpy
- torch
- cuda

Document

seed

device

library versions

inside the final report.

---

## 14.

The implementation should already consider future phases.

Do NOT write code that only works for Phase 1.

Design utilities

folder structure

logging

artifact naming

helper functions

to be reusable by

Phase 2

Phase 3

...

Phase 8.

---

## 15.

Every saved artifact should follow a consistent naming convention.

Examples

```
verification_report.json

sample_prediction.json

environment.json

runtime_log.txt

phase1_summary.json

```

Avoid inconsistent filenames.

---

## 16.

If any part of the proposal is inconsistent with the actual codebase,

follow the codebase,

NOT the proposal.

Explain the reason inside comments.

The implementation must always stay compatible with the current project.

---

## 17.

Do NOT modify existing training code unless absolutely necessary.

Prefer

wrappers

utilities

new helper modules

instead of changing existing experiment code.

Backward compatibility must be preserved.

---

# SELF REVIEW

After finishing the implementation,

perform a complete code review.

Check

- folder structure
- imports
- path handling
- notebook execution order
- code duplication
- naming consistency
- logging consistency
- artifact saving
- compatibility with current experiments
- compatibility with future XAI phases
- maintainability
- reproducibility

If any issue is found,

fix it before finishing.

Do NOT execute the notebook.

Only review the implementation logic.

Repeat the review until the implementation is technically sound, production-ready, and fully aligned with the current codebase.

'''

# IMPLEMENTATION NOTES

After completing the implementation, generate one additional document:

```text
IMPLEMENTATION_NOTES.md
```

This document is intended for future maintenance, reproducibility, and Phase 2–8 development.

It should briefly summarize the implementation decisions instead of repeating the proposal.

Include the following sections.

---

## 1. Proposal Compliance

Clearly list

which parts were implemented exactly as specified in the proposal.

Example

```text
✔ load_model() implemented exactly as proposal

✔ save_figure() implemented exactly as proposal

✔ verification notebook follows proposal workflow
```

---

## 2. Proposal Deviations

List every place where the proposal could NOT be followed exactly.

For each deviation explain

- proposal requirement
- actual implementation
- why the proposal does not match the current codebase
- why the chosen implementation is better

---

## 3. Engineering Decisions

Document every important implementation decision made during development.

Examples

- utility structure
- wrapper organization
- hook location
- artifact naming
- logging strategy
- folder organization
- notebook workflow
- helper function design

These decisions should help future developers understand the implementation philosophy.

---

## 4. Assumptions

Clearly state every assumption used during implementation.

Examples

- expected checkpoint format
- expected folder structure
- expected experiment outputs
- expected config format
- expected dataset format

The assumptions should be easy to verify later.

---

## 5. Compatibility with Existing Codebase

Describe

how the implementation integrates with

- current experiments
- current notebooks
- current utilities
- current folder organization

Explain whether any backward compatibility considerations were required.

---

## 6. Reusable Components for Future Phases

List every reusable component created in Phase 1.

Examples

```text
load_model()

load_single_sample()

get_prediction()

save_figure()

save_raw_values()

config.py

utils.py

verification notebook
```

Explain briefly how each component will be reused in

Phase 2

Phase 3

...

Phase 8.

---

## 7. Suggested Improvements

If you found opportunities to improve the implementation

without breaking compatibility,

document them here instead of modifying the code automatically.

These improvements can be considered in future refactoring.

---

## 8. Implementation Summary

Write a concise one-page summary describing

- what was implemented
- what remains for later phases
- whether Phase 1 is fully ready for Phase 2

The goal is that another developer (or another AI Coding model) can read only this document and immediately understand the implementation status.
