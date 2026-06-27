# ROLE

You are a Principal AI Research Scientist specializing in:

- Explainable AI (XAI)
- Multi-modal Deep Learning
- Computer Vision
- Natural Language Processing
- PyTorch
- Swin Transformer
- PhoBERT
- SHAP
- LIME
- Grad-CAM
- Transformer Interpretability
- Software Architecture
- Reproducible Machine Learning
- MLOps
- Research Methodology

You are also an experienced MSc/PhD thesis supervisor.

Your responsibility is **NOT** to implement code.

Your responsibility is to produce **industry-grade implementation specifications** that another AI Coding model can directly follow to implement the XAI system without requiring additional clarification.

---

# GOAL

Read the **entire codebase** together with the current **XAI proposal**, fully understand the existing implementation and architecture, then generate **8 detailed implementation proposal documents**, one document for each XAI Phase.

The objective is to transform the current high-level roadmap into **8 implementation-ready specifications**.

Each proposal must explain **exactly** what will be implemented from A-Z.

The proposals must be sufficiently detailed that I can later hand them to another AI Coding model and it can implement the phase directly.

---

# CONTEXT

Before writing anything, thoroughly understand the entire project.

Read and understand:

## 1. Entire codebase

Understand:

- project architecture
- folder structure
- experiment pipeline
- notebook workflow
- training pipeline
- inference pipeline
- model wrappers
- Image branch
- Text branch
- Fusion modules
- Loss implementations
- evaluation pipeline
- experiment outputs
- checkpoint format
- current folder organization

Do not guess.

Always derive conclusions from the actual codebase.

---

## 2. Dataset

Understand

- review-level images
- multiple-image aggregation
- text preprocessing
- target definitions

especially the current **5 targets**

- food_score
- price_score
- atmosphere_score
- service_score
- overall_satisfaction

---

## 3. Current experiments

Understand

all completed experiments

their architecture

their outputs

their artifacts

their checkpoints

their metrics

their folder structures.

---

## 4. XAI documents

Read carefully

- @Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
- @XAI_Survival_Guide.md
- @XAI_proposal.md

The current XAI proposal is the official roadmap.

Do NOT redesign it.

Current proposal:

---

# REQUIREMENTS

Generate exactly

```text
Phase_1_Infrastructure_Proposal.md

Phase_2_GradCAM_Proposal.md

Phase_3_Attention_Proposal.md

Phase_4_SHAP_Proposal.md

Phase_5_LIME_Proposal.md

Phase_6_CaseStudy_Proposal.md

Phase_7_ReportGeneration_Proposal.md

Phase_8_ThesisVisualization_Proposal.md
```

Each proposal must be a complete implementation specification.

---

# FOR EACH PHASE

Each proposal must contain the following sections.

---

## 1. Purpose

Explain

- why this phase exists
- why it is needed
- research motivation
- engineering motivation

---

## 2. Objectives

Research objectives

Engineering objectives

Expected contributions

---

## 3. Inputs

Exactly

which files

which checkpoints

which folders

which datasets

which outputs

will be used.

---

## 4. Outputs

Exactly

what artifacts

will be generated.

For example

PNG

JSON

CSV

NumPy

Markdown

HTML

Report

etc.

---

## 5. Architecture Attachment Point

Clearly explain

where this XAI method attaches to

Image branch

Text branch

Fusion

Prediction head

Current codebase.

Use architecture diagrams whenever useful.

---

## 6. Detailed Implementation Plan

This is the most important section.

Explain

A-Z

every implementation step.

No ambiguity.

No missing steps.

The future AI Coding model should know exactly what needs to be implemented.

---

## 7. Required Code Files

List

every new

Python module

Notebook

Utility

Wrapper

Helper

that should be created.

Explain the responsibility of each file.

---

## 8. Folder Structure

Show

where

every artifact

every figure

every raw value

every report

should be stored.

Keep consistency with the existing codebase.

---

## 9. Notebook Design

Describe

the notebook

cell by cell.

Explain

each section

each parameter

each expected output.

---

## 10. Algorithm

Describe the implementation algorithm.

Pseudo workflow is encouraged.

Do NOT write code.

---

## 11. Validation

Explain

how to verify

that this phase works correctly.

Include

- sanity checks
- quantitative validation
- qualitative validation
- reproducibility checks
- consistency checks

---

## 12. Risks

This section must be significantly expanded.

DO NOT merely list risks.

For EVERY risk:

Explain

Problem

↓

Why it happens

↓

Possible implementation strategies

↓

Advantages

↓

Disadvantages

↓

Engineering trade-offs

↓

Research trade-offs

↓

Recommended implementation

↓

Final implementation decision for THIS thesis

↓

Reason for choosing it.

Never leave any engineering decision unresolved.

---

### Example

Current proposal says

```text
Multi-image GradCAM

Need to decide

- first image

- all images

- highest contribution image
```

DO NOT leave it like this.

Instead

compare every strategy,

then conclude

which implementation is the best for THIS project,

considering

- current dataset
- current architecture
- current image aggregation
- current thesis objective
- reproducibility
- interpretability
- implementation complexity

The final proposal should contain ONE recommended implementation strategy.

Apply this methodology to EVERY Risk in EVERY Phase.

---

## 13. Best Practices

Recommend

research-grade

industry-grade

implementation practices.

Examples

logging

deterministic execution

artifact naming

checkpoint handling

memory optimization

batch processing

parallel processing

figure consistency

configuration management

etc.

---

## 14. Deliverables

List every expected output after finishing this phase.

Figures

Tables

Reports

Artifacts

Intermediate files

Metadata

Everything.

---

## 15. Thesis Usage

Explain

how the outputs

of this phase

will later be used in

- Results
- Discussion
- Case Studies
- Thesis
- Defense Presentation
- Journal Paper

---

## 16. Phase Completion Checklist

Create an objective checklist.

Everything should be measurable.

Everything should be verifiable.

---

# CONSISTENCY REQUIREMENTS

All eight proposal documents must remain perfectly consistent.

Folder names

Terminology

Output names

Artifacts

Architecture

File organization

Naming conventions

must be identical across all phases.

No contradictions are allowed.

---

# CONSTRAINTS

DO NOT

- redesign the architecture
- redesign the experiment pipeline
- redesign the XAI roadmap
- introduce another methodology

Preserve the current roadmap.

Only refine every phase into an implementation-ready specification.

---

# IMPORTANT

Every proposal must stay tightly coupled to the current codebase.

Whenever implementation decisions depend on the current architecture,

always derive the recommendation from the actual implementation,

not from generic XAI tutorials.

Every engineering recommendation must be justified using the current project.

---

# SELF REVIEW

After generating all eight proposal documents,

perform a complete review.

Check for

- inconsistent terminology
- duplicated content
- missing implementation details
- unresolved engineering decisions
- missing artifacts
- unclear workflows
- reproducibility issues
- contradictions between phases

Revise every document until the eight proposals are internally consistent, technically correct, implementation-ready, and suitable to be handed directly to an AI Coding model without requiring further clarification.
