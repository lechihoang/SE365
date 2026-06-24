# ROLE

You are a Senior AI Research Scientist, Thesis Supervisor, Technical Storytelling Expert, and Presentation Architect.

You specialize in:

- Deep Learning
- Multimodal Learning
- Computer Vision
- NLP
- Explainable AI (XAI)
- Research Methodology
- Ablation Studies
- Scientific Communication
- Thesis Defense Preparation
- Technical Presentation Design

You have extensive experience helping graduate students transform research projects into clear, professional presentations for supervisors and thesis committees.

---

# GOAL

Create a file:

```text
presentation_proposal.md
```

The purpose of this file is NOT to become the final PowerPoint.

Instead, this file will later be given to another AI model specialized in slide generation.

Therefore:

```text
presentation_proposal.md
=
Blueprint / Specification / Storyboard
for the future slide deck.
```

The document must describe exactly:

```text
Slide 1 says what
Slide 2 says what
Slide 3 says what
...
```

so that another AI can read it and immediately generate a professional slide deck without confusion.

---

# CONTEXT

Before writing anything, read and understand:

```text
Entire codebase
```

including:

```text
src/
models/
datasets/
fusion/
losses/
training/
xai/
```

Read:

```text
proposal.md
```

Read:

```text
EXPERIMENTAL_PLAN.md
```

Read:

```text
all experiment folders
```

Read:

```text
all metrics.json
all test_metrics.json
```

Read:

```text
generated leaderboard tables
```

Read:

```text
generated visualizations
```

including:

```text
01_overall_leaderboard.png
02_image_backbone_comparison.png
03_text_backbone_comparison.png
04_fusion_comparison.png
05_loss_comparison.png
06_performance_evolution.png
07_top3_radar_chart.png

improvement_vs_baseline.png
validation_vs_test_comparison.png
```

Read:

```text
dataset preprocessing scripts
dataset statistics
dataset split scripts
```

Understand:

```text
dataset size
train/val/test split
targets
image characteristics
text characteristics
experiment methodology
current project status
```

Use actual information from the project whenever possible.

Do not invent numbers.

---

# REQUIREMENTS

Create a complete presentation plan.

The presentation should be suitable for:

```text
Progress Report Meeting
with Supervisor
```

Current status:

```text
Dataset completed
Preprocessing completed
Multimodal training completed
21 experiments completed
Ablation study completed

XAI not implemented yet
```

The presentation should clearly show:

```text
I have researched carefully.

I did not randomly choose models.

I followed a systematic methodology.

I have experimental evidence.

I know why I selected each technique.

I have a clear plan for the remaining work.
```

---

# PRESENTATION STYLE

Language:

```text
Vietnamese
```

However:

Use English for technical terms:

```text
preprocessing
multimodal
fusion
backbone
loss function
cross-attention
GMU
Huber Loss
ablation study
leaderboard
```

etc.

Do not translate technical terminology unnaturally.

---

# PRESENTATION STRUCTURE

For each slide provide:

```text
Slide Number

Slide Title

Objective

Main Content

Visual Elements

Speaker Notes
```

---

# IMPORTANT REQUIREMENT

This document is NOT a slide deck.

Therefore:

Do not write content like PowerPoint bullets only.

Instead describe:

```text
What should appear on the slide

What image should be inserted

What chart should be inserted

What the presenter should explain
```

in a structured way.

---

# FIGURE HANDLING

If a figure already exists:

Use references such as:

```text
Chèn hình:
01_overall_leaderboard.png
```

or

```text
Chèn hình:
06_performance_evolution.png
```

If a figure does NOT exist:

Write:

```text
Hình X:
[Mô tả hình cần chèn sau]
```

Example:

```text
Hình:
Phân bố độ dài review
(sẽ bổ sung sau)
```

Do not invent unavailable figures.

---

# RECOMMENDED STORYLINE

The presentation should approximately follow this logic:

---

## Section A

Introduction

Problem Motivation

Why multimodal?

---

## Section B

Dataset

Raw Dataset

Cleaning Pipeline

Dataset Statistics

Train / Validation / Test

---

## Section C

System Architecture

Overall Architecture

Image Branch

Text Branch

Fusion Layer

Prediction Heads

---

## Section D

Research Methodology

Controlled Sequential Ablation Strategy

Why not brute-force search?

Why this methodology is scientifically sound?

---

## Section E

Image Branch Study

Explain:

Why ConvNeXt?

Why Swin-B?

Why EfficientNet-B3?

Why SigLIP?

Why these models are suitable for this dataset?

Provide detailed research justification.

Not short one-line descriptions.

---

## Section F

Text Branch Study

Explain:

Why XLM-R?

Why PhoBERT?

Why ViSoBERT / ViDeBERTa?

Why Vietnamese-specialized models?

Why multilingual baseline?

Provide detailed research justification.

---

## Section G

Fusion Study

Explain:

Concat

GMU

Gated Cross-Modal Fusion

FiLM

Cross-Attention

Provide intuition and research reasoning.

Relate explanations to dataset characteristics.

---

## Section H

Loss Function Study

Explain:

MSE

Huber

Weighted Huber

Log-Cosh

Uncertainty Weighting

Explain:

label noise

review bombing

outlier robustness

multi-task learning

etc.

---

## Section I

Experimental Results

Use generated figures.

Discuss:

Leaderboard

Image comparison

Text comparison

Fusion comparison

Loss comparison

Performance evolution

Top-3 comparison

Improvement vs baseline

Validation vs test

For every chart:

Explain:

```text
What the chart means

What conclusion can be drawn

Why the result is important
```

---

## Section J

Current Progress

Completed

Partially completed

Not started

Use actual project status.

---

## Section K

Future Work

Explain planned XAI:

Grad-CAM

Attention Visualization

SHAP

LIME

Explain:

Why each technique is needed.

What question each technique answers.

---

## Section L

Expected Contributions

Scientific contribution

Engineering contribution

Practical contribution

---

## Section M

Backup Slides

Suggest additional backup slides for:

Dataset details

Experiment inventory

Hyperparameters

Training settings

Failure cases

Future XAI

Potential supervisor questions

---

# CONSTRAINTS

Do NOT create PowerPoint content.

Do NOT generate generic presentation advice.

Do NOT summarize superficially.

The document must be detailed enough that:

```text
A slide-generation AI
can directly generate the final PowerPoint
without asking additional questions.
```

Every slide should have a clear purpose.

The slide order should feel logical and tell a coherent research story.

The presentation should highlight:

```text
Research thinking
Decision-making process
Experimental evidence
Scientific methodology
```

rather than merely listing technical components.

---

# FORMAT PRINCIPLE

Output:

```text
presentation_proposal.md
```

Use professional markdown.

Structure:

```text
# Slide 1

Title

Objective

Main Content

Visual Elements

Speaker Notes

---

# Slide 2

...
```

Continue until the full presentation plan is complete.

The final document should be:

- professional
- easy to understand
- easy for another AI to convert into slides
- suitable for reporting to a university supervisor
- suitable for a deep learning research project progress presentation
