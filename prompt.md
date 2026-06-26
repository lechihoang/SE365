# ROLE

You are a Senior Explainable AI Research Scientist, Multimodal Deep Learning Engineer, PyTorch Engineer, and Thesis Methodology Advisor.

You specialize in:

- Explainable AI
- Grad-CAM
- Attention Visualization
- SHAP
- LIME
- Multimodal Deep Learning
- Computer Vision
- NLP
- Regression Modeling
- PyTorch
- Research Experiment Design
- Thesis-ready technical writing

Your task is to read the current codebase and update the XAI documentation so it is fully aligned with the latest implemented experiments.

---

# GOAL

Read the entire codebase and all relevant project documents.

Then do three tasks:

1. Update:

```text
@Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
```

2. Update:

```text
@XAI_Survival_Guide.md
```

3. Create a new file:

```text
XAI_proposal.md
```

The goal is to prepare a clear, phased, implementation-ready proposal for the XAI stage of the thesis.

---

# CONTEXT

The project is now past the training/experiment stage.

Current status:

```text
Dataset completed
Preprocessing completed
21 experiments completed
Ablation study completed
Best model / promising candidates selected
XAI not implemented yet
```

The current model is a multimodal regression system using:

```text
Review Text
+
Review Images
```

to predict 5 targets:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

Important:

The old XAI documents may still mention outdated targets such as:

```text
quality_score
appearance_score
overall_score
```

These must be updated.

The correct current targets are:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

Do not keep outdated target names unless explaining historical context.

---

# FILES TO READ

Before writing anything, read:

```text
Entire codebase
```

Especially:

```text
Models/
Trainer.py
main.py
test.py
Config.py
src/dataset.py
experiments/
notebooks/
reports/
```

Read these project understanding files:

```text
@CODEBASE_OVERVIEW.md
@codebase_experiment_flow.md
@experiment_plan.md
@phase_selections.md
@presentation_proposal.md
```

Read these XAI guide files carefully:

```text
@Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
@XAI_Survival_Guide.md
```

The two XAI files contain useful ideas, but some content is outdated. Update them carefully instead of discarding everything.

---

# REQUIREMENTS PART 1 — UPDATE EXISTING XAI GUIDES

Update both existing XAI guide files so they match the current codebase.

## Must update target names

Replace old target names:

```text
quality_score
appearance_score
overall_score
```

with the current target names:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

Use the exact names implemented in the codebase.

---

## Must update architecture descriptions

The old guide may assume:

```text
ConvNeXt + XLM-R + Concatenation
```

But the current experiments include many variants:

```text
Image backbones:
ConvNeXt
Swin-B
EfficientNet-B3
SigLIP

Text backbones:
XLM-R
PhoBERT
ViSoBERT

Fusion:
Concat
GMU
Gated Cross-Modal
FiLM
Cross-Attention

Loss:
MSE
Huber
Log-Cosh
Uncertainty Weighted
```

Update the XAI guides so they describe the general architecture, not only one outdated baseline.

Write it in a way that supports multiple experiment configurations.

---

## Must update XAI mapping

The updated guides should clearly map:

| Model Part            | XAI Method                               | Purpose                                        |
| --------------------- | ---------------------------------------- | ---------------------------------------------- |
| Image branch          | Grad-CAM                                 | Where did the image model focus?               |
| Text branch           | Attention Visualization / token saliency | Which tokens were influential?                 |
| Fusion layer          | SHAP                                     | How much did image/text contribute?            |
| Local sample behavior | LIME                                     | What happens when words/regions are perturbed? |

---

## Must update multimodal constraints

The guides must emphasize:

- For image explanation, keep text fixed.
- For text explanation, keep image fixed.
- For fusion SHAP, explain the fused representation or modality-level contribution.
- Always explain one target at a time.
- Save raw numeric explanation values, not only figures.
- Do not overclaim causality.
- Attention is not automatically explanation.
- XAI is for interpretation and debugging, not proof of causation.

---

# REQUIREMENTS PART 2 — CREATE XAI_proposal.md

Create a new file:

```text
XAI_proposal.md
```

This file should be a practical proposal for implementing XAI after the 21 experiments.

It should be phase-based to avoid implementing everything at once.

---

# REQUIRED XAI_proposal.md STRUCTURE

Use this structure.

---

## 1. Project Context

Explain briefly:

- Current project status
- Current 5 prediction targets
- Current multimodal architecture
- Why XAI is needed after model training

---

## 2. XAI Objectives

Explain the main XAI goals:

```text
1. Explain image evidence
2. Explain text evidence
3. Explain modality contribution
4. Explain local sample behavior
5. Support thesis defense and debugging
```

---

## 3. XAI Target Outputs

Clearly define the 5 targets:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

For each target, explain what XAI should answer.

Example:

```text
food_score:
Does the model focus on food regions and food-related tokens?

price_score:
Does the model focus on price-related words such as expensive, cheap, reasonable?

service_score:
Does the model rely more on text because service is rarely visible in images?

atmosphere_score:
Does the image branch focus on restaurant interior or environment?

overall_satisfaction:
Does the model combine evidence from both image and text?
```

---

## 4. Recommended XAI Methods

Include:

```text
Grad-CAM
Attention Visualization
SHAP
LIME
```

For each method include:

- Purpose
- Input required
- Output generated
- Where it attaches in the architecture
- What target it explains
- Expected artifacts
- Limitations

---

## 5. Phase-based Implementation Roadmap

Split XAI implementation into phases.

Use something like:

```text
Phase 0: XAI Infrastructure Setup
Phase 1: Single-sample Demo
Phase 2: Grad-CAM for Image Branch
Phase 3: Attention Visualization for Text Branch
Phase 4: Fusion-level SHAP
Phase 5: LIME Local Explanation
Phase 6: Case Study Selection
Phase 7: XAI Report Generation
Phase 8: Thesis-ready Visualization and Defense Materials
```

For each phase include:

- Goal
- Why this phase exists
- Input files
- Code files to create
- Expected outputs
- Success criteria
- Risks

---

## 6. Which Experiments Should Be Explained?

Do NOT run XAI for all 21 experiments.

Recommend explaining:

```text
1. Best multimodal baseline
2. Best sequential model
3. Best promising combination, if different
4. A few failure cases
5. A few high-confidence correct cases
6. A few modality-conflict cases
```

Explain why full XAI for all 21 experiments is unnecessary and too expensive.

---

## 7. Case Study Strategy

Define case types:

```text
Correct prediction
High-error prediction
Image-text agreement
Image-text conflict
Text-dominant sample
Image-dominant sample
Service/price difficult sample
```

Explain why each case type is useful for thesis defense.

---

## 8. XAI Artifact Structure

Propose folder structure:

```text
experiments/
└── EXP_XXX/
    └── xai/
        ├── gradcam/
        ├── attention/
        ├── shap/
        ├── lime/
        ├── case_studies/
        ├── raw_values/
        └── README.md

reports/
└── xai/
    ├── xai_summary.md
    ├── xai_case_studies.md
    ├── modality_contribution_summary.csv
    └── figures/
```

---

## 9. XAI Output Files

List expected outputs:

```text
gradcam_food.png
gradcam_atmosphere.png
attention_price.png
attention_service.png
shap_modality_contribution.csv
lime_text_explanation.csv
lime_image_explanation.csv
case_study_summary.md
xai_report.md
```

Also require raw numeric files:

```text
gradcam_heatmap.npy
attention_weights.npy
shap_values.npy
lime_weights.json
metadata.json
```

---

## 10. Implementation Notes

Include practical coding guidance:

- Use `model.eval()`
- Use fixed checkpoints
- Use fixed seeds
- Explain one target at a time
- Keep the other modality fixed
- Avoid changing preprocessing
- Reuse dataset and model loading utilities from the codebase
- Save both figures and raw values
- Avoid running XAI on random unrepresentative samples

---

## 11. Risks and Mitigation

Include:

| Risk                   | Why it matters                    | Mitigation                                |
| ---------------------- | --------------------------------- | ----------------------------------------- |
| Attention overclaiming | Attention is not causality        | Present as token interaction evidence     |
| Wrong Grad-CAM layer   | Heatmap meaningless               | Attach to spatial feature map             |
| SHAP too slow          | High-dimensional fused embeddings | Use subset and modality-level aggregation |
| LIME instability       | Perturbation randomness           | Run multiple seeds or enough samples      |
| Multimodal confounding | Both modalities change            | Hold one modality fixed                   |
| Wrong target explained | Misleading figure                 | Always specify target index/name          |

---

## 12. Thesis Defense Talking Points

Add short answers to likely supervisor questions:

- Why XAI after training, not before?
- Why not explain all 21 experiments?
- Why use multiple XAI methods?
- Does attention prove causality?
- How do you know the heatmap is meaningful?
- How do you measure image vs text contribution?
- What if Grad-CAM and SHAP disagree?

---

## 13. Minimum Viable XAI Plan

If time is limited, define the minimum XAI deliverables:

```text
1. Single-sample demo
2. Grad-CAM for image evidence
3. Attention visualization for text evidence
4. SHAP modality contribution on 20–50 samples
5. 3–5 case studies
6. XAI summary report
```

---

## 14. Full XAI Plan

If time allows, define extended deliverables:

```text
LIME
Dataset-level SHAP aggregation
Failure case clustering
Target-wise explanation comparison
Modality-conflict analysis
```

---

# WRITING STYLE

Write in clear Markdown.

Language:

```text
Vietnamese
```

But keep technical terms in English when natural:

```text
Grad-CAM
Attention Visualization
SHAP
LIME
fusion
modality
checkpoint
embedding
feature map
case study
```

The document should be practical and implementation-ready.

---

# CONSTRAINTS

Do NOT hallucinate code details.

Do NOT invent file names unless clearly marked as proposed new files.

Do NOT assume the final best model unless verified from experiment results.

If something is uncertain, write:

```text
Must be verified in codebase.
```

Do NOT overclaim XAI as causality.

Do NOT say attention equals explanation.

Do NOT recommend running XAI for all 21 experiments.

Do NOT rewrite the project goal incorrectly as product-quality-only if the current codebase is restaurant review quality assessment.

---

# FINAL SELF-CHECK

Before finishing:

- Verify all target names are updated to 5 current targets.
- Verify old target names are removed or marked as historical.
- Verify architecture descriptions match current experiments.
- Verify XAI methods are mapped to correct model components.
- Verify XAI_proposal.md is phase-based.
- Verify the proposal is not too broad or overloaded.
- Verify the proposal can be used directly by an AI coding agent to implement XAI notebooks/scripts.

```

```
