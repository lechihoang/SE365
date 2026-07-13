# FINAL PRESENTATION PROPOSAL

## Explainable Multimodal Deep Learning for Vietnamese Restaurant Review Quality Assessment

## Presentation overview

- **Recommended length:** 20 slides, approximately 16–18 minutes plus demo/Q&A.
- **Narrative:** problem → five concrete contributions → controlled validation results → XAI/AI Agent demonstration → limitations → next steps.
- **Central claim:** the project contributes more than a trained model: it delivers a traceable Vietnamese multimodal research pipeline spanning Foody data construction, evidence-backed label engineering, controlled ablations, architecture-aligned XAI, and grounded explanation generation.
- **Primary experimental metric:** Mean MAE across five regression targets. All committed experiment values in this proposal are **validation** results.
- **Presentation style:** Vietnamese-first, minimal text, large numerical callouts, one visual per slide, and consistent colors for Text, Image, Fusion, XAI, and Agent layers.

The main deck should be confident about what is implemented and measured. Keep two caveats concise: the repository does not contain a locked-test metric artifact, and the current token–patch Cross-Attention implementation was introduced after the historical fusion/loss runs. Detailed provenance belongs in speaker notes, not in the visual center of the deck.

---

## Slide 1 — Explainable Multimodal Assessment of Vietnamese Restaurant Reviews

### Key message

From Vietnamese review text and restaurant images, the system predicts five quality scores and turns model evidence into a traceable explanation.

### On-slide content

- Input: one Vietnamese review + up to four images.
- Output: Food, Price, Atmosphere, Service, Overall Satisfaction.
- Pipeline: Prediction → XAI Evidence → AI Agent Report.

### Visual

A clean hero graphic showing a review and image set flowing into five score gauges, then an explanation card.

### Data / assets

- `Figures/Figure_1_1_Research_Value_Chain.png`
- `Figures/Figure_4_10_Prediction_XAI_AI_Agent_Sequence.png`
- Project title, team name, course, instructor, and presentation date.

### Speaker note

Open with the complete research value chain rather than a model name. Emphasize that prediction quality and explanation quality are separate responsibilities in the system.

---

## Slide 2 — The Problem: Five Scores from Two Imperfect Modalities

### Key message

Restaurant quality is multi-aspect, while review text and images provide complementary but incomplete evidence.

### On-slide content

- Text expresses opinion, context, price, and service experience.
- Images reveal food appearance, setting, and visual quality.
- One sample maps to a five-dimensional score vector on a 0–10 scale.
- The model must handle multiple images and modality disagreement.

### Visual

An input/output diagram with one review, three images, and five labeled regression outputs.

### Data / assets

- Target order from `src/dataset.py` and `xai/config.py`.
- Example review and images should be captured from the runtime demo, not fabricated.

### Speaker note

Frame this as multi-target regression, not sentiment classification. The difficult part is combining evidence that may be strong for one target and irrelevant or conflicting for another.

---

## Slide 3 — Contribution I: A Vietnamese Multimodal Restaurant Dataset

### Key message

The project converts noisy Foody data into a substantial, trainable Vietnamese multimodal dataset with an explicit cleaning trail.

### On-slide content

- Raw crawl: **300 restaurants, 11,111 reviews, 24,599 images**.
- After cleaning: **9,946 valid reviews** and **22,150 valid review–image pairs**.
- **6,082 reviews contain images** — **61.15%** image coverage.
- Training preparation yields **6,080 grouped multimodal samples**.

### Visual

A four-stage data funnel: raw crawl → valid reviews → reviews with images → grouped training samples.

### Data / assets

- `data_raw/cleaning_report.json`
- `data_raw/multimodal_reviews.csv`
- `data_processed/reviews_clean_enhanced.csv`
- `preprocess_data.py`
- `Figures/Figure_3_1_Dataset_Pipeline.png`

### Speaker note

The contribution is not just volume; it is the conversion from web data into review-level multimodal units. Current preprocessing retains 22,146 image records after required-field filtering and groups them into 6,080 samples.

---

## Slide 4 — Contribution II: Traceable Satisfaction Label Engineering

### Key message

Overall Satisfaction is engineered as an auditable weak label: four aspect scores plus explicit Vietnamese satisfaction evidence.

### On-slide content

- Base score: mean of Food, Service, Atmosphere, and Price.
- **14 rule categories:** 8 positive and 6 negative.
- **3,263 / 9,946 reviews (32.81%)** receive a non-zero adjustment.
- Every adjustment stores the matched rule and text evidence.
- Final value is clipped to the **0–10** score range.

### Visual

A formula card: four-aspect mean + traceable text adjustment → Overall Satisfaction, with one positive and one negative evidence example.

### Data / assets

- `data_processed/overall_satisfaction_rule_analysis.md`
- `data_processed/overall_satisfaction_rules.json`
- `data_processed/reviews_clean_enhanced.csv`

### Speaker note

This label is more defensible than an opaque synthetic score because each adjustment can be traced back to a named rule and matched phrase. Excluding Position changes at least 0.5 points for 1,044 reviews, supporting a construct focused on the depicted and described dining experience.

---

## Slide 5 — Contribution III: A Controlled Experiment Framework

### Key message

The project turns model selection into a controlled evidence chain rather than a collection of unrelated training runs.

### On-slide content

- **20 committed validation metric artifacts** with one five-target schema.
- Five comparison stages: modality, image backbone, text backbone, fusion, and loss.
- Five additional Phase 6 combination checks test whether winners compose.
- Deterministic 80/10/10 split logic with seed 42.
- Selection criterion: Mean MAE across all five targets.

### Visual

A sequential ablation ladder with one variable changing at each stage, ending in Phase 6 combination validation.

### Data / assets

- `metrics/*.json`
- `Trainer.py`
- `preprocess_data.py`
- `Figures/Figure_5_1_Controlled_Sequential_Ablation_Phases.png`
- `Figures/Figure_5_3_Promising_Combination_Validation.png`

### Speaker note

The current split-generation logic produces 4,864 train, 608 validation, and 608 test samples from 6,080 grouped reviews. The committed JSON files report validation metrics; the deck must not relabel them as test results.

---

## Slide 6 — Contribution IV: Architecture-Aligned Multi-Level XAI

### Key message

Five complementary XAI methods inspect the model at image, text, interaction, fused, and local-perturbation levels.

### On-slide content

- Grad-CAM: image regions linked to Overall Satisfaction.
- Word-level Attention: readable Vietnamese evidence, not raw subwords.
- Cross-Attention: both Token → Patch and Patch → Token interactions.
- SHAP: text-origin vs image-origin contribution after fusion.
- LIME: local sensitivity for the selected sample only.

### Visual

A five-layer XAI map aligned to the tensors each method explains.

### Data / assets

- `xai/gradcam_explainer.py`
- `xai/attention_explainer.py`
- `xai/shap_explainer.py`
- `xai/lime_explainer.py`
- `Figures/Figure_4_5_Multi_Level_XAI_Pipeline.png`

### Speaker note

No single method answers every question. The value is triangulation: region evidence, word evidence, cross-modal interaction, fused attribution, and local perturbation can support or contradict one another.

---

## Slide 7 — Contribution V: An Evidence-Grounded AI Agent

### Key message

The AI Agent is a controlled verbalization layer that organizes existing evidence; it never predicts or edits the five scores.

### On-slide content

- Loads available XAI artifacts and records missing evidence.
- Builds a reasoning graph before the language-model call.
- Compresses raw explanations into target-specific Top-K evidence.
- Validates schema, target coverage, grounding, and limitations.
- Produces Customer View and Technical View reports.

### Visual

A left-to-right Agent pipeline: Evidence Loader → Reasoning Graph → GPT-4o → Validator → two report views.

### Data / assets

- `agent/evidence_loader.py`, `agent/evidence_builder.py`
- `agent/reasoning.py`, `agent/prompt_builder.py`
- `agent/openai_client.py`, `agent/validator.py`
- `agent/report_generator.py`
- `Figures/Figure_4_7_Evidence_Based_AI_Agent_Pipeline.png`

### Speaker note

The design limits free-form generation by placing structured reasoning and evidence selection before GPT-4o and validation after it. At runtime the Agent runs only when `OPENAI_API_KEY` is available; otherwise the XAI demo continues cleanly.

---

## Slide 8 — Current Multimodal Architecture: PhoBERT ↔ Swin-B

### Key message

The current implementation performs real bidirectional Cross-Attention between Vietnamese text tokens and visual patch features.

### On-slide content

- PhoBERT encodes contextual token representations.
- Swin-B encodes patch-level visual representations across real images.
- Two 8-head attention blocks model Token → Patch and Patch → Token.
- Masked pooling forms a 1,024-dimensional fused representation.
- A shared MLP predicts five continuous scores.

### Visual

The token–patch Cross-Attention diagram with both directions clearly labeled.

### Data / assets

- `Models/TextModel.py`
- `Models/ImageModel.py`
- `Models/CrossAttentionFusion.py`
- `Figures/Figure_4_2_Cross_Attention.png`

### Speaker note

This is the current architecture in code. Keep it separate from the historical fusion/loss validation claims because those metrics were produced before the token–patch refactor.

---

## Slide 9 — Result I: Text Carries Most of the Signal; Images Add a Small Gain

### Key message

The multimodal baseline is best, but its gain over text-only is modest: Vietnamese review text remains the dominant predictive source.

### On-slide content

- Image-only ConvNeXt: **1.4949 Mean MAE**.
- Text-only XLM-R: **1.2434**.
- Multimodal ConvNeXt + XLM-R: **1.2385**.
- Multimodal improves **0.4%** over text-only and **17.2%** over image-only.

### Visual

A three-bar validation Mean MAE chart, lower is better, with the multimodal bar highlighted.

### Data / assets

- `metrics/metrics_EXP_010_text_only_xlmr_mse.json`
- `metrics/metrics_EXP_011_image_only_convnext_meanpool_mse.json`
- `metrics/metrics_EXP_012_multimodal_convnext_xlmr_concat_mse.json`

### Speaker note

This comparison is now supported by committed metric files. The honest interpretation is useful: images contribute complementary signal, but the aggregate improvement over a strong text baseline is small.

---

## Slide 10 — Result II: Swin-B Is the Strongest Image Backbone

### Key message

Under the same XLM-R + Concatenation + MSE setting, Swin-B gives the lowest validation error among the tested image encoders.

### On-slide content

- **Swin-B: 1.2169 Mean MAE**.
- SigLIP: 1.2296.
- EfficientNet-B3: 1.2800.
- Swin-B improves **1.0%** over SigLIP and **4.9%** over EfficientNet-B3.

### Visual

A ranked horizontal bar chart with Swin-B in the project accent color.

### Data / assets

- `metrics/metrics_EXP_020B_swinb_xlmr_concat_mse.json`
- `metrics/metrics_EXP_020E_siglip_xlmr_concat_mse.json`
- `metrics/metrics_EXP_020D_efficientnetb3_xlmr_concat_mse.json`

### Speaker note

Only the image backbone changes in this phase. Use Mean MAE as the comparison criterion and avoid cluttering the slide with all per-target values.

---

## Slide 11 — Result III: PhoBERT Produces the Largest Sequential Gain

### Key message

Choosing a Vietnamese-specific text backbone has a much larger effect than later fusion or loss refinements.

### On-slide content

- **PhoBERT: 1.1145 Mean MAE**.
- XLM-R: 1.2169.
- ViSoBERT: 1.2328.
- PhoBERT reduces error by **8.4%** versus XLM-R.
- This is the strongest improvement in the sequential ablation chain.

### Visual

A three-point lollipop chart with a large “−8.4%” callout between XLM-R and PhoBERT.

### Data / assets

- `metrics/metrics_EXP_030B_bestimage_phobert_concat_mse.json`
- `metrics/metrics_EXP_020B_swinb_xlmr_concat_mse.json`
- `metrics/metrics_EXP_030D_bestimage_visobert_concat_mse.json`

### Speaker note

This is the clearest experimental result in the deck. It supports the project’s focus on Vietnamese language representation while remaining a within-protocol validation comparison.

---

## Slide 12 — Result IV: Fusion Helps, but the Top Methods Are Nearly Tied

### Key message

Cross-Attention records the lowest historical Mean MAE, yet its numerical advantage over Gated Cross-Modal fusion is only 0.03%.

### On-slide content

- **Cross-Attention: 1.1079 Mean MAE**.
- Gated Cross-Modal: 1.1082; Concatenation: 1.1145.
- GMU: 1.1160; FiLM: 1.1195.
- Cross-Attention improves **0.6%** over Concatenation.
- Prefer it for interaction modeling and XAI value, not a large accuracy claim.

### Visual

A zoomed ranked dot plot of the five validation Mean MAE values with the top-two gap annotated.

### Data / assets

- `metrics/metrics_EXP_030B_bestimage_phobert_concat_mse.json`
- `metrics/metrics_EXP_040B_bestimage_besttext_gmu_mse.json`
- `metrics/metrics_EXP_040C_bestimage_besttext_gatedcrossmodal_mse.json`
- `metrics/metrics_EXP_041A_bestimage_besttext_film_mse.json`
- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`

### Speaker note

These fusion metrics use the historical pre-refactor Cross-Attention semantics. The current token–patch architecture requires a fresh controlled rerun before the same value can be attributed to it.

---

## Slide 13 — Result V: Loss Choice Does Not Materially Change Mean MAE

### Key message

All four loss strategies converge to almost the same aggregate validation error; there is no honest large winner.

### On-slide content

- **MSE: 1.1079 Mean MAE** — best aggregate value.
- Log-Cosh: 1.1080 — best Overall MAE at **0.9130**.
- Uncertainty weighting: 1.1080 — best Overall R² at **0.6337**.
- Huber: 1.1085.
- Maximum Mean MAE spread: **0.0007**.

### Visual

A compact heatmap table with rows for losses and columns for Mean MAE, Overall MAE, and Overall R².

### Data / assets

- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`
- `metrics/metrics_EXP_050B_bestfusion_huber.json`
- `metrics/metrics_EXP_050C_bestfusion_logcosh.json`
- `metrics/metrics_EXP_051D_bestfusion_uncertaintyweighted.json`

### Speaker note

Do not compare raw loss values across different objectives. The measured conclusion is that evaluation metrics are effectively tied; multi-seed analysis would be needed to establish a reliable ordering.

---

## Slide 14 — Result VI: Good Components Do Not Automatically Form a Better Combination

### Key message

Phase 6 confirms that the controlled sequential winner remains stronger than alternative mixtures of individually promising components.

### On-slide content

- **EXP_060A sequential configuration: 1.1080 Mean MAE**.
- EXP_060E: 1.1248; EXP_060C: 1.1256.
- EXP_060B: 1.2300; EXP_060D: 1.2829.
- Component interactions matter; “best + best” is not automatically best.

### Visual

A ranked five-bar chart for the Phase 6 configurations, with EXP_060A highlighted.

### Data / assets

- `metrics/metrics_EXP_060A_bestsequential_full_configuration.json`
- `metrics/metrics_EXP_060B_swinb_visobert_gmu_uncertainty.json`
- `metrics/metrics_EXP_060C_efficientnetb3_phobert_film_huber.json`
- `metrics/metrics_EXP_060D_efficientnetb3_visobert_crossattention_logcosh.json`
- `metrics/metrics_EXP_060E_convnext_phobert_gatedcrossmodal_autoweight.json`

### Speaker note

These newly committed artifacts complete the planned combination comparison. Cross-Attention combinations in this family should still be rerun after migration before being used as evidence for the current token–patch implementation.

---

## Slide 15 — Best Recorded Validation Profile

### Key message

The lowest recorded aggregate validation error is 1.1079 Mean MAE, with Overall Satisfaction predicted most accurately.

### On-slide content

- Configuration: Swin-B + PhoBERT + Cross-Attention + MSE.
- **Mean MAE: 1.1079**; Overall R²: **0.6335**.
- Food: 1.1024; Price: 1.1728.
- Atmosphere: 1.1743; Service: 1.1756.
- **Overall Satisfaction: 0.9143 MAE**.

### Visual

A five-bar per-target MAE chart with a large 1.1079 aggregate callout.

### Data / assets

- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`

### Speaker note

Call this the “best recorded historical validation run,” not a locked-test result and not a post-refactor token–patch result. The target profile shows that Overall Satisfaction is easier than the four aspect scores in this run.

---

## Slide 16 — XAI Demo: One Prediction, Five Complementary Views

### Key message

The demo should tell one coherent evidence story, not display a gallery of disconnected explanation plots.

### On-slide content

- Select three distinct cases: accurate, error/conflict, and evidence-rich.
- Show **Overall Satisfaction Grad-CAM only** for visual focus.
- Display merged word-level attention plus both Cross-Attention directions.
- Label SHAP as text-origin/image-origin after fusion, not pure modalities.
- Present LIME explicitly as a local explanation for the current sample.

### Visual

One composite case dashboard: review/images, prediction vs ground truth, Grad-CAM, readable words, bidirectional token–patch evidence, and compact SHAP/LIME summaries.

### Data / assets

- `Success_End_to_End_XAI_AI_Agent.ipynb`
- `xai/case_study.py`
- Runtime artifacts under the configured experiment `xai/` and demo output directories.
- `Figures/Figure_4_5_Multi_Level_XAI_Pipeline.png` as fallback if runtime capture is unavailable.

### Speaker note

The notebook contains robust three-case selection and safe per-method fallbacks, but it has no committed cell outputs. Capture the final composite from a successful Colab run before building the PowerPoint; until then, use the pipeline figure rather than invented examples.

---

## Slide 17 — AI Agent Demo: From Evidence to Two Audience-Specific Reports

### Key message

The Agent converts fixed predictions and XAI evidence into a structured explanation while exposing missing evidence and validation warnings.

### On-slide content

- Prediction scores remain unchanged throughout the Agent pipeline.
- Reasoning graph records support, conflict, and missing evidence.
- Validator checks all five targets and SHAP grounding.
- Customer View is concise; Technical View preserves provenance and limitations.
- GPT-4o runs only when `OPENAI_API_KEY` is available.

### Visual

A split-screen runtime capture: Customer View on the left, Technical View with evidence completeness on the right.

### Data / assets

- `Success_End_to_End_XAI_AI_Agent.ipynb`
- `agent/output_schema.py`, `agent/validator.py`, `agent/report_generator.py`
- `Figures/Figure_4_7_Evidence_Based_AI_Agent_Pipeline.png`
- Runtime-generated Agent Markdown/JSON report; none is currently committed.

### Speaker note

If the API key is unavailable during the presentation, the notebook skips the Agent without breaking the XAI pipeline. Use a pre-captured report only after it has been generated from the same selected sample and artifact set.

---

## Slide 18 — Limitations: What the Current Evidence Does Not Yet Prove

### Key message

The system is technically broad and experimentally promising, but final generalization and explanation quality still require stronger evaluation.

### On-slide content

- No committed locked-test metrics, predictions, or split snapshot.
- Historical Cross-Attention metrics pre-date the current token–patch refactor.
- Rule-based weak labels retain phrase, negation, and discourse noise.
- XAI shows model association or local sensitivity, not causality.
- No committed XAI/Agent runtime package or human evaluation yet.

### Visual

A five-item “evidence boundary” panel, each limitation paired with its required validation action.

### Data / assets

- `data_processed/overall_satisfaction_rule_analysis.md`
- Git history for `Models/CrossAttentionFusion.py` and `metrics/`
- `xai/config.py` for the remaining 1–10 versus data 0–10 range inconsistency.
- Notebook metadata showing zero executed cells and zero outputs.

### Speaker note

Keep this slide factual and brief. Also note that `xai/config.py` currently declares a 1–10 range although source labels include 0; this should be corrected before final artifact generation.

---

## Slide 19 — Future Work: Turn the Pipeline into Reproducible Final Evidence

### Key message

The next priority is not another architecture—it is a versioned, statistically defensible evaluation of the current system.

### On-slide content

- Retrain the token–patch model with multiple seeds.
- Freeze and version train/validation/test splits and checkpoint hashes.
- Run one locked-test evaluation with prediction-level artifacts.
- Generate complete XAI case packages and conduct human evaluation.
- Test missing-modality robustness and restaurant-level leakage controls.

### Visual

A short roadmap with three milestones: reproducibility → final evaluation → explanation validation.

### Data / assets

- `Figures/Figure_7_1_Proposed_Deployment_Architecture.png`
- `Trainer.py`, `test.py`, and planned `test_metrics.json` / predictions artifacts.

### Speaker note

Fix the score-range inconsistency before producing final explanations. Human evaluation should assess usefulness, faithfulness, clarity, and whether the Agent stays grounded in the supplied evidence.

---

## Slide 20 — Takeaway: A Complete Research Pipeline, Not Just a Model

### Key message

The project’s strongest contribution is an end-to-end, traceable Vietnamese multimodal research system—from data and labels to controlled results and human-readable evidence.

### On-slide content

- **Data:** 9,946 valid reviews; 22,150 valid review–image pairs.
- **Labels:** 14 evidence-bearing rules; 3,263 adjusted reviews.
- **Experiments:** 20 validation artifacts; best recorded Mean MAE **1.1079**.
- **Explanation:** five XAI views plus a grounded reporting Agent.
- Final message: prediction, evidence, and reporting remain explicitly separated.

### Visual

A four-pillar closing graphic—Dataset, Labels, Experiments, Explanations—feeding one traceable research pipeline.

### Data / assets

- `Figures/Figure_1_1_Research_Value_Chain.png`
- The four headline statistics above from the listed dataset, label, and metric artifacts.

### Speaker note

Close on system-level value rather than a marginal metric difference. The project demonstrates how a Vietnamese multimodal prediction task can be made measurable, explainable, and ready for stronger final validation.

---

## Compact metric and source summary

All values below are validation Mean MAE unless otherwise stated; lower is better.

| Comparison | Values to present | Main conclusion | Exact source |
|---|---|---|---|
| Modality | Text 1.2434; Image 1.4949; Multimodal 1.2385 | Multimodal is best, but only 0.4% better than text-only | `metrics_EXP_010`, `011`, `012` JSONs |
| Image backbone | Swin-B 1.2169; SigLIP 1.2296; EfficientNet-B3 1.2800 | Swin-B wins the controlled image phase | `metrics_EXP_020B`, `020E`, `020D` JSONs |
| Text backbone | PhoBERT 1.1145; XLM-R 1.2169; ViSoBERT 1.2328 | PhoBERT gives the largest sequential gain: 8.4% vs XLM-R | `metrics_EXP_030B`, `020B`, `030D` JSONs |
| Fusion | Cross-Attention 1.1079; Gated 1.1082; Concat 1.1145; GMU 1.1160; FiLM 1.1195 | Cross-Attention is lowest historically; top two are nearly tied | `metrics_EXP_041B`, `040C`, `030B`, `040B`, `041A` JSONs |
| Loss | MSE 1.1079; Log-Cosh 1.1080; Uncertainty 1.1080; Huber 1.1085 | Aggregate differences are negligible | `metrics_EXP_041B`, `050C`, `051D`, `050B` JSONs |
| Phase 6 | 060A 1.1080; 060E 1.1248; 060C 1.1256; 060B 1.2300; 060D 1.2829 | Alternative combinations do not beat the sequential configuration | Five `metrics_EXP_060*.json` files |
| Best recorded profile | Mean 1.1079; Overall MAE 0.9143; Overall R² 0.6335 | Best historical aggregate validation result | `metrics_EXP_041B_bestimage_besttext_crossattention_mse.json` |

Dataset headline sources:

- Raw and cleaned counts: `data_raw/cleaning_report.json`.
- Label rules and coverage: `data_processed/overall_satisfaction_rule_analysis.md` and `overall_satisfaction_rules.json`.
- Grouped sample construction and deterministic split logic: `preprocess_data.py` plus the two current source CSVs.
- Metric semantics: `Trainer.validate()` and committed `metrics/*.json` files.

Do not mix older progress-report test tables with this validation series; their backbones, target definitions, and protocols differ.

---

## Visual asset checklist

| Priority | Asset | Source / creation instruction | Status before deck generation |
|---|---|---|---|
| P0 | Title value-chain graphic | Adapt `Figure_1_1_Research_Value_Chain.png` | Available |
| P0 | Problem input/output diagram | Create from target schema; use a real runtime sample | Create |
| P0 | Dataset funnel | Rebuild from `cleaning_report.json` and preprocessing counts | Create |
| P0 | Label formula/evidence card | Create from rule analysis; no long rule table | Create |
| P0 | Controlled ablation ladder | Simplify `Figure_5_1_Controlled_Sequential_Ablation_Phases.png` | Available / simplify |
| P0 | Current token–patch architecture | Use `Figure_4_2_Cross_Attention.png` | Available |
| P0 | Six result charts | Generate directly from the 20 metric JSON files | Create programmatically |
| P0 | Best model per-target bars | Generate from EXP_041B JSON | Create programmatically |
| P0 | XAI case dashboard | Capture from a successful end-to-end Colab run | Runtime required |
| P0 | Agent two-view report | Capture from the same sample and evidence package | Runtime/API key required |
| P1 | Limitation boundary panel | Create as five icon/action pairs | Create |
| P1 | Closing four-pillar graphic | Adapt the research value chain | Create |

Use a consistent rule for chart precision: four decimals in source notes, four decimals for close model comparisons, and no more than two decimals for percentages. Every chart must state **Validation Mean MAE — lower is better**.

---

## Short validation checklist

- [ ] Exactly 20 slides; each slide communicates one message and uses one main visual.
- [ ] Every experimental chart is generated from `metrics/*.json`, not copied from an older report.
- [ ] All experiment results are labeled **validation**, never test.
- [ ] The best value is described as “best recorded historical validation,” not current locked-test performance.
- [ ] The current token–patch architecture is not assigned pre-refactor metrics.
- [ ] Dataset scale is consistent: 9,946 valid reviews, 22,150 valid pairs, 6,080 grouped trainable samples.
- [ ] Score range is presented as 0–10; fix `xai/config.py` before final runtime artifact generation.
- [ ] Grad-CAM visible output is Overall Satisfaction only; Attention is merged to readable words.
- [ ] Cross-Attention shows both directions; SHAP origins are not described as pure modalities; LIME is called local.
- [ ] XAI and Agent screenshots are used only after real runtime artifacts exist for the same sample.
- [ ] GPT-4o is described only as an optional evidence verbalizer; it never predicts or modifies scores.
- [ ] Limitations and future work remain concise, specific, and action-oriented.
