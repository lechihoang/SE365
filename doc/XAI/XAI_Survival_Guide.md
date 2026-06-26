# 1. Executive Summary (One Page)

- XAI exists because prediction scores alone do not prove that the model used valid restaurant-review evidence.
- Your model must explain four things: image evidence, text evidence, modality contribution, and local robustness.
- One XAI method is not enough because each method observes a different object inside the architecture.
- `Grad-CAM` answers: where in the image did visual evidence support a selected score?
- `Attention Visualization` answers: which tokens interacted strongly inside the text encoder?
- `SHAP` answers: which fused features or modalities contributed to the final regression output?
- `LIME` answers: what happens locally when image regions or words are removed?
- Multimodal explanation must mirror multimodal architecture: image branch, text branch, fusion layer, prediction head.
- Grad-CAM must attach to a spatial feature map from the image encoder (e.g., Swin-B, ConvNeXt, EfficientNet-B3), not the pooled embedding vector.
- Attention is useful but dangerous: it shows token interaction, not guaranteed causal importance.
- SHAP is strongest for fusion-level contribution, especially image-vs-text dominance.
- LIME is strongest as a local perturbation sanity check, not as global evidence.
- Every explanation must target one output head: `food_score`, `price_score`, `atmosphere_score`, `service_score`, or `overall_satisfaction`.
- The system predicts 5 targets, not a single score. Each target captures a different aspect of restaurant review quality.
- Never compare explanations unless they are generated for the same sample, same target score, same fixed modality.
- For image XAI, hold text fixed; for text XAI, hold image fixed.
- For SHAP, use representative background fused embeddings, not arbitrary samples.
- For LIME, run multiple seeds or enough perturbations because explanations can be unstable.
- Save raw explanation values, not only pretty figures.
- The system supports multiple image and text backbones, multiple fusion methods, and multiple loss functions. The best-performing configuration is Swin-B + PhoBERT + Cross-Attention + LogCosh.
- Strong thesis claim: the system is not fully transparent, but its reasoning is inspected at multiple levels.
- Defense-safe sentence: "Each method is partial; the strength comes from aligned, multi-level evidence."

# 2. The 20 Things I Must Never Forget

## #1

**Concept:** Attention is not explanation.

**Why it matters:** Text encoder attention weights show token-to-token information flow, not guaranteed final decision importance. This applies whether using XLM-R, PhoBERT, or ViSoBERT.

**What happens if I forget it:** You overclaim and examiners will attack your interpretation.

**Defense question that may appear:** "Does high attention mean the word caused the prediction?"

**Strong answer:** "No. Attention indicates interaction strength inside the transformer. I use it as text-branch evidence, but I validate importance with perturbation methods such as LIME or contribution methods such as SHAP."

## #2

**Concept:** Grad-CAM requires spatial feature maps.

**Why it matters:** Grad-CAM needs `[B,C,H,W]` activations and gradients. All image encoders used in this system (ConvNeXt, Swin-B, EfficientNet-B3, SigLIP) produce spatial feature maps before global pooling, making them Grad-CAM compatible.

**What happens if I forget it:** You attach it to the pooled `image_embedding [B, image_dim]` and produce meaningless localization.

**Defense question that may appear:** "Where exactly did you attach Grad-CAM?"

**Strong answer:** "To the last spatial feature map of the image encoder before global pooling, not to the pooled embedding. For example, with Swin-B this is the output of the final stage; with ConvNeXt it is the last convolutional feature map."

## #3

**Concept:** Explain one target score at a time.

**Why it matters:** The model predicts 5 regression outputs (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`), each with different semantics.

**What happens if I forget it:** You explain `overall_satisfaction` while discussing `price_score`.

**Defense question that may appear:** "Which output did this heatmap explain?"

**Strong answer:** "This explanation was generated from the scalar target `price_score`; the other heads were not used for that explanation."

## #4

**Concept:** Hold the other modality fixed.

**Why it matters:** Multimodal explanations become confounded if both image and text change.

**What happens if I forget it:** You cannot say whether the observed change came from image or text.

**Defense question that may appear:** "How did you isolate the image branch?"

**Strong answer:** "For image explanations, I kept the text input constant and varied only the image evidence; for text explanations, I fixed the image."

## #5

**Concept:** SHAP explains contribution, not localization.

**Why it matters:** Fusion SHAP can quantify image/text influence but cannot show specific food presentation details or restaurant interior features.

**What happens if I forget it:** You claim SHAP 'found the food quality issue' when it only measured embedding contribution.

**Defense question that may appear:** "Can SHAP tell where the visual evidence is in the photo?"

**Strong answer:** "Not directly at the fusion level. Grad-CAM localizes image evidence; SHAP quantifies contribution of fused features or modalities."

## #6

**Concept:** LIME is local, not global.

**Why it matters:** LIME explains behavior near one sample.

**What happens if I forget it:** You generalize one case study to the full dataset.

**Defense question that may appear:** "Does this LIME result prove the model always uses price words?"

**Strong answer:** "No. It shows local behavior for this sample. Dataset-level conclusions require repeated analysis across many samples."

## #7

**Concept:** Background choice controls SHAP meaning.

**Why it matters:** SHAP values are relative to a baseline expectation.

**What happens if I forget it:** Contributions become unstable or misleading.

**Defense question that may appear:** "What is the SHAP base value?"

**Strong answer:** "It is the expected model output over the background set; SHAP explains how the sample moves away from that baseline."

## #8

**Concept:** The image encoder must preserve late spatial maps for Grad-CAM.

**Why it matters:** All image backbones in this system (ConvNeXt, Swin-B, EfficientNet-B3, SigLIP) are loaded via `timm` (`Models/ImageModel.py`) and produce spatial feature maps before pooling, making Grad-CAM applicable.

**What happens if I forget it:** You cannot justify Grad-CAM for your image branch.

**Defense question that may appear:** "Why can you use Grad-CAM with your image encoder?"

**Strong answer:** "The image encoder produces spatial feature maps before global pooling. Gradients from a target score can weight those maps for localization, regardless of which specific timm backbone is used."

## #9

**Concept:** Text tokenization must be inspected, especially for Vietnamese.

**Why it matters:** Vietnamese words may split into subwords. The system supports XLM-R, PhoBERT (`vinai/phobert-base-v2`), and ViSoBERT (`uitnlp/visobert`), each with different tokenization strategies. PhoBERT uses syllable-level BPE while XLM-R uses SentencePiece; ViSoBERT is also BPE-based.

**What happens if I forget it:** Your attention or LIME text visualization becomes unreadable or wrong.

**Defense question that may appear:** "Are these words or subword tokens?"

**Strong answer:** "The model uses subword tokens internally, but for human-facing analysis I inspect and aggregate subwords where appropriate. The tokenization behavior depends on which text backbone is selected."

## #10

**Concept:** Explanations must survive sanity checks.

**Why it matters:** Pretty plots can be false confidence.

**What happens if I forget it:** You defend visualizations that do not depend on the trained model.

**Defense question that may appear:** "How do you know the heatmap is meaningful?"

**Strong answer:** "I compare heads, inspect gradients, check failure cases, and use randomization or perturbation sanity checks where possible."

## #11

**Concept:** Multi-method disagreement is evidence, not embarrassment.

**Why it matters:** Grad-CAM, LIME, SHAP, and attention measure different things.

**What happens if I forget it:** You hide important model behavior.

**Defense question that may appear:** "What if Grad-CAM and LIME disagree?"

**Strong answer:** "I treat disagreement as diagnostic. It may reveal instability, branch imbalance, or differences between internal gradient evidence and perturbation response."

## #12

**Concept:** Fusion SHAP is your best modality-balance tool.

**Why it matters:** Your central research question is multimodal contribution.

**What happens if I forget it:** You cannot prove whether image or text dominates.

**Defense question that may appear:** "How do you measure modality dominance?"

**Strong answer:** "I aggregate absolute SHAP values over the image dimensions `0:image_dim` and text dimensions `image_dim:image_dim+text_dim`. The exact split depends on the backbone choice. For example, with Swin-B (`image_dim=1024`) + PhoBERT (`text_dim=768`), image features occupy `0:1024` and text features occupy `1024:1792` in the concatenated fusion vector."

## #13

**Concept:** Signed and absolute SHAP answer different questions.

**Why it matters:** Magnitude shows influence; sign shows direction.

**What happens if I forget it:** You confuse 'important' with 'positive.'

**Defense question that may appear:** "Did text help or hurt the prediction?"

**Strong answer:** "I use signed SHAP to determine direction and absolute SHAP to measure contribution magnitude."

## #14

**Concept:** LIME requires careful regression adaptation.

**Why it matters:** LIME often expects classification-like output.

**What happens if I forget it:** Your prediction wrapper returns incompatible or poorly calibrated values.

**Defense question that may appear:** "How did you apply LIME to regression?"

**Strong answer:** "I explained one regression target through a wrapper, either returning the scalar appropriately or mapping it into a two-column pseudo-output for the explainer."

## #15

**Concept:** `model.eval()` is mandatory during explanation.

**Why it matters:** Dropout and training-time behavior make explanations non-reproducible.

**What happens if I forget it:** Same sample produces different explanations.

**Defense question that may appear:** "How did you ensure reproducibility?"

**Strong answer:** "I used a fixed checkpoint, `eval()` mode, fixed background sets, saved seeds, and stored raw explanation values."

## #16

**Concept:** Raw figures are not enough.

**Why it matters:** Thesis evidence must be auditable.

**What happens if I forget it:** You cannot reproduce or quantify the visual claims.

**Defense question that may appear:** "Can you compare explanations numerically?"

**Strong answer:** "Yes. I save heatmaps, SHAP vectors, modality percentages, LIME weights, and metadata alongside figures."

## #17

**Concept:** Explanation quality depends on model quality.

**Why it matters:** Explaining a bad model does not make it scientific.

**What happens if I forget it:** You decorate poor predictions with XAI plots.

**Defense question that may appear:** "Why should we trust explanations if the model is inaccurate?"

**Strong answer:** "I interpret explanations together with predictive performance and error analysis; XAI is not a substitute for model validity."

## #18

**Concept:** Do not over-interpret embedding dimensions.

**Why it matters:** A single dimension in the fused vector may not map to a human concept. The fusion vector size varies by backbone (e.g., 1792 for Swin-B + PhoBERT, 1536 for ConvNeXt + XLM-R).

**What happens if I forget it:** You invent unsupported semantic stories.

**Defense question that may appear:** "What does dimension 914 mean?"

**Strong answer:** "I do not assign direct semantics to isolated dimensions unless supported by probing. I mainly use grouped and cross-sample contribution patterns."

## #19

**Concept:** Explanation must be per-sample and per-head.

**Why it matters:** `food_score`, `price_score`, `atmosphere_score`, `service_score`, and `overall_satisfaction` should not have identical reasoning.

**What happens if I forget it:** You miss branch collapse or head collapse.

**Defense question that may appear:** "Why do all heads have similar explanations?"

**Strong answer:** "If they are identical, that is a warning sign. I would inspect head specialization, labels, loss weighting, and fusion behavior."

## #20

**Concept:** XAI does not prove causality.

**Why it matters:** Most methods are post hoc evidence.

**What happens if I forget it:** You make indefensible claims.

**Defense question that may appear:** "Does Grad-CAM prove the food presentation caused the score?"

**Strong answer:** "No. It shows target-linked supportive regions. Causal claims require controlled perturbation, counterfactuals, or stronger experimental design."

# 3. XAI Mental Model

| Question | Correct XAI Tool | Why |
| -------- | ---------------- | --- |
| Where did the model look in the food/restaurant photo? | Grad-CAM | Uses gradients over the image encoder's spatial feature maps. |
| Which image regions locally affect the score if removed? | LIME Image | Perturbs superpixels and observes prediction change. |
| Which words interact strongly in the review? | Attention Visualization | Shows text encoder token-to-token attention patterns. |
| Which words locally affect the score if removed? | LIME Text | Perturbs words while holding the image fixed. |
| Which modality dominates the final score? | SHAP on fusion embedding | Aggregates contribution over image/text embedding segments. |
| Did text raise or lower the predicted score? | Signed SHAP | Directional contribution relative to baseline. |
| How much did image/text matter regardless of direction? | Absolute SHAP | Measures contribution magnitude. |
| Is the model using background or watermark instead of food? | Grad-CAM + LIME Image | Localization plus perturbation check. |
| Is price text affecting `price_score` more than `atmosphere_score`? | LIME Text + SHAP per head | Tests target-specific behavior. |
| Are explanations stable for this sample? | LIME repeated seeds + SHAP background sensitivity | Checks robustness. |
| Are all heads using the same evidence? | Grad-CAM/SHAP per output head | Detects head collapse. |
| Did fusion ignore one branch? | SHAP modality contribution + ablation | Measures dominance and confirms with removal. |
| What happens if visual evidence is removed? | LIME Image or deletion test | Measures local prediction response. |
| What happens if text evidence is removed? | LIME Text or token deletion | Measures local text sensitivity. |
| Can I explain raw pixels with fusion SHAP? | No; use Grad-CAM/LIME Image | Fusion SHAP explains embeddings, not pixel locations. |
| Which of the multiple review images matters most? | Grad-CAM per image + ablation | The system handles up to 4 images per review with masked average pooling; explain each image separately. |

# 4. Implementation Cheat Sheet

## Grad-CAM

**Input:** `pixel_values [B, N, 3, H, W]` (up to 4 images per review), fixed `input_ids`, fixed `attention_mask`, chosen `score_index` (0-4).

**Output:** Heatmap `[Hf,Wf]` upsampled to image size; overlay on original image.

**Attach to:** Last spatial feature map of the image encoder before global pooling, e.g. `[B*N, C, Hf, Wf]` where C depends on the backbone (1024 for Swin-B, 768 for ConvNeXt-base, 1536 for EfficientNet-B3).

**Never attach to:** Pooled `image_embedding [B, image_dim]`, logits alone, post-pooling vector, or fusion embedding.

**Multi-image note:** The system processes up to 4 images per review (`Models/ImageModel.py`). For Grad-CAM, select which image to explain and generate a separate heatmap for it. The `num_images` mask determines which images are real vs. padding.

**Common bug:** Explaining the wrong score head or using a layer with no spatial structure.

**Sanity check:** Different target heads should sometimes produce different heatmaps; randomized model weights should degrade explanations.

**Expected visualization:** Coarse regions over food presentation, plating, restaurant interior, decor, or table setting; not exact pixel boundaries.

## Attention Visualization

**Input:** Tokenized review with `output_attentions=True`.

**Output:** Attention tensors per layer: `[B,num_heads,L,L]`; heatmap over tokens.

**Attach to:** Text encoder attention outputs (XLM-R, PhoBERT, or ViSoBERT depending on configuration).

**Never attach to:** Final text embedding and pretend it is token-level attention.

**Common bug:** Treating `<s>`, `</s>`, `<pad>`, or subwords as direct human evidence. PhoBERT uses `<s>`/`</s>` while XLM-R also uses `<s>`/`</s>`.

**Sanity check:** Inspect tokenization first; compare last-layer mean, last-four-layer mean, and selected heads.

**Expected visualization:** Strong interactions around food quality, price, atmosphere, and service terms, contrast words such as "nhung" (but), and sentiment-bearing tokens like "ngon" (delicious), "dat" (expensive), "dep" (beautiful).

## SHAP

**Input:** Fused embedding `[B, text_dim + image_dim]` or background/sample fused embeddings. The fusion vector size depends on the backbone choice (e.g., 1792 for Swin-B + PhoBERT, 1536 for ConvNeXt + XLM-R).

**Output:** SHAP values `[B, text_dim + image_dim]` per target output; grouped image/text contribution.

**Attach to:** Fusion MLP or wrapper mapping fused embedding to a selected score. In the concat fusion model (`Models/FusionModel.py`), this is the `fusion_fc` + `factor_head` path. For other fusion types (GMU, Gated Cross-Modal, FiLM, Cross-Attention), attach to the `head` module.

**Never attach to:** Raw token ids as if numeric ids have semantic distance; raw pixels unless using a carefully designed image SHAP setup.

**Common bug:** Bad background set or interpreting individual embedding dimensions as human concepts.

**Sanity check:** Verify additive relation approximately: prediction ~= base value + sum SHAP values; compare per target head.

**Expected visualization:** Waterfall/force plot for one sample; grouped bar showing image vs text contribution.

## LIME

**Input:** Image array `[H,W,3]` or raw text; fixed complementary modality; prediction wrapper for one score.

**Output:** Superpixel weights or word weights for one local sample.

**Attach to:** Full multimodal prediction function, not internal tensors.

**Never attach to:** Both modalities perturbed simultaneously if you want isolated modality explanation.

**Common bug:** Too few samples, unstable segmentation, or invalid regression wrapper.

**Sanity check:** Repeat with multiple seeds; increase perturbation samples; confirm known words/regions affect the expected score.

**Expected visualization:** Positive/negative superpixels or words; local evidence, not global truth.

# 5. Defense Questions That Examiners Love

| Question | Why examiner asks it | Strong answer | Weak answer |
|---|---|---|---|
| Why do you need XAI if accuracy is good? | Tests research maturity. | Accuracy tells how often; XAI examines why and detects shortcuts. | Because XAI looks nice. |
| Why four XAI methods? | Tests method justification. | Each explains a different architectural level. | More methods are better. |
| Why is attention not explanation? | Tests theory. | It shows token interaction, not guaranteed causal contribution. | Attention shows important words. |
| Where is Grad-CAM attached? | Tests implementation. | Last spatial feature map of the image encoder before pooling (e.g., Swin-B final stage output). | To the encoder output. |
| Why not attach Grad-CAM to the pooled embedding? | Tests tensor understanding. | It has no spatial dimensions; `[B, image_dim]` cannot produce a heatmap. | It is the final feature. |
| What does Grad-CAM compute? | Tests math. | Gradient-weighted sum of feature maps for a target score. | A heatmap of important pixels. |
| Does Grad-CAM prove causality? | Tests overclaiming. | No, it gives target-linked supportive evidence. | Yes, red areas caused it. |
| Why this image encoder? | Tests architecture choice. | Strong visual encoder with spatial maps usable for Grad-CAM; backbone chosen through systematic experiments comparing ConvNeXt, Swin-B, EfficientNet-B3, and SigLIP. | It is modern. |
| Why this text encoder? | Tests text choice. | Suitable for Vietnamese text; backbone chosen through experiments comparing XLM-R, PhoBERT, and ViSoBERT. PhoBERT is optimized for Vietnamese with syllable-level BPE. | It performs well. |
| What is attention tensor shape? | Tests implementation detail. | Tuple per layer, each `[B,H,L,L]`. | A matrix of words. |
| How handle Vietnamese subwords? | Tests NLP rigor. | Inspect tokenization and aggregate subwords for human plots. PhoBERT tokenizes at syllable level, requiring different aggregation than XLM-R. | Ignore them. |
| Why SHAP at fusion? | Tests multimodal reasoning. | Fusion directly represents image/text combination with a manageable `[text_dim + image_dim]` input (e.g., 1792 for the best model). | SHAP can explain anything. |
| What is SHAP base value? | Tests math. | Expected model output over background data. | Average score. |
| How compute modality contribution? | Tests method. | Sum absolute SHAP over image dimensions `0:image_dim` and text dimensions `image_dim:image_dim+text_dim`. For the best model (Swin-B + PhoBERT): image `0:1024`, text `1024:1792`. | Count important features. |
| Signed vs absolute SHAP? | Tests nuance. | Signed shows direction; absolute shows magnitude. | They are the same. |
| Why not use SHAP only? | Tests scope. | SHAP does not localize visual regions or show token interactions. | SHAP is enough. |
| Why use LIME if you have Grad-CAM? | Tests complementarity. | LIME tests local perturbation response in human units. | To add another plot. |
| What does local explanation mean? | Tests LIME theory. | Valid near one sample, not globally. | It explains the whole model. |
| How adapt LIME to regression? | Tests implementation. | Use target-specific wrapper or pseudo-output mapping. | Use default classifier. |
| What if LIME is unstable? | Tests validation. | Increase samples, repeat seeds, inspect segmentation. | Choose the best-looking result. |
| What if Grad-CAM and LIME disagree? | Tests scientific honesty. | Treat as diagnostic; investigate instability or branch behavior. | Pick Grad-CAM. |
| How know image branch is not ignored? | Tests multimodal validity. | SHAP modality contribution, ablation, and performance drop without image. | It is in the architecture. |
| How know text branch is not ignored? | Tests multimodal validity. | Text ablation, SHAP text contribution, LIME text sensitivity. | Attention has colors. |
| What is branch collapse? | Tests failure awareness. | One modality dominates while the other contributes little. | When training fails. |
| Why explain each output separately? | Tests regression-head understanding. | Each of the 5 scores (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`) has different semantics and gradients. | One explanation covers all. |
| How handle disagreement between image and text? | Tests fusion reasoning. | Analyze fusion contribution and target-specific effects. | The model averages them. |
| Are explanations reproducible? | Tests rigor. | Fixed checkpoint, eval mode, fixed background, saved raw values. | I rerun plots. |
| What are XAI limitations? | Tests honesty. | Post hoc, approximate, sensitive to layers/background/perturbations. | There are none. |
| How validate explanation faithfulness? | Tests publication readiness. | Deletion/insertion, randomization, ablation, counterfactual tests. | Human inspection only. |
| What makes this thesis scientific? | Tests contribution. | Architecture-aligned XAI, per-head analysis, modality contribution, failure analysis, systematic comparison of backbones and fusion methods. | It uses deep learning and XAI. |
| Why 5 fusion methods? | Tests experimental design. | To systematically compare how fusion strategy affects both prediction and explainability: Concat, GMU, Gated Cross-Modal, FiLM, and Cross-Attention each fuse differently. | To try everything. |
| How do you handle multiple images per review? | Tests data awareness. | The system supports up to 4 images per review with masked average pooling (`Models/ImageModel.py`). For XAI, I generate Grad-CAM per image and analyze which image contributes most. | I use one image. |

# 6. Common Misconceptions and How To Destroy Them

| Misconception | Why It Is Wrong | Correct Explanation |
|---|---|---|
| Attention = explanation | Attention is not necessarily causal. | Treat it as token interaction evidence. |
| Grad-CAM proves causality | It is gradient-based post hoc evidence. | It shows supportive regions for a target. |
| SHAP explains pixels | Fusion SHAP explains embedding dimensions. | Use Grad-CAM/LIME for image localization. |
| LIME is always trustworthy | It is sampling- and segmentation-sensitive. | Test stability across seeds. |
| More XAI methods = more truth | Bad methods can multiply confusion. | Align each method to a question. |
| One heatmap explains all outputs | Each of the 5 outputs has separate gradients. | Explain each score head separately. |
| Pooled embeddings can produce Grad-CAM | They lack spatial dimensions. | Use `[B,C,H,W]` feature maps. |
| High SHAP means positive effect | High absolute SHAP can be negative or positive. | Separate magnitude and sign. |
| Background set does not matter | SHAP is baseline-relative. | Use representative training embeddings. |
| LIME explains the model globally | LIME is local. | Use many samples for broader claims. |
| Superpixels equal objects | Segmentation may not match semantics. | Interpret boundaries cautiously. |
| Token ids are meaningful numeric features | IDs are arbitrary indices. | Explain text through words/tokens or embeddings carefully. |
| XAI fixes model bias | It can reveal bias but not remove it. | Pair XAI with data auditing. |
| Pretty visualizations prove validity | Visual plausibility can mislead. | Add sanity checks and quantitative tests. |
| Identical explanations across heads are fine | Heads should differ when semantics differ (`food_score` vs `atmosphere_score`). | Check head collapse or loss design. |
| Image dominance always means better model | It may mean text branch is ignored. | Validate with ablation and SHAP. |
| Text dominance always means better model | It may exploit review bias. | Check image contribution and shortcuts. |
| Attention to special tokens is always meaningful | Special tokens often aggregate information. | Inspect before interpreting. |
| SHAP dimension names are semantic concepts | Embedding dimensions are distributed. | Prefer grouped modality-level analysis. |
| XAI makes black boxes fully transparent | XAI is partial and approximate. | Claim increased interpretability, not complete transparency. |
| All fusion methods produce the same SHAP layout | Different fusion methods (Concat, GMU, FiLM, Cross-Attention, Gated Cross-Modal) produce different internal representations. | Adapt SHAP attachment and grouping to the specific fusion architecture used. |

# 7. Red Flags During Implementation

| Area | Symptom | Possible cause | Fix |
|---|---|---|---|
| Grad-CAM | Blank heatmap | No gradient to selected layer | Check target layer, scalar output, `requires_grad`. |
| Grad-CAM | Uniform heatmap | Wrong layer or gradient saturation | Try later spatial layer; inspect gradients. |
| Grad-CAM | Highlights background/watermark instead of food | Model learned shortcut | Improve preprocessing, crop, augment, audit data. |
| Grad-CAM | Same map for all 5 heads | Wrong target index or head collapse | Generate per-head gradients; inspect prediction head (index 0-4 for `food_score` through `overall_satisfaction`). |
| Grad-CAM | Heatmap too precise in claims | Grad-CAM is low-resolution | Describe regions, not exact pixels. |
| Grad-CAM | Different results per review image | Multi-image system (up to 4) | This is expected; analyze which image carries the evidence. |
| Attention | All focus on special tokens | Special-token aggregation or artifact | Mask/inspect special tokens; compare layers. |
| Attention | Word labels unreadable | Subword tokenization (differs by encoder: XLM-R, PhoBERT, ViSoBERT) | Aggregate subwords for reporting. |
| Attention | Same attention for all conclusions | Attention not target-specific | Use LIME/SHAP for target-specific support. |
| Attention | Noisy map | Bad aggregation | Compare heads, last layer, last four layers, rollout. |
| SHAP | Values change wildly | Bad background or too few samples | Use representative `50-200` fused embeddings. |
| SHAP | Image always 99% | Text branch ignored or scale imbalance | Check text pooling, loss, normalization, ablation. |
| SHAP | Text always 99% | Image branch ignored or dataset text bias | Check image encoder, visual labels, image ablation. |
| SHAP | Additivity fails badly | Wrong wrapper/output shape | Return one scalar target; verify model eval mode. |
| SHAP | Semantic claims about dimension IDs | Over-interpretation | Report grouped or cross-sample patterns. |
| SHAP | Dimension grouping wrong | Used hardcoded split instead of actual backbone dimensions | Use `0:image_dim` and `image_dim:image_dim+text_dim` based on the actual encoders. |
| LIME | Different result each run | Too few perturbations | Increase `num_samples`, fix seeds, report stability. |
| LIME | Weird image regions | Bad superpixel segmentation | Tune segmentation or compare methods. |
| LIME | Text result ignores obvious word like "ngon" | Tokenization/preprocessing mismatch | Match training preprocessing and tokenizer. |
| LIME | Runtime too slow | Too many perturbations on full model | Use representative case studies; batch inference. |
| Fusion | One modality contributes near zero | Branch collapse | Ablation, loss balancing, branch-specific training checks. |
| Fusion | Good accuracy, bad explanations | Shortcut learning | Audit data, background, watermark, leakage, label bias. |

# 8. What Makes An XAI Thesis Strong

| Level | Characteristics |
|---|---|
| Weak thesis | Shows Grad-CAM and attention plots without proving they target the correct output. Claims attention explains decisions. No per-head analysis. No modality isolation. No limitations. |
| Good thesis | Correctly attaches Grad-CAM to the image encoder's spatial feature maps, extracts text encoder attention, explains each of the 5 target scores separately, and discusses limitations honestly. |
| Excellent thesis | Adds SHAP at the fusion embedding, quantifies image-vs-text contribution, uses LIME as local perturbation validation, compares correct and incorrect predictions, and analyzes method disagreement. |
| Publication-quality thesis | Adds faithfulness tests, deletion/insertion or ablation experiments, branch-collapse diagnostics, explanation stability analysis, failure taxonomy, counterfactual examples, and reproducible raw explanation artifacts. |

For your architecture, the thesis becomes strong when the explanation strategy is not decorative. It must answer research questions:

- Does the image encoder focus on real restaurant review evidence (food presentation, interior, plating)?
- Does the text encoder capture aspect-specific text cues (food quality, price, atmosphere, service)?
- Does fusion combine modalities or collapse into one?
- Do explanations differ across `food_score`, `price_score`, `atmosphere_score`, `service_score`, and `overall_satisfaction`?
- How does the choice of fusion method (Concat, GMU, Gated Cross-Modal, FiLM, Cross-Attention) affect explainability?
- Do perturbation tests support the visual explanations?
- Are limitations stated before examiners force them out?

> Defense posture: "I do not claim perfect interpretability. I claim architecture-aligned, target-specific, multi-method evidence that helps inspect and debug multimodal reasoning."

# 9. Final 5-Minute Revision Sheet

## 1-page ultra-condensed notes

- `Grad-CAM`: image localization. Attach to image encoder spatial map `[B,C,H,W]`. Never to pooled `[B, image_dim]`.
- `Attention`: token interaction. Useful, but not causal explanation.
- `SHAP`: fusion contribution. Use fused embedding `[B, text_dim + image_dim]`; group `0:image_dim` for image, `image_dim:image_dim+text_dim` for text. For best model (Swin-B + PhoBERT): image `0:1024`, text `1024:1792`.
- `LIME`: local perturbation. Image superpixels or text words; keep other modality fixed.
- Always explain one target score at a time (5 outputs: `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`).
- Always use `model.eval()`.
- Always save raw values plus figures.
- If methods disagree, investigate; do not hide it.
- XAI supports trust and debugging; it does not prove causality.
- Multi-image reviews: explain each image's Grad-CAM separately.

## formulas worth remembering

```text
Multimodal prediction:
y_hat = g([f_img(I); f_text(T)])

Fusion (concat):
e_fusion = [e_img ; e_text] in R^(image_dim + text_dim)
e.g. R^1792 for Swin-B (1024) + PhoBERT (768)
```

```text
Grad-CAM:
alpha_k^t = (1/Z) sum_i sum_j d y^t / d A_ij^k
L^t = ReLU(sum_k alpha_k^t A^k)
```

```text
Attention:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

```text
SHAP:
f(x) = phi0 + sum_i phi_i

Image contribution:
sum |phi_0:image_dim|

Text contribution:
sum |phi_image_dim:image_dim+text_dim|
```

```text
LIME:
xi(x) = argmin_g L(f,g,pi_x) + Omega(g)
```

## concepts worth remembering

- Grad-CAM = where.
- Attention = token interaction.
- SHAP = contribution.
- LIME = local removal test.
- Multimodal XAI = explain branches plus fusion.
- Regression XAI = target-specific scalar explanation.
- 5 targets: food, price, atmosphere, service, overall satisfaction.

## pitfalls worth remembering

- Attention is not explanation.
- SHAP does not localize pixels.
- LIME is not global.
- Grad-CAM is not causal.
- Wrong target head invalidates the explanation.
- Changing both modalities invalidates modality-specific claims.
- Hardcoding dimension splits (e.g., 768) without checking actual backbone dimensions.
- Ignoring multi-image handling when generating Grad-CAM.

## answers worth remembering

- "I use multiple methods because each explains a different level of the architecture."
- "I hold the other modality fixed to isolate the evidence source."
- "I apply SHAP at fusion because it directly measures image/text contribution."
- "I treat disagreement across methods as diagnostic, not as failure."
- "The system is not fully transparent; it is more inspectable through aligned XAI evidence."
- "The best model uses Swin-B + PhoBERT + Cross-Attention fusion + LogCosh loss, selected through systematic experimentation."

## key codebase files

- `Models/TextModel.py` - text encoder (XLM-R, PhoBERT, ViSoBERT via HuggingFace AutoModel)
- `Models/ImageModel.py` - image encoder (ConvNeXt, Swin-B, EfficientNet-B3, SigLIP via timm); handles up to 4 images with masked average pooling
- `Models/FusionModel.py` - concat fusion
- `Models/CrossAttentionFusion.py` - cross-attention fusion (best model)
- `Models/GMUFusion.py` - Gated Multimodal Unit fusion
- `Models/GatedCrossModalFusion.py` - Gated Cross-Modal fusion
- `Models/FiLMFusion.py` - Feature-wise Linear Modulation fusion
- `Trainer.py` - training loop with MSE, Huber, LogCosh, and HomoscedasticUncertaintyLoss
- `Config.py` - argument parsing and experiment configuration
- `src/dataset.py` - dataset with 5 targets: `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`
- `main.py` - entry point
- `test.py` - evaluation
