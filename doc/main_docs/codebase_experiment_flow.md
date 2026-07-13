# Codebase Experiment Flow

> How every experiment in this project is organized, executed, and produces results.

---

## 1. Overall Experiment Flow

Every experiment follows this execution flow:

```
Notebook (.ipynb on Google Colab)
│
├── STEP 1: Mount Google Drive
├── STEP 2: git clone repo + pip install requirements
├── STEP 3: Download data.zip from Google Drive → extract to ./data/
├── STEP 4: Set EXP_ID, DRIVE_ROOT, DRIVE_EXP_PATH
│
├── STEP 5: Load pretrained weights (if needed)
│   └── Copy .pth files from Google Drive → ./checkpoints/
│
├── STEP 6: !python main.py --mode <mode> --<flags>
│   │
│   ├── Config.py: get_args() → parse CLI arguments
│   ├── main.py: set_seed() → reproducibility
│   │
│   ├── AutoTokenizer.from_pretrained(text_model_name)
│   ├── AutoImageProcessor / TimmProcessor(image_model_name)
│   │
│   ├── MultimodalDataset(train.csv) → train_loader
│   ├── MultimodalDataset(val.csv)   → val_loader
│   │
│   ├── Build model (based on --mode):
│   │   ├── train_text  → TextModel
│   │   ├── train_image → ImageModel
│   │   └── train_fusion → TextModel + ImageModel + FusionModel
│   │       └── Load pretrained unimodal weights from ./checkpoints/
│   │
│   ├── Trainer(model, train_loader, val_loader, device, args)
│   │   ├── Select loss function (MSE / Huber / LogCosh / AutoWeight)
│   │   ├── AdamW optimizer + cosine warmup scheduler
│   │   └── Optional AMP (mixed precision)
│   │
│   └── Trainer.run()
│       ├── for epoch in range(epochs):
│       │   ├── train_epoch() → forward, loss, backward, grad clip, step
│       │   ├── validate()    → compute MAE, RMSE, R² per factor
│       │   ├── Early stopping (patience-based on mean_mae)
│       │   └── Save best checkpoint → experiments/<EXP_ID>/best_model_<mode>.pth
│       │
│       └── Final evaluation with best checkpoint:
│           ├── Save metrics.json    (MAE, RMSE, R² for 5 factors)
│           ├── Save predictions.csv (y_true, y_pred, absolute_error per sample)
│           ├── Save config.yaml     (all hyperparameters)
│           └── Save train.log       (epoch-by-epoch log)
│
├── STEP 7 (optional): !python test.py → evaluate on test set
│   └── Produces: test_metrics.json, test_predictions.csv, PNG plots
│
└── STEP 8: Copy artifacts from ./experiments/ to Google Drive
```

### Output Artifacts per Experiment

| Artifact | Content |
|---|---|
| `best_model_<mode>.pth` | Model weights + optimizer + scheduler + best metrics |
| `metrics.json` | Validation MAE/RMSE/R² for food, price, atmos, service, overall |
| `predictions.csv` | Per-sample y_true, y_pred, absolute_error for all 5 factors |
| `config.yaml` / `config.json` | All CLI arguments used |
| `train.log` | Timestamped epoch-by-epoch training log |
| `test_metrics.json` | Test set metrics (Phase 6+ only) |
| `test_predictions.csv` | Test set per-sample predictions (Phase 6+ only) |
| `test_*.png` | Visualization plots (Phase 6+ only) |

---

## 2. Why Use Both .py and .ipynb?

### Pattern: Notebooks as Orchestrators, Python Files as Reusable Logic

**Why logic lives in `.py` files:**
- `main.py`, `Trainer.py`, `Models/*.py` are **reusable across all 21 experiments**
- A single `Trainer.py` handles every training loop — no copy-paste across notebooks
- Models (`TextModel.py`, `ImageModel.py`, fusion variants) are imported and configured via CLI flags
- Changing a bug in `Trainer.py` fixes it for all experiments simultaneously

**Why notebooks only orchestrate:**
- Each notebook is a **self-contained recipe** that runs one experiment on Colab
- Notebooks handle Colab-specific setup: Drive mounting, repo cloning, data download
- Notebooks set experiment-specific flags (`--image_model_name`, `--fusion_type`, `--loss_fn`)
- Notebooks copy pretrained weights and save results to persistent storage (Google Drive)

**What belongs where:**

| In `.py` modules | In `.ipynb` notebooks |
|---|---|
| Model architectures | Drive mount + clone + data setup |
| Training loop | CLI flag configuration per experiment |
| Loss functions | Pretrained weight loading from prior experiments |
| Dataset loading | Artifact copy to Google Drive |
| Evaluation metrics | Results printing/display |

**Advantages:**
- Zero code duplication across experiments
- Easy to add a new experiment: just create a new notebook with different flags
- All experiments share the exact same training infrastructure
- Can run via CLI (`python main.py`) without notebooks for debugging

---

## 3. Runtime Dependency Map

```
Notebook (.ipynb)
│
└── !python main.py
    │
    ├── Config.py ─────────── get_args() → argparse
    │
    ├── src/dataset.py ────── MultimodalDataset
    │                         AdvancedMultimodalDataset (unused in current experiments)
    │
    ├── Models/
    │   ├── TextModel.py ──── TextModel (HuggingFace AutoModel wrapper)
    │   ├── ImageModel.py ─── ImageModel (timm model wrapper)
    │   │
    │   └── Fusion variants (selected by --fusion_type):
    │       ├── FusionModel.py ────────── FusionModel          (concat, default)
    │       ├── GMUFusion.py ──────────── GMUFusion             (gmu)
    │       ├── GatedCrossModalFusion.py  GatedCrossModalFusion (gated_cross)
    │       ├── FiLMFusion.py ─────────── FiLMFusion            (film)
    │       └── CrossAttentionFusion.py ── CrossAttentionFusion  (cross_attention)
    │
    └── Trainer.py ────────── Trainer
        │                     LogCoshLoss
        │                     HomoscedasticUncertaintyLoss
        │
        ├── train_epoch() ─── forward → loss → backward → optimize
        ├── validate() ────── MAE, RMSE, R² per factor
        └── run() ─────────── training loop + save checkpoint/metrics/predictions

!python test.py ───────── test() → load checkpoint → inference on test set
                          → test_metrics.json, test_predictions.csv, PNG plots
```

---

## 4. General Experiment Template

The common execution flow shared by almost every experiment:

```
Notebook
│
├── Mount Drive, clone repo, install deps, download data
│
├── Set EXP_ID and paths
│
├── Load pretrained weights (copy .pth from Drive → ./checkpoints/)
│   ├── best_model_train_text.pth   (from a prior text experiment)
│   └── best_model_train_image.pth  (from a prior image experiment)
│
├── [Optional] Pre-train unimodal branch (only Phase 2-3):
│   └── !python main.py --mode train_image/train_text ...
│
├── Train fusion:
│   └── !python main.py \
│       --mode train_fusion \
│       --fusion_type <concat|gmu|gated_cross|film|cross_attention> \
│       --text_model_name <model> \
│       --image_model_name <model> \
│       --loss_fn <mse|huber|logcosh|auto_weight> \
│       --epochs 15 --batch_size 16 --lr 1e-5 \
│       --grad_accum_steps 2 --patience 5 \
│       --unfreeze_text_layers 1 --unfreeze_image_layers 1 \
│       --seed 42 --use_amp \
│       --exp_id <EXP_ID> --exp_dir ./experiments
│
├── [Optional] Evaluate on test set (Phase 6+):
│   └── !python test.py --mode train_fusion --fusion_type <type> ...
│
└── Copy results to Google Drive + print metrics
```

### Shared Hyperparameters (Fusion Training)

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Gradient accumulation | 2 |
| Early stopping patience | 5 |
| Unfrozen text layers | 1 |
| Unfrozen image layers | 1 |
| Seed | 42 (except EXP_070: 123) |
| AMP | Enabled |
| Optimizer | AdamW (weight_decay=1e-2) |
| Scheduler | Cosine with warmup (ratio=0.1) |

---

## 5. Detailed Flow of Every Experiment

### Experiment Dependency Chain

```
Phase 1: Baselines
  EXP_010 (Text-Only)  ──┐
  EXP_011 (Image-Only) ──┤
  EXP_012 (Multimodal)   │ (uses weights from EXP_010 + EXP_011)
                          │
Phase 2: Image Backbone Ablation (all reuse EXP_010 text weights)
  EXP_020B (Swin-B)    ──┤
  EXP_020D (EffNet-B3) ──┤
  EXP_020E (SigLIP)    ──┤
                          │
Phase 3: Text Backbone Ablation (all reuse EXP_020B image weights)
  EXP_030B (PhoBERT)   ──┤
  EXP_030D (ViSoBERT)  ──┤
                          │
Phase 4: Fusion Ablation (all reuse EXP_020B image + EXP_030B text)
  EXP_040B (GMU)       ──┤
  EXP_040C (GatedCross)──┤
  EXP_041A (FiLM)      ──┤
  EXP_041B (CrossAttn)───┤
                          │
Phase 5: Loss Ablation (all reuse EXP_020B image + EXP_030B text, CrossAttn fusion)
  EXP_050B (Huber)     ──┤
  EXP_050C (LogCosh)   ──┤
  EXP_051D (AutoWeight)──┤
                          │
Phase 6: Promising Combinations
  EXP_060A (Eval-only: best sequential) ──┤
  EXP_060B (Swin-B + ViSoBERT + GMU + AutoWeight) ──┤
  EXP_060C (EffNet-B3 + PhoBERT + FiLM + Huber) ──┤
  EXP_060D (EffNet-B3 + ViSoBERT + CrossAttn + LogCosh) ──┤
  EXP_060E (ConvNeXt + PhoBERT + GatedCross + AutoWeight) ──┤
                          │
Phase 7: Stability
  EXP_070 (Seed 123)
```

---

### EXP_010_text_only_xlmr_mse

**Purpose:** Text-only baseline — how strong is review text alone?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_text` |
| Text Backbone | `xlm-roberta-base` |
| Image Backbone | DISABLED |
| Fusion | None |
| Loss | MSE |
| Epochs | 20 |
| Batch Size | 32 |

**Runtime Flow:**
```
Notebook → main.py --mode train_text
  → TextModel(xlm-roberta-base)
  → Trainer.run()
  → metrics.json + predictions.csv + best_model_train_text.pth
```

**Files Used:**
```
EXP_010_text_only_xlmr_mse.ipynb → main.py → Config.py
  → src/dataset.py::MultimodalDataset
  → Models/TextModel.py::TextModel
  → Trainer.py::Trainer
```

**Important Functions:**
- `TextModel.__init__(model_name)` — loads HuggingFace AutoModel
- `TextModel.forward(input_ids, attention_mask)` — returns (predictions, features)
- `Trainer.run()` — full training loop with early stopping

**Notes:** This is the foundational text experiment. Its `best_model_train_text.pth` is reused by EXP_012, EXP_020B, EXP_020D, EXP_020E as the pretrained text encoder.

---

### EXP_011_image_only_convnext_meanpool_mse

**Purpose:** Image-only baseline — how much signal exists in review images alone?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_image` |
| Text Backbone | DISABLED |
| Image Backbone | `convnext_base_in22k` |
| Fusion | None |
| Loss | MSE |
| Epochs | 20 |
| Batch Size | 32 |

**Runtime Flow:**
```
Notebook → main.py --mode train_image
  → ImageModel(convnext_base_in22k)
  → Trainer.run()
  → metrics.json + predictions.csv + best_model_train_image.pth
```

**Files Used:**
```
EXP_011_image_only_convnext_mse.ipynb → main.py → Config.py
  → src/dataset.py::MultimodalDataset
  → Models/ImageModel.py::ImageModel
  → Trainer.py::Trainer
```

**Important Functions:**
- `ImageModel.__init__(model_name)` — loads timm model with `num_classes=0`
- `ImageModel.forward(pixel_values, num_images)` — handles multi-image average pooling
- `TimmProcessor.__call__()` — wraps timm transforms as HuggingFace-style processor

**Notes:** Uses `TimmProcessor` (not `AutoImageProcessor`) since `convnext_base_in22k` is a timm model. Its checkpoint is used by EXP_012 and EXP_060E.

---

### EXP_012_multimodal_convnext_xlmr_concat_mse

**Purpose:** Multimodal baseline — does simple fusion improve over unimodal?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `xlm-roberta-base` (weights from EXP_010) |
| Image Backbone | `convnext_base_in22k` (weights from EXP_011) |
| Fusion | Concat (default `FusionModel`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_text.pth from EXP_010
  → Copy best_model_train_image.pth from EXP_011
  → main.py --mode train_fusion
    → TextModel + ImageModel (load pretrained weights)
    → FusionModel(text_model, image_model)
    → Trainer.run()
    → metrics.json + best_model_train_fusion.pth
```

**Files Used:**
```
EXP_012_multimodal_convnext_xlmr_concat_mse.ipynb → main.py → Config.py
  → src/dataset.py::MultimodalDataset
  → Models/TextModel.py::TextModel
  → Models/ImageModel.py::ImageModel
  → Models/FusionModel.py::FusionModel
  → Trainer.py::Trainer
```

**Important Functions:**
- `FusionModel.__init__()` — freezes backbone weights, unfreezes last N layers
- `FusionModel.forward()` — extracts text+image features, concatenates, passes through MLP

**Notes:** First multimodal experiment. Loads pretrained unimodal weights from EXP_010 and EXP_011.

---

### EXP_020B_swinb_xlmr_concat_mse

**Purpose:** Image ablation — does Swin-B provide better visual features than ConvNeXt?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `xlm-roberta-base` (weights from EXP_010) |
| Image Backbone | `swin_base_patch4_window7_224` (pre-trained in same notebook) |
| Fusion | Concat (`FusionModel`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_text.pth from EXP_010
  → Pre-train image branch: main.py --mode train_image --image_model_name swin_base_patch4_window7_224
  → Train fusion: main.py --mode train_fusion
    → FusionModel(TextModel, ImageModel)
    → Trainer.run()
    → metrics.json + best_model_train_fusion.pth
```

**Files Used:**
```
EXP_020B_swinb_xlmr_concat_mse.ipynb → main.py → Config.py
  → Models/TextModel.py, Models/ImageModel.py, Models/FusionModel.py
  → Trainer.py::Trainer
```

**Notes:** Has two training steps: first pre-trains Swin-B image branch (20 epochs), then trains fusion (15 epochs). This pattern is used by all Phase 2 and 3 experiments. The image weights from this experiment become the "best image" used by Phase 3, 4, and 5.

---

### EXP_020D_efficientnetb3_xlmr_concat_mse

**Purpose:** Image ablation — does EfficientNet-B3 offer better compute efficiency?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `xlm-roberta-base` (weights from EXP_010) |
| Image Backbone | `efficientnet_b3` (pre-trained in same notebook) |
| Fusion | Concat (`FusionModel`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_text.pth from EXP_010
  → Pre-train image: main.py --mode train_image --image_model_name efficientnet_b3
  → Train fusion: main.py --mode train_fusion
    → FusionModel(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_020B, with `efficientnet_b3` instead of Swin-B.

**Notes:** Uses `TimmProcessor` for image preprocessing. Its image weights are used by EXP_060C and EXP_060D.

---

### EXP_020E_siglip_xlmr_concat_mse

**Purpose:** Image ablation — does SigLIP provide better visual features?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `xlm-roberta-base` (weights from EXP_010) |
| Image Backbone | `vit_base_patch16_siglip_256` (pre-trained in same notebook) |
| Fusion | Concat (`FusionModel`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_text.pth from EXP_010
  → Pre-train image: main.py --mode train_image --image_model_name vit_base_patch16_siglip_256
  → Train fusion: main.py --mode train_fusion
    → FusionModel(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_020B, with SigLIP model.

**Notes:** Uses `AutoImageProcessor.from_pretrained('google/siglip-base-patch16-256')` as a special-case fallback in `main.py` for SigLIP models.

---

### EXP_030B_bestimage_phobert_concat_mse

**Purpose:** Text ablation — does PhoBERT (Vietnamese-specific) improve over XLM-R?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (pre-trained in same notebook) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Concat (`FusionModel`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_image.pth from EXP_020B
  → Pre-train text: main.py --mode train_text --text_model_name vinai/phobert-base-v2
  → Copy best_model_train_text.pth to checkpoints
  → Train fusion: main.py --mode train_fusion
    → FusionModel(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_020B with `vinai/phobert-base-v2`.

**Notes:** Has two training steps (text pre-train + fusion), like Phase 2 experiments. The text weights from this experiment become the "best text" used by Phase 4 and 5.

---

### EXP_030D_bestimage_visobert_concat_mse

**Purpose:** Text ablation — does ViSoBERT (Vietnamese social media) improve over XLM-R?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `uitnlp/visobert` (pre-trained in same notebook) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Concat (`FusionModel`) |
| Loss | MSE |

**Runtime Flow:** Same as EXP_030B with `uitnlp/visobert`.

**Files Used:** Same as EXP_030B.

**Notes:** ViSoBERT text weights are used by EXP_060B and EXP_060D.

---

### EXP_040B_bestimage_besttext_gmu_mse

**Purpose:** Fusion ablation — does Gated Multimodal Unit improve over Concat?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | GMU (`--fusion_type gmu`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Copy best_model_train_text.pth from EXP_030B
  → Copy best_model_train_image.pth from EXP_020B
  → main.py --mode train_fusion --fusion_type gmu
    → GMUFusion(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:**
```
EXP_040B_bestimage_besttext_gmu_mse.ipynb → main.py
  → Models/TextModel.py, Models/ImageModel.py
  → Models/GMUFusion.py::GMUFusion
  → Trainer.py::Trainer
```

**Important Functions:**
- `GMUFusion.__init__()` — creates text/image projections + sigmoid gate
- `GMUFusion.forward()` — `gate = sigmoid(W[text;image])`, `fused = gate*text_proj + (1-gate)*image_proj`

**Notes:** Only one training step (fusion only) — no unimodal pre-training needed since it reuses pretrained weights.

---

### EXP_040C_bestimage_besttext_gatedcrossmodal_mse

**Purpose:** Fusion ablation — does Gated Cross-Modal Fusion improve over GMU?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Gated Cross-Modal (`--fusion_type gated_cross`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type gated_cross
    → GatedCrossModalFusion(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_040B but uses `Models/GatedCrossModalFusion.py`.

**Important Functions:**
- `GatedCrossModalFusion.forward()` — each modality is conditioned on the other via tanh cross-projections, then gated

**Notes:** Cross-modal conditioning: `text_enh = text + tanh(W_t2i * image)`, `image_enh = image + tanh(W_i2t * text)`.

---

### EXP_041A_bestimage_besttext_film_mse

**Purpose:** Fusion ablation — does FiLM (Feature-wise Linear Modulation) improve fusion?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | FiLM (`--fusion_type film`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type film
    → FiLMFusion(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_040B but uses `Models/FiLMFusion.py`.

**Important Functions:**
- `FiLMFusion.forward()` — text generates gamma/beta to modulate image: `modulated = gamma * image + beta`

**Notes:** Text modulates image features asymmetrically (text → image direction only).

---

### EXP_041B_bestimage_besttext_crossattention_mse

**Purpose:** Fusion ablation — does Cross-Attention improve fine-grained interaction?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Cross-Attention (`--fusion_type cross_attention`) |
| Loss | MSE |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type cross_attention
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer.run()
```

**Files Used:** Same as EXP_040B but uses `Models/CrossAttentionFusion.py`.

**Important Functions:**
- `CrossAttentionFusion.forward()` — bidirectional cross-attention: text→image and image→text via `nn.MultiheadAttention`

**Notes:** Uses 8-head attention. Produces the best Phase 4 result, becoming the "best fusion" for Phase 5.

---

### EXP_050B_bestfusion_huber

**Purpose:** Loss ablation — does Huber loss handle outliers better than MSE?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Cross-Attention |
| Loss | **Huber** (`--loss_fn huber`) |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type cross_attention --loss_fn huber
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer(criterion=nn.HuberLoss())
    → Trainer.run()
```

**Files Used:** Same as EXP_041B but with `--loss_fn huber`.

**Notes:** Architecture identical to EXP_041B; only the loss function changes.

---

### EXP_050C_bestfusion_logcosh

**Purpose:** Loss ablation — does Log-Cosh loss provide smoother gradients?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Cross-Attention |
| Loss | **LogCosh** (`--loss_fn logcosh`) |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type cross_attention --loss_fn logcosh
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer(criterion=LogCoshLoss())
    → Trainer.run()
```

**Files Used:** Same as EXP_050B but with `LogCoshLoss` (defined in `Trainer.py`).

**Notes:** LogCosh is numerically stabilized: `log(cosh(x)) = |x| - log(2) + softplus(-2|x|)`.

---

### EXP_051D_bestfusion_uncertaintyweighted

**Purpose:** Loss ablation — does learnable task uncertainty weighting improve 5-target balance?

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | Cross-Attention |
| Loss | **HomoscedasticUncertaintyLoss** (`--loss_fn auto_weight`) |

**Runtime Flow:**
```
Notebook
  → Load pretrained text + image weights
  → main.py --mode train_fusion --fusion_type cross_attention --loss_fn auto_weight
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer(criterion=HomoscedasticUncertaintyLoss(num_tasks=5))
    → Trainer.run()
```

**Files Used:** Same as EXP_050B but with `HomoscedasticUncertaintyLoss`.

**Important Functions:**
- `HomoscedasticUncertaintyLoss.__init__()` — learnable `log_vars` parameter per task
- `HomoscedasticUncertaintyLoss.forward()` — `loss = exp(-log_var) * mse + log_var`

**Notes:** The loss parameters are added to the optimizer alongside model parameters.

---

### EXP_060A_bestsequential_full_configuration

**Purpose:** Evaluate the Phase 5 winner (best sequential pipeline) on the test set. No additional training.

**Configuration:**

| Component | Value |
|---|---|
| Mode | Evaluation only (`test.py`) |
| Text Backbone | `vinai/phobert-base-v2` |
| Image Backbone | `swin_base_patch4_window7_224` |
| Fusion | Cross-Attention |
| Loss | LogCosh |

**Runtime Flow:**
```
Notebook
  → Copy entire EXP_050C experiment directory to ./checkpoints/
  → test.py --mode train_fusion --fusion_type cross_attention --loss_fn logcosh
    → Load best_model_train_fusion.pth
    → Inference on test set
    → test_metrics.json + test_predictions.csv + PNG plots
```

**Files Used:**
```
EXP_060A_bestsequential_full_configuration.ipynb → test.py → Config.py
  → Models/TextModel.py, Models/ImageModel.py
  → Models/CrossAttentionFusion.py::CrossAttentionFusion
  → src/dataset.py::MultimodalDataset
```

**Notes:** This is the only experiment that does NO training — it only evaluates a previously trained checkpoint from EXP_050C.

---

### EXP_060B_swinb_visobert_gmu_uncertainty

**Purpose:** Alternative combination — Swin-B + ViSoBERT + GMU + Uncertainty-Weighted loss.

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `uitnlp/visobert` (weights from EXP_030D) |
| Image Backbone | `swin_base_patch4_window7_224` (weights from EXP_020B) |
| Fusion | GMU (`--fusion_type gmu`) |
| Loss | AutoWeight (`--loss_fn auto_weight`) |

**Runtime Flow:**
```
Notebook
  → Copy text weights from EXP_030D, image weights from EXP_020B
  → main.py --mode train_fusion --fusion_type gmu --loss_fn auto_weight
    → GMUFusion(TextModel, ImageModel)
    → Trainer.run()
  → test.py → test_metrics.json
```

**Files Used:** `Models/GMUFusion.py`, `Trainer.py::HomoscedasticUncertaintyLoss`

**Notes:** Combines ViSoBERT (social media text) with GMU fusion and uncertainty loss. Includes test set evaluation via `test.py`.

---

### EXP_060C_efficientnetb3_phobert_film_huber

**Purpose:** Alternative combination — EfficientNet-B3 + PhoBERT + FiLM + Huber.

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `efficientnet_b3` (weights from EXP_020D) |
| Fusion | FiLM (`--fusion_type film`) |
| Loss | Huber (`--loss_fn huber`) |

**Runtime Flow:**
```
Notebook
  → Copy text weights from EXP_030B, image weights from EXP_020D
  → main.py --mode train_fusion --fusion_type film --loss_fn huber
    → FiLMFusion(TextModel, ImageModel)
    → Trainer.run()
  → test.py → test_metrics.json
```

**Files Used:** `Models/FiLMFusion.py`, `Trainer.py`

**Notes:** Uses EfficientNet-B3 image backbone (from Phase 2) instead of Swin-B.

---

### EXP_060D_efficientnetb3_visobert_crossattention_logcosh

**Purpose:** Alternative combination — EfficientNet-B3 + ViSoBERT + Cross-Attention + LogCosh.

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `uitnlp/visobert` (weights from EXP_030D) |
| Image Backbone | `efficientnet_b3` (weights from EXP_020D) |
| Fusion | Cross-Attention (`--fusion_type cross_attention`) |
| Loss | LogCosh (`--loss_fn logcosh`) |

**Runtime Flow:**
```
Notebook
  → Copy text weights from EXP_030D, image weights from EXP_020D
  → main.py --mode train_fusion --fusion_type cross_attention --loss_fn logcosh
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer.run()
  → test.py → test_metrics.json
```

**Files Used:** `Models/CrossAttentionFusion.py`, `Trainer.py::LogCoshLoss`

**Notes:** Combines the second-best image (EfficientNet-B3) with ViSoBERT and the best-performing loss (LogCosh).

---

### EXP_060E_convnext_phobert_gatedcrossmodal_autoweight

**Purpose:** Alternative combination — ConvNeXt + PhoBERT + Gated Cross-Modal + Auto-Weight.

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (weights from EXP_030B) |
| Image Backbone | `convnext_base_in22k` (weights from **EXP_011**, Phase 1) |
| Fusion | Gated Cross-Modal (`--fusion_type gated_cross`) |
| Loss | AutoWeight (`--loss_fn auto_weight`) |

**Runtime Flow:**
```
Notebook
  → Copy text weights from EXP_030B, image weights from EXP_011
  → main.py --mode train_fusion --fusion_type gated_cross --loss_fn auto_weight
    → GatedCrossModalFusion(TextModel, ImageModel)
    → Trainer.run()
  → test.py → test_metrics.json
```

**Files Used:** `Models/GatedCrossModalFusion.py`, `Trainer.py::HomoscedasticUncertaintyLoss`

**Notes:** Unlike other Phase 6 experiments, this one loads image weights from Phase 1 (EXP_011) instead of Phase 2, using the original ConvNeXt baseline.

---

### EXP_070_bestmodel_seed123

**Purpose:** Multi-seed stability validation — re-train the best architecture with seed 123.

**Configuration:**

| Component | Value |
|---|---|
| Mode | `train_fusion` |
| Text Backbone | `vinai/phobert-base-v2` (placeholder — to be set by user) |
| Image Backbone | `convnext_base_in22k` (placeholder — to be set by user) |
| Fusion | Cross-Attention |
| Loss | LogCosh |
| **Seed** | **123** (not 42) |

**Runtime Flow:**
```
Notebook
  → Copy text + image weights from best experiments (user must configure)
  → main.py --mode train_fusion --fusion_type cross_attention --loss_fn logcosh --seed 123
    → CrossAttentionFusion(TextModel, ImageModel)
    → Trainer.run()
  → test.py → test_metrics.json
```

**Files Used:** `Models/CrossAttentionFusion.py`, `Trainer.py::LogCoshLoss`

**Notes:** The notebook has TODO placeholders for `BEST_TEXT_EXP_ID` and `BEST_IMAGE_EXP_ID` — the user must fill these based on Phase 6 results. Default values in the notebook use `convnext_base_in22k` + `vinai/phobert-base-v2`.

---

## 6. Code Reuse Summary

| File | Used By | Responsibility |
|---|---|---|
| `main.py` | All training experiments | Entry point: parse args, build model, create Trainer, call run() |
| `test.py` | EXP_060A, 060B, 060C, 060D, 060E, 070 | Test set evaluation: load checkpoint, inference, save metrics/plots |
| `Config.py` | All experiments (via main.py/test.py) | CLI argument parsing (argparse) |
| `Trainer.py` | All training experiments | Training loop, validation, checkpointing, metrics, early stopping |
| `Trainer.py::LogCoshLoss` | EXP_050C, 060A, 060D, 070 | Log-Cosh loss function |
| `Trainer.py::HomoscedasticUncertaintyLoss` | EXP_051D, 060B, 060E | Learnable per-task uncertainty weighting |
| `src/dataset.py::MultimodalDataset` | All experiments | Dataset: loads CSV, tokenizes text, processes images, returns tensors |
| `Models/TextModel.py` | All experiments | Text encoder wrapper (HuggingFace AutoModel → features → predictions) |
| `Models/ImageModel.py` | All experiments except EXP_010 | Image encoder wrapper (timm model → multi-image avg pool → predictions) |
| `Models/FusionModel.py` | EXP_012, 020B, 020D, 020E, 030B, 030D | Concat fusion (default): `cat(text, image) → MLP → predictions` |
| `Models/GMUFusion.py` | EXP_040B, 060B | Gated Multimodal Unit: sigmoid gate selects text vs image |
| `Models/GatedCrossModalFusion.py` | EXP_040C, 060E | Cross-modal conditioning + gating |
| `Models/FiLMFusion.py` | EXP_041A, 060C | Text generates gamma/beta to modulate image features |
| `Models/CrossAttentionFusion.py` | EXP_041B, 050B, 050C, 051D, 060A, 060D, 070 | Bidirectional multi-head cross-attention |
| `preprocess_data.py` | Data preparation (run once) | Merge raw CSVs, group by review_id, split train/val/test |
| `download_images.py` | Data preparation (run once) | Download images from URLs, hash-based dedup, parallel download |
| `verify_dataset_script.py` | Data verification | Check image count and CSV row counts |
| `test_fusion_shapes.py` | Development/testing | Unit test all 5 fusion architectures with mock models |

---

## 7. Folder Responsibility

```
SE365/
│
├── Models/                    # All model architectures
│   ├── TextModel.py           # HuggingFace text encoder wrapper
│   ├── ImageModel.py          # timm image encoder wrapper  
│   ├── FusionModel.py         # Concat fusion (baseline)
│   ├── GMUFusion.py           # Gated Multimodal Unit fusion
│   ├── GatedCrossModalFusion.py  # Gated Cross-Modal fusion
│   ├── FiLMFusion.py          # Feature-wise Linear Modulation fusion
│   └── CrossAttentionFusion.py   # Multi-head Cross-Attention fusion
│
├── src/                       # Data loading
│   └── dataset.py             # MultimodalDataset, AdvancedMultimodalDataset
│
├── notebook/                  # All experiment notebooks (run on Colab)
│   ├── EXP_010_*.ipynb        # Phase 1: Text-only baseline
│   ├── EXP_011_*.ipynb        # Phase 1: Image-only baseline
│   ├── EXP_012_*.ipynb        # Phase 1: Multimodal baseline
│   ├── EXP_020*.ipynb         # Phase 2: Image backbone ablation
│   ├── EXP_030*.ipynb         # Phase 3: Text backbone ablation
│   ├── EXP_04*.ipynb          # Phase 4: Fusion architecture ablation
│   ├── EXP_05*.ipynb          # Phase 5: Loss function ablation
│   ├── EXP_06*.ipynb          # Phase 6: Promising combinations + test eval
│   ├── EXP_070_*.ipynb        # Phase 7: Multi-seed stability
│   ├── clean_foody_dataset.ipynb      # Data cleaning notebook
│   ├── crawl_data_from_foody.ipynb    # Data crawling notebook
│   ├── demo_single_sample_exp060A.ipynb  # Single-sample demo
│   ├── generate_experiment_leaderboard.ipynb  # Results comparison
│   └── kaggle_*.ipynb, *.ipynb        # Other experimental notebooks
│
├── data_raw/                  # Raw crawled data (CSVs, JSONs)
├── data_processed/            # Cleaned/enhanced data
├── doc/                       # Documentation and guides
├── draft/                     # Draft documents
│
├── main.py                    # Training entry point
├── test.py                    # Test set evaluation entry point
├── Config.py                  # CLI argument definitions
├── Trainer.py                 # Training loop + loss functions
├── preprocess_data.py         # Data preprocessing pipeline
├── download_images.py         # Image download utility
├── verify_dataset_script.py   # Dataset verification
├── test_fusion_shapes.py      # Unit test for fusion architectures
│
├── requirements.txt           # Python dependencies
├── experiment_plan.md         # Experimental roadmap
└── metrics_EXP_*.json         # Cached validation metrics per experiment
```

---

## 8. End-to-End Example: EXP_050C_bestfusion_logcosh

This experiment trains the best fusion architecture (Swin-B + PhoBERT + Cross-Attention) with LogCosh loss.

### Step-by-step execution:

**1. Notebook Setup (Colab)**
```
Mount Google Drive → /content/drive
Clone repo → /content/SE365
Install: numpy, pandas, Pillow, torch, transformers, timm, ...
Download data.zip → extract to ./data/ (./data/text/*.csv + ./data/image/*.jpg)
```

**2. Set Experiment Config**
```python
EXP_ID = 'EXP_050C_bestfusion_logcosh'
DRIVE_EXP_PATH = '/content/drive/MyDrive/SE365/experiments/EXP_050C_bestfusion_logcosh'
```

**3. Load Pretrained Weights**
```python
shutil.copy('.../EXP_030B_.../best_model_train_text.pth', './checkpoints/best_model_train_text.pth')
shutil.copy('.../EXP_020B_.../best_model_train_image.pth', './checkpoints/best_model_train_image.pth')
```

**4. Execute Training** (`!python main.py ...`)

Inside `main.py`:
```
Config.get_args() → parse CLI flags
  --mode train_fusion
  --fusion_type cross_attention
  --text_model_name vinai/phobert-base-v2
  --image_model_name swin_base_patch4_window7_224
  --loss_fn logcosh
  --seed 42 --use_amp

set_seed(42) → reproducibility

AutoTokenizer.from_pretrained('vinai/phobert-base-v2') → tokenizer
TimmProcessor('swin_base_patch4_window7_224') → image_processor

MultimodalDataset('data/text/train.csv', tokenizer, image_processor) → train_dataset
MultimodalDataset('data/text/val.csv', tokenizer, image_processor)   → val_dataset

DataLoader(train_dataset, batch_size=16, shuffle=True) → train_loader
DataLoader(val_dataset, batch_size=16, shuffle=False)   → val_loader
```

Model construction:
```
TextModel('vinai/phobert-base-v2')
  → AutoModel.from_pretrained('vinai/phobert-base-v2')
  → Linear(hidden_size → 256) + ReLU + Dropout
  → Linear(256 → 5) factor_head

ImageModel('swin_base_patch4_window7_224')
  → timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
  → Linear(num_features → 256) + ReLU + Dropout
  → Linear(256 → 5) factor_head

Load pretrained weights:
  text_model.load_state_dict(checkpoints/best_model_train_text.pth)
  image_model.load_state_dict(checkpoints/best_model_train_image.pth)

CrossAttentionFusion(text_model, image_model)
  → Freeze all backbone params
  → Unfreeze last 1 text layer + last 1 image stage block
  → text_proj: Linear(text_dim → 512)
  → image_proj: Linear(image_dim → 512)
  → cross_attn_t2i: MultiheadAttention(512, 8 heads)
  → cross_attn_i2t: MultiheadAttention(512, 8 heads)
  → head: Linear(1024 → 512) → ReLU → Dropout → Linear(512 → 256) → ReLU → Linear(256 → 5)
```

Trainer initialization:
```
Trainer(model, train_loader, val_loader, device, args)
  → criterion = LogCoshLoss()
  → optimizer = AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
  → scheduler = cosine_warmup(warmup_ratio=0.1)
  → scaler = GradScaler(enabled=True)  # AMP
```

Training loop (`Trainer.run()`):
```
Create experiment directory: ./experiments/EXP_050C_bestfusion_logcosh/
Save config.yaml

for epoch in range(15):
  train_epoch():
    for batch in train_loader:
      inputs = {input_ids, attention_mask, pixel_values, num_images} → device
      with autocast:
        output = model(**inputs)                    # CrossAttentionFusion.forward()
        pred_factors = output                       # shape [B, 5]
        loss = LogCoshLoss(pred_factors, true_factors) / grad_accum_steps
      scaler.scale(loss).backward()
      if (step+1) % 2 == 0:                        # grad_accum_steps=2
        clip_grad_norm_(max=1.0)
        scaler.step(optimizer)
        scheduler.step()
        optimizer.zero_grad()

  validate():
    for batch in val_loader:
      output = model(**inputs)
      collect predictions + targets
    Compute per-factor: MAE, RMSE, R² for food/price/atmos/service/overall
    Compute mean_mae, aspect_mae, overall_mae

  if mean_mae improved:
    Save checkpoint → best_model_train_fusion.pth
    Reset patience counter
  else:
    patience_counter += 1
    if patience_counter >= 5: break  # early stopping
```

Final evaluation:
```
Load best checkpoint
validate(return_predictions=True) → final_metrics, final_preds, final_targets
Save metrics.json
Save predictions.csv (per-sample: index, y_true, y_pred, absolute_error)
```

**5. Save Artifacts to Drive**
```python
cp -r ./experiments/EXP_050C_bestfusion_logcosh/* /content/drive/MyDrive/SE365/experiments/EXP_050C_bestfusion_logcosh/
```

**6. Print Results**
```
=== EXP_050C_bestfusion_logcosh Results ===
Loss (val)   : 0.6413
  food     : MAE 1.1066  RMSE 1.5006  R² 0.5722
  price    : MAE 1.1694  RMSE 1.5671  R² 0.4502
  atmos    : MAE 1.1739  RMSE 1.5250  R² 0.4008
  service  : MAE 1.1770  RMSE 1.5697  R² 0.5194
  overall  : MAE 0.9130  RMSE 1.2254  R² 0.6312
  mean_mae : 1.1080
```

### Data Flow Through the Model (Single Forward Pass)

```
Input batch:
  input_ids:      [16, 256]          # tokenized text
  attention_mask: [16, 256]          # text mask
  pixel_values:   [16, 4, 3, 224, 224]  # up to 4 images per review
  num_images:     [16]               # actual image count per sample

TextModel.forward():
  AutoModel(input_ids, attention_mask)
  → pooler_output: [16, 768]        # CLS token feature
  → fc: [16, 768] → [16, 256]
  → returns (factor_head: [16, 5], features: [16, 256])

ImageModel.forward():
  pixel_values: [16, 4, 3, 224, 224] → reshape [64, 3, 224, 224]
  timm_model → [64, 1024]
  reshape → [16, 4, 1024]
  average_pool (masked by num_images) → [16, 1024]
  → fc: [16, 1024] → [16, 256]
  → returns (factor_head: [16, 5], features: [16, 256])

CrossAttentionFusion.forward():
  text_feat:  [16, 768]   (raw encoder feature, not fc output)
  image_feat: [16, 1024]  (raw encoder feature, not fc output)
  
  text_proj:  [16, 768] → [16, 512] → unsqueeze → [16, 1, 512]
  image_proj: [16, 1024] → [16, 512] → unsqueeze → [16, 1, 512]
  
  cross_attn_t2i(Q=text, K=image, V=image) → [16, 1, 512]
  cross_attn_i2t(Q=image, K=text, V=text)  → [16, 1, 512]
  
  concat → [16, 1024]
  head: [16, 1024] → [16, 512] → [16, 256] → [16, 5]  # 5 factor predictions
```
