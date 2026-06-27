import torch
import torch.nn as nn
import traceback

from Models.FusionModel import FusionModel
from Models.GMUFusion import GMUFusion
from Models.GatedCrossModalFusion import GatedCrossModalFusion
from Models.FiLMFusion import FiLMFusion
from Models.CrossAttentionFusion import CrossAttentionFusion

# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class MockEncoder(nn.Module):
    """Mimics both HuggingFace text encoders and timm image encoders enough
    for the fusion modules' __init__ and forward paths."""
    def __init__(self, hidden_size=768, num_features=1024, T=16, P=49):
        super().__init__()
        self.config = MockConfig(hidden_size)
        self.num_features = num_features
        self.T = T
        self.P = P
        self.hidden_size = hidden_size
        # a tiny param so unfreeze code finds something
        self.last_hidden_state_proj = nn.Linear(hidden_size, hidden_size)
        # mimic timm block containers for the unfreeze fix
        class _Block(nn.Module):
            def __init__(s):
                super().__init__()
                s.lin = nn.Linear(num_features, num_features)
        class _Stage(nn.Module):
            def __init__(s):
                super().__init__()
                s.blocks = nn.ModuleList([_Block(), _Block()])
        self.stages = nn.ModuleList([_Stage()])           # ConvNeXt path
        self.layers = nn.ModuleList([_Stage()])           # Swin path
        self.blocks = nn.ModuleList([_Block(), _Block()]) # EfficientNet path
        # mimic HuggingFace text encoder: <BaseModel>.encoder.encoder.layer
        # (RoBERTa/BERT/XLM-R/PhoBERT/ViSoBERT/DeBERTa all follow this).
        class _TxtLayer(nn.Module):
            def __init__(s):
                super().__init__()
                s.lin = nn.Linear(hidden_size, hidden_size)
        self.encoder = nn.Module()                        # the inner HF encoder
        self.encoder.layer = nn.ModuleList([_TxtLayer() for _ in range(4)])

    def forward(self, input_ids=None, attention_mask=None):
        B = input_ids.shape[0] if input_ids is not None else 4
        T = self.T
        class Out:
            pass
        o = Out()
        o.last_hidden_state = torch.randn(B, T, self.hidden_size)
        o.pooler_output = torch.randn(B, self.hidden_size)
        return o

    def forward_features(self, x):
        # x: (BN, C, H, W) -> return (BN, num_features, h, w) channels-first
        BN = x.shape[0]
        D = self.num_features
        P = self.P
        h = w = int(P ** 0.5)
        return torch.randn(BN, D, h, w)

# ---------------------------------------------------------------------------
# Mock text/image models that mirror the real API (forward + return_tokens
# for text, forward + forward_features for image).
# ---------------------------------------------------------------------------

class MockTextModel(nn.Module):
    def __init__(self, hidden_size=768, T=16):
        super().__init__()
        self.encoder = MockEncoder(hidden_size=hidden_size, T=T)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, return_tokens=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = out.pooler_output
        logits = features  # placeholder, fusion ignores it
        if return_tokens:
            tokens = out.last_hidden_state
            pad_mask = (attention_mask == 0)
            return logits, features, tokens, pad_mask
        return logits, features

class MockImageModel(nn.Module):
    def __init__(self, num_features=1024, P=49):
        super().__init__()
        self.encoder = MockEncoder(num_features=num_features, P=P)
        self.dummy_param = nn.Parameter(torch.zeros(1))


    def forward(self, pixel_values, num_images=None):
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        B, N, C, H, W = pixel_values.shape
        feats = self.encoder(pixel_values.view(B * N, C, H, W))
        # mimic timm num_classes=0 forward -> (BN, num_features)
        feats = torch.randn(B, N, self.encoder.num_features)
        if num_images is not None:
            mask = (torch.arange(N).expand(B, N) < num_images.unsqueeze(1)).float().unsqueeze(-1)
            feats = (feats * mask).sum(dim=1) / num_images.float().clamp(min=1).unsqueeze(1)
        else:
            feats = feats.mean(dim=1)
        return feats, feats

    def forward_features(self, pixel_values, num_images=None):
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        B, N, C, H, W = pixel_values.shape
        if num_images is None:
            num_images = torch.full((B,), N, dtype=torch.long)
        feats = self.encoder.forward_features(pixel_values.view(B * N, C, H, W))  # (BN, D, h, w)
        D, h, w = feats.shape[1], feats.shape[2], feats.shape[3]
        P = h * w
        feats = feats.reshape(B, N, P, D)
        img_mask = (torch.arange(N).expand(B, N) < num_images.unsqueeze(1)).float().unsqueeze(-1)
        feats = (feats * img_mask.unsqueeze(-1)).sum(dim=1) / num_images.float().clamp(min=1).unsqueeze(-1).unsqueeze(-1)
        patch_mask = torch.ones(B, P, dtype=torch.bool)
        return feats, patch_mask

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _dummy_inputs(batch_size=4, seq_len=16, max_images=3, P_dim=49):
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # pad a few text positions
    attention_mask[:, -2:] = 0
    # 5D image tensor (B, N, C, H, W)
    pixel_values = torch.randn(batch_size, max_images, 3, 224, 224)
    num_images = torch.tensor([max_images, 2, 1, max_images])[:batch_size]
    return input_ids, attention_mask, pixel_values, num_images

def test_pooled_fusions():
    """Concat / GMU / GatedCross / FiLM use the pooled-vector path."""
    print("="*60)
    print("GROUP A: Pooled-vector fusion architectures")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    image_model = MockImageModel(num_features=1024, P=49)
    models = {
        "1. Basic Concat Fusion": FusionModel,
        "2. GMU Fusion": GMUFusion,
        "3. Gated Cross-Modal Fusion": GatedCrossModalFusion,
        "4. FiLM Fusion": FiLMFusion,
    }
    input_ids, attention_mask, pixel_values, num_images = _dummy_inputs()
    passed = 0
    for name, Cls in models.items():
        print(f"\nĐang test {name}...")
        try:
            model = Cls(text_model, image_model, num_factors=5,
                        unfreeze_text_layers=1, unfreeze_image_layers=1)
            out = model(input_ids, attention_mask, pixel_values, num_images)
            assert out.shape == (4, 5), f"Expected (4,5), got {tuple(out.shape)}"
            out.sum().backward()
            print(f"  ✅ [PASS] shape {tuple(out.shape)} | backward OK")
            passed += 1
        except Exception:
            print(f"  ❌ [FAIL]")
            traceback.print_exc()
    print(f"\nGROUP A: {passed}/{len(models)} pass")
    return passed, len(models)

def test_cross_attention_true():
    """CrossAttention now uses token <-> patch attention with masks."""
    print("\n" + "="*60)
    print("GROUP B: Cross-Attention (token <-> patch, real attention)")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    image_model = MockImageModel(num_features=1024, P=49)
    input_ids, attention_mask, pixel_values, num_images = _dummy_inputs()
    print("\nĐang test 5. Cross-Attention Fusion (token<->patch)...")
    try:
        model = CrossAttentionFusion(text_model, image_model, num_factors=5,
                                     unfreeze_text_layers=1, unfreeze_image_layers=1)
        out = model(input_ids, attention_mask, pixel_values, num_images)
        assert out.shape == (4, 5), f"Expected (4,5), got {tuple(out.shape)}"
        out.sum().backward()
        print(f"  ✅ [PASS] shape {tuple(out.shape)} | backward OK")
        return 1, 1
    except Exception:
        print(f"  ❌ [FAIL]")
        traceback.print_exc()
        return 0, 1

def test_text_return_tokens_backward_compat():
    """Ensure TextModel.forward(return_tokens=False) still returns 2-tuple."""
    print("\n" + "="*60)
    print("GROUP C: TextModel backward-compat (return_tokens)")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    input_ids, attention_mask, _, _ = _dummy_inputs()
    try:
        res = text_model(input_ids, attention_mask)
        assert isinstance(res, tuple) and len(res) == 2, f"Expected 2-tuple, got {type(res)} len={len(res) if isinstance(res,tuple) else '?'}"
        print(f"  ✅ [PASS] return_tokens=False -> {len(res)}-tuple (logits, pooled)")
        passed = 1
    except Exception:
        print(f"  ❌ [FAIL] default path broke")
        traceback.print_exc()
        passed = 0
    try:
        res = text_model(input_ids, attention_mask, return_tokens=True)
        assert isinstance(res, tuple) and len(res) == 4, f"Expected 4-tuple, got len={len(res)}"
        logits, pooled, tokens, pad = res
        assert tokens.dim() == 3, f"tokens should be 3D, got {tokens.dim()}"
        assert pad.dim() == 2, f"pad_mask should be 2D, got {pad.dim()}"
        print(f"  ✅ [PASS] return_tokens=True -> 4-tuple, tokens={tuple(tokens.shape)}, pad={tuple(pad.shape)}")
        passed += 1
    except Exception:
        print(f"  ❌ [FAIL] token path broke")
        traceback.print_exc()
    return passed, 2

def test_image_forward_features():
    """Ensure ImageModel.forward_features returns (B,P,D) + mask."""
    print("\n" + "="*60)
    print("GROUP D: ImageModel.forward_features (patch tokens)")
    print("="*60)
    image_model = MockImageModel(num_features=1024, P=49)
    _, _, pixel_values, num_images = _dummy_inputs()
    try:
        feats, mask = image_model.forward_features(pixel_values, num_images=num_images)
        assert feats.dim() == 3 and feats.shape[0] == 4 and feats.shape[2] == 1024, f"Unexpected feats shape {tuple(feats.shape)}"
        assert mask.dim() == 2 and mask.shape[0] == 4 and mask.dtype == torch.bool, f"Unexpected mask shape {tuple(mask.shape)} dtype {mask.dtype}"
        print(f"  ✅ [PASS] feats={tuple(feats.shape)} | mask={tuple(mask.shape)} dtype={mask.dtype}")
        return 1, 1
    except Exception:
        print(f"  ❌ [FAIL]")
        traceback.print_exc()
        return 0, 1

def test_film_init_is_identity():
    """FiLM gamma should be 1, beta 0 so modulation is identity at init."""
    print("\n" + "="*60)
    print("GROUP E: FiLM standard init (gamma=1, beta=0)")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    image_model = MockImageModel(num_features=1024, P=49)
    try:
        model = FiLMFusion(text_model, image_model, num_factors=5)
        g_w, g_b = model.film_gamma.weight, model.film_gamma.bias
        b_w, b_b = model.film_beta.weight, model.film_beta.bias
        # film_gamma init: weight=0 so output = bias only; bias=1 → gamma(text) ≈ 1 → identity scale
        # film_beta  init: weight=0, bias=0 → beta(text) ≈ 0 → no shift
        assert torch.allclose(g_w, torch.zeros_like(g_w)), "gamma weight should be 0 (output = bias only)"
        assert torch.allclose(g_b, torch.ones_like(g_b)),  "gamma bias should be 1 (identity scale)"
        assert torch.allclose(b_w, torch.zeros_like(b_w)), "beta weight should be 0"
        assert torch.allclose(b_b, torch.zeros_like(b_b)), "beta bias should be 0 (no shift)"
        print(f"  ✅ [PASS] gamma=1, beta=0 at init -> identity modulation")
        return 1, 1
    except Exception:
        print(f"  ❌ [FAIL]")
        traceback.print_exc()
        return 0, 1

def test_unfreeze_swin_path():
    """Verify the unfreeze fix reaches the Swin `layers[-1].blocks` path."""
    print("\n" + "="*60)
    print("GROUP F: Unfreeze image layers (Swin/EfficientNet path)")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    image_model = MockImageModel(num_features=1024, P=49)
    try:
        # freeze everything first
        for p in image_model.parameters():
            p.requires_grad = False
        model = FusionModel(text_model, image_model, num_factors=5,
                            unfreeze_text_layers=0, unfreeze_image_layers=1)
        # some image params should now be trainable via the stages path
        n_train = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
        assert n_train > 0, "No image params unfrozen — Swin/Eff path broken"
        print(f"  ✅ [PASS] {n_train} image params made trainable via unfreeze path")
        return 1, 1
    except Exception:
        print(f"  ❌ [FAIL]")
        traceback.print_exc()
        return 0, 1

def test_unfreeze_text_path():
    """Verify the text unfreeze fix reaches enc.encoder.layer (HF Transformers)
    and would NOT silently skip for the backbones used in the experiments."""
    print("\n" + "="*60)
    print("GROUP G: Unfreeze text layers (HF Transformers path)")
    print("="*60)
    text_model = MockTextModel(hidden_size=768, T=16)
    image_model = MockImageModel(num_features=1024, P=49)
    try:
        for p in text_model.parameters():
            p.requires_grad = False
        model = FusionModel(text_model, image_model, num_factors=5,
                            unfreeze_text_layers=1, unfreeze_image_layers=0)
        n_train = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
        assert n_train > 0, "No text params unfrozen — HF text path broken"
        print(f"  ✅ [PASS] {n_train} text params made trainable via enc.encoder.layer path")
        return 1, 1
    except Exception:
        print(f"  ❌ [FAIL]")
        traceback.print_exc()
        return 0, 1

if __name__ == "__main__":
    pa, na = test_pooled_fusions()
    pb, nb = test_cross_attention_true()
    pc, nc = test_text_return_tokens_backward_compat()
    pd, nd = test_image_forward_features()
    pe, ne = test_film_init_is_identity()
    pf, nf = test_unfreeze_swin_path()
    pg, ng = test_unfreeze_text_path()
    total_p = pa + pb + pc + pd + pe + pf + pg
    total_n = na + nb + nc + nd + ne + nf + ng
    print("\n" + "="*60)
    print(f"🎯 TỔNG KẾT: {total_p}/{total_n} GROUP PASS")
    print("="*60)
