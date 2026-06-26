"""
Core utility functions for all XAI phases (1-8).

Provides model loading, single-sample inference, artifact saving, and
preprocessing functions that replicate the exact training-time pipeline.

Usage:
    from xai.utils import load_model, load_single_sample, get_prediction
    model, config = load_model(exp_dir)
    sample = load_single_sample(csv_path, idx=0, tokenizer=tok, image_processor=proc)
    result = get_prediction(model, sample)
"""

import os
import sys
import json
import ast
import hashlib
import random
import datetime
import logging
from typing import Optional, Dict, Any, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from xai.config import (
    TARGET_NAMES, FACTOR_NAMES, LABEL_COLS, INDEX_TO_FACTOR,
    DEFAULT_SEED, DEFAULT_DPI, DEFAULT_MAX_LENGTH, DEFAULT_MAX_IMAGES,
    BEST_TEXT_MODEL, BEST_IMAGE_MODEL, BEST_FUSION_TYPE,
)

logger = logging.getLogger('xai')


# ══════════════════════════════════════════════════════════════════════════════
# TimmProcessor — exact replica of main.py (lines 13-22) for Swin-B / ConvNeXt
# ══════════════════════════════════════════════════════════════════════════════

class _TimmProcessor:
    """Wrapper to make timm transforms behave like HuggingFace AutoImageProcessor.
    Replicated from main.py to avoid importing main.py (which triggers argparse).
    """
    def __init__(self, model_name: str):
        import timm
        data_config = timm.data.resolve_model_data_config(model_name)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

    def __call__(self, images, return_tensors: str = "pt"):
        pixel_values = torch.stack([self.transform(img.convert('RGB')) for img in images])
        return {'pixel_values': pixel_values}


# ══════════════════════════════════════════════════════════════════════════════
# Device & Seed — matches main.py (lines 24-34, 51-54)
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """Auto-detect best available device (cuda > mps > cpu).
    Matches main.py and test.py device detection logic.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set all random seeds for reproducibility.
    Exact replica of main.py set_seed() (lines 24-34).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer & Image Processor
# ══════════════════════════════════════════════════════════════════════════════

def get_tokenizer(text_model_name: str = BEST_TEXT_MODEL):
    """Load HuggingFace tokenizer. Matches main.py line 60."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(text_model_name)


def get_image_processor(image_model_name: str = BEST_IMAGE_MODEL):
    """Load image processor with the same fallback chain as main.py (lines 61-69).

    Priority:
      1. AutoImageProcessor.from_pretrained()
      2. SigLIP-specific fallback
      3. TimmProcessor (for timm models like Swin-B, ConvNeXt)
    """
    from transformers import AutoImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(image_model_name)
    except Exception:
        if 'siglip' in image_model_name.lower():
            processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-256')
            print(f'[XAI] Loaded HuggingFace processor for {image_model_name}')
            return processor
        else:
            processor = _TimmProcessor(image_model_name)
            print(f'[XAI] Loaded timm processor for {image_model_name}')
            return processor


# ══════════════════════════════════════════════════════════════════════════════
# Config Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(exp_dir: str) -> Dict[str, Any]:
    """Load experiment config from yaml, json, or checkpoint args.

    Searches in order: config.yaml, config.json, checkpoint 'args' key.
    Returns empty dict if nothing found.
    """
    config_yaml = os.path.join(exp_dir, 'config.yaml')
    config_json = os.path.join(exp_dir, 'config.json')

    if os.path.isfile(config_yaml):
        try:
            import yaml
            with open(config_yaml, 'r') as f:
                config = yaml.safe_load(f)
            print(f'[XAI] Loaded config: {config_yaml}')
            return config or {}
        except ImportError:
            pass

    if os.path.isfile(config_json):
        with open(config_json, 'r') as f:
            config = json.load(f)
        print(f'[XAI] Loaded config: {config_json}')
        return config

    # fallback: extract from checkpoint
    ckpt_path = os.path.join(exp_dir, 'best_model_train_fusion.pth')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        config = ckpt.get('args', {})
        del ckpt
        if config:
            print(f'[XAI] Extracted config from checkpoint: {ckpt_path}')
            return config

    print('[XAI] WARNING: No config file found. Using defaults.')
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading — matches test.py (lines 59-97)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    exp_dir: str,
    device: Optional[torch.device] = None,
    text_model_name: Optional[str] = None,
    image_model_name: Optional[str] = None,
    fusion_type: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load trained multimodal model from experiment directory.

    Replicates the model construction and checkpoint loading logic from test.py.

    Args:
        exp_dir: Path to experiment folder (e.g., experiments/EXP_060A_...)
        device: Target device. Auto-detected if None.
        text_model_name: Override text backbone. Read from config if None.
        image_model_name: Override image backbone. Read from config if None.
        fusion_type: Override fusion type. Read from config if None.

    Returns:
        Tuple of (model in eval mode, config dict)

    Raises:
        FileNotFoundError: If checkpoint file is missing.
        RuntimeError: If state_dict keys don't match model architecture.
    """
    if device is None:
        device = get_device()

    config = _load_config(exp_dir)

    # fill from config, then fall back to best-model defaults
    text_model_name = text_model_name or config.get('text_model_name', BEST_TEXT_MODEL)
    image_model_name = image_model_name or config.get('image_model_name', BEST_IMAGE_MODEL)
    fusion_type = fusion_type or config.get('fusion_type', BEST_FUSION_TYPE)

    print(f'[XAI] Building model: text={text_model_name}, image={image_model_name}, fusion={fusion_type}')

    # build sub-models (same as test.py lines 70-71)
    from Models.TextModel import TextModel
    from Models.ImageModel import ImageModel

    text_model = TextModel(model_name=text_model_name)
    image_model = ImageModel(model_name=image_model_name)

    # build fusion model (same as test.py lines 73-87)
    fusion_kwargs = dict(text_model=text_model, image_model=image_model)
    if fusion_type == 'gmu':
        from Models.GMUFusion import GMUFusion
        model = GMUFusion(**fusion_kwargs)
    elif fusion_type == 'gated_cross':
        from Models.GatedCrossModalFusion import GatedCrossModalFusion
        model = GatedCrossModalFusion(**fusion_kwargs)
    elif fusion_type == 'film':
        from Models.FiLMFusion import FiLMFusion
        model = FiLMFusion(**fusion_kwargs)
    elif fusion_type == 'cross_attention':
        from Models.CrossAttentionFusion import CrossAttentionFusion
        model = CrossAttentionFusion(**fusion_kwargs)
    else:
        from Models.FusionModel import FusionModel
        model = FusionModel(**fusion_kwargs)

    # load checkpoint (same as test.py load_ckpt, lines 26-29)
    ckpt_path = os.path.join(exp_dir, 'best_model_train_fusion.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            f'Make sure the experiment folder contains best_model_train_fusion.pth'
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # handle DataParallel 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        print('[XAI] WARNING: Stripping "module." prefix from state_dict keys (DataParallel checkpoint)')
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    # store checkpoint metadata in config for downstream use
    config['_checkpoint_path'] = ckpt_path
    config['_best_mean_mae'] = ckpt.get('best_mean_mae', None)
    config['_best_val_loss'] = ckpt.get('best_val_loss', None)
    config['_checkpoint_epoch'] = ckpt.get('epoch', None)

    del ckpt

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[XAI] Model loaded: {model.__class__.__name__} ({total_params:,} params) on {device}')

    return model, config


# ══════════════════════════════════════════════════════════════════════════════
# Single-Sample Loading — replicates src/dataset.py MultimodalDataset.__getitem__
# ══════════════════════════════════════════════════════════════════════════════

def _load_image_from_cache(url: str, image_dir: str) -> Image.Image:
    """Load image from local MD5-hashed cache.
    Matches src/dataset.py _load_image() but without network download for safety.
    """
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    local_path = os.path.join(image_dir, f'{url_hash}.jpg')
    if os.path.exists(local_path):
        return Image.open(local_path).convert('RGB')
    # fallback: black placeholder (same as dataset.py line 40)
    return Image.new('RGB', (224, 224), color='black')


def load_single_sample(
    csv_path: str,
    idx: int,
    tokenizer,
    image_processor,
    max_length: int = DEFAULT_MAX_LENGTH,
    max_images: int = DEFAULT_MAX_IMAGES,
    image_dir: str = './data/image',
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load and preprocess one sample for inference.

    Replicates the exact preprocessing from src/dataset.py
    MultimodalDataset.__getitem__() to ensure numerical consistency.

    Args:
        csv_path: Path to data CSV (train/val/test).
        idx: Row index in the CSV.
        tokenizer: HuggingFace tokenizer.
        image_processor: Image processor (AutoImageProcessor or TimmProcessor).
        max_length: Max token sequence length. Default 256 (from Config.py).
        max_images: Max images per review. Default 4 (from dataset.py).
        image_dir: Directory with cached image files.
        device: Target device. Auto-detected if None.

    Returns:
        Dictionary containing:
          - input_ids [1, seq_len], attention_mask [1, seq_len]
          - pixel_values [1, max_images, C, H, W], num_images [1]
          - factor_scores [5], text (str), image_urls (list)
          - loaded_images (list[PIL.Image]), num_real_images (int)
          - sample_idx (int), csv_path (str)
    """
    if device is None:
        device = get_device()

    df = pd.read_csv(csv_path)
    if idx >= len(df):
        raise IndexError(f'Sample index {idx} out of range (dataset has {len(df)} rows)')

    row = df.iloc[idx]

    # ── Text (same as dataset.py lines 46-53) ──
    text = str(row['comment_clean'])
    text_inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
    )

    # ── Images (same as dataset.py lines 55-71) ──
    try:
        image_urls = ast.literal_eval(row['image_url'])
    except Exception:
        image_urls = [str(row['image_url'])]

    num_real_images = min(len(image_urls), max_images)

    loaded_images = []
    images_for_processing = []
    for i in range(max_images):
        if i < len(image_urls):
            img = _load_image_from_cache(image_urls[i], image_dir)
        else:
            img = Image.new('RGB', (224, 224), color='black')
        if i < num_real_images:
            loaded_images.append(img)
        images_for_processing.append(img)

    image_inputs = image_processor(images_for_processing, return_tensors='pt')['pixel_values']

    # ── Labels (same as dataset.py lines 74-80) ──
    factor_scores = torch.tensor([
        row['food_score'],
        row['price_score'],
        row['atmosphere_score'],
        row['service_score'],
        row['overall_satisfaction'],
    ], dtype=torch.float)

    # ── Add batch dimension and move to device ──
    return {
        'input_ids': text_inputs['input_ids'].to(device),                         # [1, seq_len]
        'attention_mask': text_inputs['attention_mask'].to(device),               # [1, seq_len]
        'pixel_values': image_inputs.unsqueeze(0).to(device),                     # [1, max_images, C, H, W]
        'num_images': torch.tensor([num_real_images], dtype=torch.long).to(device),  # [1]
        'factor_scores': factor_scores,                                           # [5]
        'text': text,
        'image_urls': image_urls[:num_real_images],
        'loaded_images': loaded_images,
        'num_real_images': num_real_images,
        'sample_idx': idx,
        'csv_path': csv_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prediction — matches test.py (lines 105-121)
# ══════════════════════════════════════════════════════════════════════════════

def get_prediction(model: nn.Module, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run single-sample inference and return structured results.

    Args:
        model: Trained model in eval mode.
        sample_dict: Output from load_single_sample().

    Returns:
        Dictionary with predictions, ground_truth, absolute_errors (per target),
        predictions_array, ground_truth_array, and mean_mae.
    """
    with torch.no_grad():
        output = model(
            input_ids=sample_dict['input_ids'],
            attention_mask=sample_dict['attention_mask'],
            pixel_values=sample_dict['pixel_values'],
            num_images=sample_dict['num_images'],
        )
        # fusion models return tensor, unimodal return tuple (test.py line 116)
        preds = output[0] if isinstance(output, tuple) else output

    preds_np = preds.cpu().numpy().flatten()  # (5,)
    gt_np = sample_dict['factor_scores'].cpu().numpy().flatten()  # (5,)
    errors_np = np.abs(preds_np - gt_np)

    predictions = {}
    ground_truth = {}
    absolute_errors = {}
    for i, name in enumerate(TARGET_NAMES):
        predictions[name] = float(preds_np[i])
        ground_truth[name] = float(gt_np[i])
        absolute_errors[name] = float(errors_np[i])

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'absolute_errors': absolute_errors,
        'predictions_array': preds_np,
        'ground_truth_array': gt_np,
        'mean_mae': float(errors_np.mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Artifact Saving
# ══════════════════════════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def save_figure(
    fig,
    save_path: str,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> str:
    """Save matplotlib figure with consistent settings.

    Args:
        fig: matplotlib Figure object.
        save_path: Destination path (creates parent dirs automatically).
        dpi: Resolution. Use DEFAULT_DPI (150) for drafts, THESIS_DPI (300) for thesis.
        close: Whether to close the figure after saving.

    Returns:
        The save_path for confirmation.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if close:
        plt.close(fig)

    print(f'Saved: {save_path}')
    return save_path


def save_raw_values(data: Any, save_path: str) -> str:
    """Save raw numerical data in JSON, NPY, NPZ, or CSV format.

    Format is auto-detected from file extension.

    Args:
        data: Data to save. Dict for JSON/CSV, ndarray for NPY, dict of arrays for NPZ.
        save_path: Destination path (creates parent dirs automatically).

    Returns:
        The save_path for confirmation.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ext = os.path.splitext(save_path)[1].lower()
    if ext == '.json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder, ensure_ascii=False)
    elif ext == '.npy':
        np.save(save_path, data)
    elif ext == '.npz':
        np.savez(save_path, **data)
    elif ext == '.csv':
        pd.DataFrame(data).to_csv(save_path, index=False)
    else:
        raise ValueError(f'Unsupported format: {ext}. Use .json, .npy, .npz, or .csv')

    print(f'Saved: {save_path}')
    return save_path


def get_metadata(
    exp_id: str = '',
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    seed: int = DEFAULT_SEED,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Generate standard metadata dict for any XAI artifact.

    Args:
        exp_id: Experiment identifier.
        config: Experiment config dict (from load_model).
        device: Torch device used.
        seed: Random seed used.
        extra: Additional key-value pairs to include.

    Returns:
        Metadata dictionary with timestamp, versions, experiment info.
    """
    meta = {
        'experiment_id': exp_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'seed': seed,
        'device': str(device) if device else str(get_device()),
        'torch_version': torch.__version__,
        'target_names': TARGET_NAMES,
        'factor_names': FACTOR_NAMES,
    }

    if config:
        meta['text_model'] = config.get('text_model_name', '')
        meta['image_model'] = config.get('image_model_name', '')
        meta['fusion_type'] = config.get('fusion_type', '')
        meta['loss_fn'] = config.get('loss_fn', '')
        meta['checkpoint_best_mean_mae'] = config.get('_best_mean_mae', None)

    # attempt git commit hash
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta['git_commit'] = result.stdout.strip()
    except Exception:
        pass

    if extra:
        meta.update(extra)

    return meta
