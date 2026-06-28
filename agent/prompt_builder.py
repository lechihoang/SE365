"""
Build system and user prompts for the AI Agent.

Constructs evidence-grounded prompts with explicit anti-hallucination
rules, causality warnings, and output schema instructions.
"""

import json
from typing import Dict, Any, Optional, List

from agent.config import AgentConfig
from agent.output_schema import AGENT_OUTPUT_SCHEMA


SYSTEM_PROMPT = """\
You are an AI explanation assistant for a multimodal restaurant review \
quality assessment system. You receive model predictions (5 scores on a \
1-10 scale) and structured XAI evidence from Grad-CAM, Attention, \
Cross-Attention, SHAP, and LIME analyses.

Your role:
- Translate numerical predictions and XAI evidence into clear \
natural-language explanations.
- Every statement must be grounded in the provided evidence.
- Never invent evidence that is not in the input data.
- If evidence for a claim is missing, say "Không có bằng chứng XAI \
cho phần này" (or "Evidence not available" in English).
- Use "text-origin" and "image-origin" for SHAP contributions, NOT \
"pure text" or "pure image", because cross-attended features contain \
information from both modalities.
- Do NOT claim causality. Say "mô hình tập trung vào" (the model \
focused on) rather than "vì" (because) when linking evidence to scores.
- Frame recommendations as suggestions based on model evidence, not \
definitive quality assessments.

Score interpretation guide:
- 1-3: Low (Thấp)
- 4-5: Below average (Dưới trung bình)
- 6-7: Average to good (Trung bình - Khá)
- 8-9: Good to excellent (Tốt - Xuất sắc)
- 10: Excellent (Xuất sắc)

You MUST respond with valid JSON matching the provided schema."""


_USER_TEMPLATE = """\
## Sample Information
- Sample ID: {sample_id}
- Review text: "{review_text}"
- Number of images: {num_images}

## Model Predictions
{predictions_block}

{ground_truth_block}

## XAI Evidence

### Grad-CAM
{gradcam}

### PhoBERT Self-Attention
{attention}

### Cross-Attention (Token × Patch)
{cross_attention}

### SHAP Modality Contribution
{shap}

### LIME
{lime}

### Evidence Availability
{missing_summary}

## Task
Generate a structured explanation in {language_name} following the \
JSON schema below. Include all required fields. For any score without \
evidence, state that evidence is not available instead of guessing.

```json
{schema_summary}
```"""


class PromptBuilder:
    """Constructs prompts for the OpenAI API call."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def build(
        self,
        sample_id: str,
        review_text: str,
        predictions: Dict[str, float],
        evidence_blocks: Dict[str, str],
        ground_truth: Optional[Dict[str, float]] = None,
        case_type: Optional[str] = None,
        language: Optional[str] = None,
        num_images: int = 1,
    ) -> Dict[str, Any]:
        """Build the messages list for an OpenAI chat completion call.

        Returns:
            Dict with 'messages' key containing the system + user messages.
        """
        lang = language or self.config.language
        lang_name = 'Vietnamese' if lang == 'vi' else 'English'

        # Predictions table
        pred_lines = ['| Target | Predicted |', '|--------|-----------|']
        display = {
            'food_score': 'Food Score', 'price_score': 'Price Score',
            'atmosphere_score': 'Atmosphere Score',
            'service_score': 'Service Score',
            'overall_satisfaction': 'Overall Satisfaction',
        }
        for key, name in display.items():
            val = predictions.get(key, 0)
            pred_lines.append(f'| {name} | {val:.2f} |')
        predictions_block = '\n'.join(pred_lines)

        # Ground truth block
        if ground_truth:
            gt_lines = [
                '## Ground Truth',
                '| Target | True | Error |',
                '|--------|------|-------|',
            ]
            for key, name in display.items():
                gt = ground_truth.get(key, 0)
                pred = predictions.get(key, 0)
                err = abs(pred - gt)
                gt_lines.append(f'| {name} | {gt:.2f} | {err:.3f} |')
            ground_truth_block = '\n'.join(gt_lines)
        else:
            ground_truth_block = '(Ground truth not available)'

        # Schema summary (compact)
        schema_summary = json.dumps({
            'sample_id': 'string',
            'language': lang,
            'summary': 'string (2-3 sentences)',
            'scores': {
                'food': {'score': 'number', 'level': 'string',
                         'explanation': 'string', 'evidence': '...'},
            },
            'modality_contribution': {
                'text_origin_pct': 'number',
                'image_origin_pct': 'number',
                'interpretation': 'string',
            },
            'cross_modal_insights': 'string',
            'method_agreement': 'string',
            'limitations': ['string'],
            'recommendations': ['string'],
            'confidence': 'low|medium|high',
        }, indent=2, ensure_ascii=False)

        user_content = _USER_TEMPLATE.format(
            sample_id=sample_id,
            review_text=review_text[:500],
            num_images=num_images,
            predictions_block=predictions_block,
            ground_truth_block=ground_truth_block,
            gradcam=evidence_blocks.get('gradcam', 'Not available'),
            attention=evidence_blocks.get('attention', 'Not available'),
            cross_attention=evidence_blocks.get(
                'cross_attention', 'Not available'),
            shap=evidence_blocks.get('shap', 'Not available'),
            lime=evidence_blocks.get('lime', 'Not available'),
            missing_summary=evidence_blocks.get(
                'missing_summary', 'Unknown'),
            language_name=lang_name,
            schema_summary=schema_summary,
        )

        if case_type:
            user_content += (
                f'\n\nThis sample was selected as a **{case_type}** '
                f'case study. Tailor the explanation accordingly.')

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ]

        return {'messages': messages, 'language': lang}
