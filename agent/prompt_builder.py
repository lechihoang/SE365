"""
Build system and user prompts for the AI Agent.

Constructs evidence-grounded prompts with explicit anti-hallucination
rules, causality warnings, and output schema instructions.
"""

import json
from typing import Dict, Any, Optional

from agent.config import AgentConfig


SYSTEM_PROMPT = """\
You are an AI explanation assistant for a multimodal restaurant review \
quality assessment system. You receive model predictions (5 scores on a \
1-10 scale) and structured XAI evidence from Grad-CAM, Attention, \
Cross-Attention, SHAP, and LIME analyses.

## STRICT RULES — FOLLOW EXACTLY

1. EVIDENCE GROUNDING: Every claim MUST reference specific evidence from \
the input. If no XAI evidence supports a claim, write: \
"Không có bằng chứng XAI trực tiếp cho nhận định này" (Vietnamese) or \
"No direct XAI evidence supports this claim" (English).

2. NO HALLUCINATION: NEVER invent evidence. If the review does not \
mention price, do NOT say "price was considered reasonable." If Grad-CAM \
data is missing, do NOT describe image regions.

3. NO CAUSALITY CLAIMS: Say "mô hình tập trung vào" (the model focused \
on) or "bằng chứng cho thấy" (evidence shows). NEVER say "vì" (because) \
or "nguyên nhân" (the cause) when linking evidence to scores.

4. ALL 5 TARGETS REQUIRED: You MUST explain ALL 5 scores: food, price, \
atmos, service, overall. For targets with weak evidence, explicitly state \
that evidence is limited.

5. SHAP TERMINOLOGY: Use "text-origin" and "image-origin", NOT "pure \
text" or "pure image". Cross-attention mixes modality information, so \
dims 0:512 are text-origin (text features after attending to image), not \
pure text.

6. SHAP PER-TARGET: When discussing SHAP, always specify WHICH target \
the percentages apply to. Never give percentages without naming the target.

7. CROSS-ATTENTION SPECIFICS: When cross-attention evidence is available, \
reference actual token names, patch coordinates (row,col), and attention \
scores from the provided data. Do not give generic descriptions.

8. RECOMMENDATIONS: Only suggest improvements that are directly supported \
by evidence in the review text or XAI artifacts. If the review does not \
mention a topic, do not recommend changes for it.

9. LIMITATIONS: Always include meaningful limitations. Required items:
   - XAI shows correlation, not causation
   - SHAP uses text-origin/image-origin grouping (not pure modality)
   - Attention reflects model focus, not necessarily true importance
   - This report explains model behavior, not objective restaurant quality

10. TECHNICAL TERMS: Keep these in English even in Vietnamese output: \
Grad-CAM, Cross-Attention, SHAP, LIME, text-origin, image-origin, \
Top-K, token, patch.

## Score level mapping (use these EXACT level strings):
- 0-2: "very_poor"
- 2-4: "poor"
- 4-6: "average"
- 6-8: "good"
- 8-10: "excellent"

## CONFIDENCE RULES:
- "high": 4-5 XAI methods available AND methods mostly agree
- "medium": 2-3 XAI methods available OR some disagreement
- "low": 0-1 XAI methods available OR major disagreement

You MUST respond with valid JSON matching the schema provided."""


_USER_TEMPLATE = """\
## Sample Information
- Sample ID: {sample_id}
- Review text: "{review_text}"
- Number of images: {num_images}

## Model Predictions
{predictions_block}

{ground_truth_block}

## XAI Evidence

### Grad-CAM (Image region importance)
{gradcam}

### PhoBERT Self-Attention (Token importance)
{attention}

### Cross-Attention — Token × Patch (Cross-modal links)
{cross_attention}

### SHAP Modality Contribution (Text-origin vs Image-origin per target)
{shap}

### LIME (Perturbation-based word/region importance)
{lime}

### Evidence Availability
{missing_summary}

## REQUIRED OUTPUT

Generate a JSON object in {language_name} with this structure.
You MUST include explanations for ALL 5 targets (food, price, atmos, \
service, overall). Do NOT skip any target.

```json
{schema_summary}
```

IMPORTANT REMINDERS:
- "level" must be one of: very_poor, poor, average, good, excellent
- Include "evidence_completeness" showing which XAI methods are available
- Include "per_target" SHAP breakdown in modality_contribution
- Include "confidence_reasoning" explaining your confidence choice
- Include "method_agreement" comparing what different XAI methods show
- Include meaningful "limitations" (at least 3 items)
- "recommendations" must ONLY reference evidence actually present"""


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
        """Build the messages list for an OpenAI chat completion call."""
        lang = language or self.config.language
        lang_name = 'Vietnamese' if lang == 'vi' else 'English'

        # Predictions table
        display = {
            'food_score': 'Food Score',
            'price_score': 'Price Score',
            'atmosphere_score': 'Atmosphere Score',
            'service_score': 'Service Score',
            'overall_satisfaction': 'Overall Satisfaction',
        }
        pred_lines = ['| Target | Predicted |', '|--------|-----------|']
        for key, name in display.items():
            val = predictions.get(key, 0)
            pred_lines.append(f'| {name} | {val:.2f} |')
        predictions_block = '\n'.join(pred_lines)

        # Ground truth
        if ground_truth:
            gt_lines = [
                '## Ground Truth & Error',
                '| Target | True | Predicted | Error |',
                '|--------|------|-----------|-------|',
            ]
            for key, name in display.items():
                gt = ground_truth.get(key, 0)
                pred = predictions.get(key, 0)
                err = abs(pred - gt)
                gt_lines.append(
                    f'| {name} | {gt:.2f} | {pred:.2f} | {err:.3f} |')
            ground_truth_block = '\n'.join(gt_lines)
        else:
            ground_truth_block = '(Ground truth not available)'

        # Schema example
        schema_summary = json.dumps({
            'sample_id': sample_id,
            'language': lang,
            'summary': '2-3 sentence overall interpretation',
            'scores': {
                'food': {
                    'score': 0.0, 'level': 'good',
                    'explanation': 'Evidence-grounded explanation',
                    'evidence': {
                        'gradcam': 'description or null',
                        'attention': ['token1 (score)', 'token2 (score)'],
                        'cross_attention': 'token→patch details or null',
                        'shap': {'text_origin_pct': 0, 'image_origin_pct': 0},
                        'lime_text': [{'word': 'x', 'weight': 0}],
                    },
                },
                'price': {'score': 0, 'level': 'average', 'explanation': '...'},
                'atmos': {'score': 0, 'level': 'good', 'explanation': '...'},
                'service': {'score': 0, 'level': 'good', 'explanation': '...'},
                'overall': {'score': 0, 'level': 'good', 'explanation': '...'},
            },
            'modality_contribution': {
                'text_origin_pct': 0, 'image_origin_pct': 0,
                'per_target': {
                    'food': {'text_origin_pct': 0, 'image_origin_pct': 0},
                },
                'interpretation': 'Which modality dominated and why',
            },
            'evidence_completeness': {
                'gradcam': True, 'attention': True,
                'cross_attention': True, 'shap': True, 'lime': True,
                'total': '5/5',
            },
            'cross_modal_insights': 'Specific token↔patch connections',
            'method_agreement': 'How XAI methods agree/disagree',
            'limitations': ['limitation 1', 'limitation 2', 'limitation 3'],
            'recommendations': ['evidence-grounded suggestion'],
            'confidence': 'high',
            'confidence_reasoning': 'Why this confidence level',
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
                f'\n\nThis sample is a **{case_type}** case study. '
                f'Tailor the explanation to highlight why this case type '
                f'was selected.')

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ]

        return {'messages': messages, 'language': lang}
