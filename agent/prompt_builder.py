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

1. EVIDENCE GROUNDING: Every claim MUST reference specific evidence \
from the input. If no XAI evidence supports a claim, write: \
"Không có bằng chứng XAI trực tiếp" (Vietnamese) or \
"No direct XAI evidence available" (English). \
NEVER leave an evidence field as null — use empty string "" instead.

2. NO HALLUCINATION: NEVER invent evidence. Do NOT use speculative \
phrases like "the model may have..." or "this could indicate...". \
If the review does not mention a topic, do NOT describe model behavior \
for that topic. Only state what the evidence explicitly shows.

3. NO CAUSALITY: Say "mô hình tập trung vào" (the model focused on). \
NEVER say "vì" (because) or "nguyên nhân" (the cause).

4. ALL 5 TARGETS REQUIRED: Explain ALL 5 scores: food, price, atmos, \
service, overall. For targets with no evidence, write: \
"Không có bằng chứng XAI trực tiếp cho target này. Điểm dự đoán \
dựa trên learned representation của mô hình."

5. SHAP TERMINOLOGY: Use "text-origin" and "image-origin", NOT "pure \
text" or "pure image". Always specify WHICH target the SHAP \
percentages refer to.

6. CROSS-ATTENTION SPECIFICS: Reference actual token names, patch \
coordinates (row,col), and attention scores from the data. Example: \
"Token 'ngon' → Patch(3,4) với attention score 0.087". Do NOT give \
generic descriptions like "the model focuses on keywords".

7. RECOMMENDATIONS: Only suggest improvements directly supported by \
evidence in the review text or XAI artifacts. If the review never \
mentions a topic (e.g., price), do NOT recommend changes for it.

8. LIMITATIONS (at least 4 items, include ALL of these):
   - XAI cho thấy tương quan, không phải nhân quả
   - SHAP dùng text-origin/image-origin (không phải pure modality)
   - Attention phản ánh model focus, không nhất thiết là importance thực
   - Báo cáo giải thích hành vi mô hình, không phải chất lượng thực tế

9. TECHNICAL TERMS in English even in Vietnamese output: Grad-CAM, \
Cross-Attention, SHAP, LIME, text-origin, image-origin, token, patch.

10. CUSTOMER VIEW: Include a "customer_view" object with: \
a simple Vietnamese summary (no technical terms), key highlights \
(list of short bullet points), and simple recommendations.

## Score levels (use EXACT strings):
- 0-2: "very_poor" — 2-4: "poor" — 4-6: "average"
- 6-8: "good" — 8-10: "excellent"

## CONFIDENCE:
- "high": 4-5 XAI methods available AND mostly agree
- "medium": 2-3 methods OR some disagreement
- "low": 0-1 methods OR major disagreement
Include "confidence_reasoning" explaining the determination.

## METHOD AGREEMENT:
Compare what Grad-CAM, Attention, Cross-Attention, SHAP, and LIME \
each show. State whether they agree, partially agree, or disagree. \
Mention which specific methods support which conclusions.

Respond with valid JSON only. No null values — use "" for empty strings."""


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

### SHAP Modality Contribution (per target)
{shap}

### LIME (Perturbation-based importance)
{lime}

### Evidence Availability
{missing_summary}

## REQUIRED OUTPUT (JSON)

You MUST include ALL of these in your response:
- "scores" with ALL 5 targets (food, price, atmos, service, overall)
- "modality_contribution" with "per_target" SHAP for each target
- "evidence_completeness" (boolean per method)
- "cross_modal_insights" referencing specific tokens and patches
- "method_agreement" comparing all available XAI methods
- "limitations" (at least 4 items)
- "recommendations" (evidence-grounded only)
- "confidence" + "confidence_reasoning"
- "customer_view" with simple summary, highlights, recommendations
- All string fields must be "" not null

Output language: {language_name}

Schema:
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
        """Build the messages list for an OpenAI chat completion call."""
        lang = language or self.config.language
        lang_name = 'Vietnamese' if lang == 'vi' else 'English'

        display = {
            'food_score': 'Food Score',
            'price_score': 'Price Score',
            'atmosphere_score': 'Atmosphere Score',
            'service_score': 'Service Score',
            'overall_satisfaction': 'Overall Satisfaction',
        }

        pred_lines = ['| Target | Predicted |', '|--------|-----------|']
        for key, name in display.items():
            pred_lines.append(f'| {name} | {predictions.get(key, 0):.2f} |')
        predictions_block = '\n'.join(pred_lines)

        if ground_truth:
            gt_lines = [
                '## Ground Truth & Error',
                '| Target | True | Predicted | Error |',
                '|--------|------|-----------|-------|',
            ]
            for key, name in display.items():
                gt = ground_truth.get(key, 0)
                pred = predictions.get(key, 0)
                gt_lines.append(
                    f'| {name} | {gt:.2f} | {pred:.2f} '
                    f'| {abs(pred - gt):.3f} |')
            ground_truth_block = '\n'.join(gt_lines)
        else:
            ground_truth_block = '(Ground truth not available)'

        schema_summary = json.dumps({
            'sample_id': sample_id, 'language': lang,
            'summary': '2-3 sentence interpretation',
            'scores': {
                'food': {'score': 0, 'level': 'good',
                         'explanation': 'grounded explanation',
                         'evidence': {'gradcam': '', 'attention': [],
                                      'cross_attention': '',
                                      'shap': {'text_origin_pct': 0,
                                               'image_origin_pct': 0},
                                      'lime_text': []}},
                'price': '...', 'atmos': '...', 'service': '...',
                'overall': '...',
            },
            'modality_contribution': {
                'text_origin_pct': 0, 'image_origin_pct': 0,
                'per_target': {'food': {'text_origin_pct': 0,
                                        'image_origin_pct': 0}},
                'interpretation': 'string',
            },
            'evidence_completeness': {
                'gradcam': True, 'attention': True,
                'cross_attention': True, 'shap': True, 'lime': True,
                'total': '5/5',
            },
            'cross_modal_insights': 'specific token-patch connections',
            'method_agreement': 'how methods agree/disagree',
            'limitations': ['item1', 'item2', 'item3', 'item4'],
            'recommendations': ['evidence-grounded only'],
            'confidence': 'high',
            'confidence_reasoning': 'why',
            'customer_view': {
                'summary': 'simple Vietnamese summary',
                'highlights': ['point 1', 'point 2'],
                'recommendations': ['simple suggestion'],
            },
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
                f'\n\nThis is a **{case_type}** case study. '
                f'Tailor explanation accordingly.')

        return {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content},
            ],
            'language': lang,
        }
