"""
AI Agent for generating natural-language explanations from model
predictions and XAI evidence.

Usage:
    from agent import ExplanationAgent, AgentConfig

    config = AgentConfig(batch_model='gpt-4o-mini', report_model='gpt-4o')
    agent = ExplanationAgent(config)

    result = agent.explain_sample(
        sample_id='sample_0000',
        review_text='Đồ ăn ngon nhưng giá hơi cao...',
        predictions={'food_score': 8.2, 'price_score': 6.1, ...},
        xai_dir='/path/to/xai',
        language='vi',
    )
"""

import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional

from agent.config import AgentConfig
from agent.evidence_loader import EvidenceLoader
from agent.evidence_builder import EvidenceBuilder
from agent.prompt_builder import PromptBuilder
from agent.openai_client import OpenAIClient
from agent.output_schema import AGENT_OUTPUT_SCHEMA
from agent.report_generator import ReportGenerator
from agent.validator import OutputValidator


class ExplanationAgent:
    """High-level AI Agent for generating natural-language explanations.

    Orchestrates evidence loading, compression, prompt building,
    OpenAI API calls, validation, and report generation.

    Args:
        config: AgentConfig with model names, API key, and parameters.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.evidence_loader = EvidenceLoader()
        self.evidence_builder = EvidenceBuilder(self.config)
        self.prompt_builder = PromptBuilder(self.config)
        self.client = OpenAIClient(self.config)
        self.validator = OutputValidator()
        self.reporter = ReportGenerator()

    def explain_sample(
        self,
        sample_id: str,
        review_text: str,
        predictions: Dict[str, float],
        xai_dir: str,
        ground_truth: Optional[Dict[str, float]] = None,
        case_type: Optional[str] = None,
        language: Optional[str] = None,
        mode: str = 'text_only',
        num_images: int = 1,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a natural-language explanation for a single sample.

        Args:
            sample_id: e.g. 'sample_0042'.
            review_text: The Vietnamese review text.
            predictions: Dict of predicted scores keyed by TARGET_NAMES.
            xai_dir: Base XAI directory with phase subdirectories.
            ground_truth: Optional ground truth scores.
            case_type: Optional case study type for context.
            language: 'vi' or 'en'. Defaults to config.language.
            mode: 'text_only' or 'vision'.
            num_images: Number of images in the review.
            output_dir: If provided, save reports to this directory.

        Returns:
            Structured explanation dict (matches AGENT_OUTPUT_SCHEMA).
        """
        lang = language or self.config.language

        # 1. Load evidence
        evidence = self.evidence_loader.load(sample_id, xai_dir)

        # 2. Build compressed evidence
        blocks = self.evidence_builder.build(evidence)

        # 3. Build prompt
        prompt = self.prompt_builder.build(
            sample_id=sample_id,
            review_text=review_text,
            predictions=predictions,
            evidence_blocks=blocks,
            ground_truth=ground_truth,
            case_type=case_type,
            language=lang,
            num_images=num_images,
        )

        # 4. Call OpenAI
        model = (self.config.report_model if mode == 'vision'
                 else self.config.batch_model)
        result = self.client.generate_json(
            messages=prompt['messages'],
            model=model,
        )

        # 5. Ensure required fields
        result.setdefault('sample_id', sample_id)
        result.setdefault('language', lang)
        result.setdefault('timestamp',
                          datetime.datetime.now().isoformat())

        # 5b. Override evidence_completeness with ground truth
        #     (LLM may guess wrong about which artifacts exist)
        ec = {
            'gradcam': 'gradcam' in evidence,
            'attention': 'attention' in evidence,
            'cross_attention': 'cross_attention' in evidence,
            'shap': 'shap' in evidence,
            'lime': 'lime' in evidence,
        }
        total = sum(ec.values())
        ec['total'] = f'{total}/5'
        result['evidence_completeness'] = ec

        # 6. Validate
        warnings = self.validator.validate(result, evidence)
        if warnings:
            result['validation_warnings'] = warnings

        # 7. Save reports if output_dir given
        if output_dir:
            self.reporter.save_sample_report(
                result, output_dir, review_text,
                predictions, ground_truth)

        return result

    def explain_case_study(
        self,
        case_id: str,
        case_dir: str,
        xai_dir: str,
        language: Optional[str] = None,
        mode: str = 'text_only',
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate explanation from a Phase 6 case study metadata.json.

        Args:
            case_id: e.g. 'case_correct_001'.
            case_dir: Path to the case study directory.
            xai_dir: Base XAI directory.
            language: 'vi' or 'en'.
            mode: 'text_only' or 'vision'.
            output_dir: If provided, save reports here.

        Returns:
            Structured explanation dict.
        """
        meta_path = os.path.join(case_dir, 'metadata.json')
        if not os.path.isfile(meta_path):
            return {
                'sample_id': case_id,
                'error': f'metadata.json not found at {meta_path}',
            }

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        sample_id = meta.get('sample_id', case_id)
        review_text = meta.get('review_text', '')
        case_type = meta.get('case_type', None)

        # Convert display-name predictions to target-name predictions
        preds_display = meta.get('predictions', {})
        _display_to_target = {
            'Food Score': 'food_score',
            'Price Score': 'price_score',
            'Atmosphere Score': 'atmosphere_score',
            'Service Score': 'service_score',
            'Overall Satisfaction': 'overall_satisfaction',
        }
        predictions = {}
        for display_name, target_name in _display_to_target.items():
            predictions[target_name] = preds_display.get(display_name, 0)

        gt_display = meta.get('ground_truth', {})
        ground_truth = {}
        for display_name, target_name in _display_to_target.items():
            ground_truth[target_name] = gt_display.get(display_name, 0)

        return self.explain_sample(
            sample_id=sample_id,
            review_text=review_text,
            predictions=predictions,
            xai_dir=xai_dir,
            ground_truth=ground_truth,
            case_type=case_type,
            language=language,
            mode=mode,
            output_dir=output_dir,
        )

    def explain_batch(
        self,
        samples: List[Dict[str, Any]],
        xai_dir: str,
        language: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-process multiple samples.

        Each item in samples should have: sample_id, review_text,
        predictions. Optional: ground_truth, case_type.

        Args:
            samples: List of sample dicts.
            xai_dir: Base XAI directory.
            language: Output language.
            output_dir: If provided, save per-sample + batch reports.

        Returns:
            List of result dicts (one per sample).
        """
        results: List[Dict[str, Any]] = []

        for sample in samples:
            t0 = time.time()
            try:
                result = self.explain_sample(
                    sample_id=sample['sample_id'],
                    review_text=sample.get('review_text', ''),
                    predictions=sample.get('predictions', {}),
                    xai_dir=xai_dir,
                    ground_truth=sample.get('ground_truth'),
                    case_type=sample.get('case_type'),
                    language=language,
                    mode='text_only',
                    output_dir=output_dir,
                )
                result['elapsed_s'] = round(time.time() - t0, 1)
                results.append(result)
                print(f'[Agent] {sample["sample_id"]}: OK '
                      f'({result["elapsed_s"]}s)')
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                err_result = {
                    'sample_id': sample.get('sample_id', '?'),
                    'error': str(e),
                    'elapsed_s': elapsed,
                }
                results.append(err_result)
                print(f'[Agent] {sample.get("sample_id", "?")}: '
                      f'FAILED ({e})')

        # Save batch summary
        if output_dir:
            self.reporter.save_batch_summary(results, output_dir)

        return results
