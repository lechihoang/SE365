"""
Generate Markdown and JSON report files from AI Agent output.

Reports follow a fixed section order for thesis-readiness:
Review → Predictions → Summary → Score Explanations → Evidence →
SHAP → Cross-Method Agreement → Limitations → Recommendations.
"""

import os
import json
import csv
import datetime
from typing import Dict, Any, List, Optional

from agent.output_schema import level_display

_DISPLAY_NAMES = {
    'food': 'Food Score', 'price': 'Price Score',
    'atmos': 'Atmosphere Score', 'service': 'Service Score',
    'overall': 'Overall Satisfaction',
}
_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']


class ReportGenerator:
    """Generate structured report files from agent output."""

    def save_sample_report(
        self,
        output: Dict[str, Any],
        output_dir: str,
        review_text: str = '',
        predictions: Optional[Dict[str, float]] = None,
        ground_truth: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """Save JSON and Markdown reports for a single sample."""
        os.makedirs(output_dir, exist_ok=True)
        sid = output.get('sample_id', 'unknown')
        lang = output.get('language', 'vi')
        paths: Dict[str, str] = {}

        # JSON
        json_path = os.path.join(output_dir, f'{sid}_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        paths['json'] = json_path

        # Markdown
        md = self._build_markdown(
            output, review_text, predictions, ground_truth, lang)
        md_path = os.path.join(output_dir, f'{sid}_report_{lang}.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        paths['markdown'] = md_path

        return paths

    def save_batch_summary(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        api_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Save batch summary CSV, JSON, and optional API log."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # JSON
        json_path = os.path.join(output_dir, 'batch_summary.json')
        summary_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_samples': len(results),
            'succeeded': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'results': results,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        paths['json'] = json_path

        # CSV
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        fieldnames = [
            'sample_id', 'confidence', 'summary',
            'food', 'price', 'atmos', 'service', 'overall', 'status',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                scores = r.get('scores', {})
                row = {
                    'sample_id': r.get('sample_id', '?'),
                    'confidence': r.get('confidence', '?'),
                    'summary': (r.get('summary', '')[:100]
                                if isinstance(r.get('summary'), str)
                                else ''),
                    'status': 'error' if 'error' in r else 'ok',
                }
                for fn in _FACTOR_NAMES:
                    entry = scores.get(fn, {})
                    row[fn] = entry.get('score', '') if entry else ''
                writer.writerow(row)
        paths['csv'] = csv_path

        if api_log:
            log_path = os.path.join(output_dir, 'batch_log.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(api_log, f, indent=2, ensure_ascii=False)
            paths['log'] = log_path

        return paths

    # ── Markdown builder ─────────────────────────────────────────────

    def _build_markdown(
        self,
        output: Dict[str, Any],
        review_text: str,
        predictions: Optional[Dict[str, float]],
        ground_truth: Optional[Dict[str, float]],
        lang: str,
    ) -> str:
        sid = output.get('sample_id', 'unknown')
        lines = [
            f'# AI Agent Report — {sid}', '',
            f'**Language:** {lang}  ',
            f'**Confidence:** {output.get("confidence", "?")}  ',
            f'**Generated:** {output.get("timestamp", "")}', '',
        ]

        # 1. Review
        lines.extend(['## Review Text', ''])
        if review_text:
            lines.append(f'> {review_text[:500]}')
        else:
            lines.append('*Not available.*')
        lines.append('')

        # 2. Predictions
        lines.extend(['## Predictions', ''])
        if predictions:
            lines.extend([
                '| Target | Predicted |', '|--------|-----------|'])
            for key in ['food_score', 'price_score', 'atmosphere_score',
                        'service_score', 'overall_satisfaction']:
                lines.append(f'| {key} | {predictions.get(key, 0):.2f} |')
        else:
            lines.append('*Not available.*')
        lines.append('')

        # 3. Ground truth + error
        lines.extend(['## Ground Truth', ''])
        if ground_truth:
            lines.extend([
                '| Target | True | Predicted | Error |',
                '|--------|------|-----------|-------|',
            ])
            for key in ['food_score', 'price_score', 'atmosphere_score',
                        'service_score', 'overall_satisfaction']:
                gt = ground_truth.get(key, 0)
                pred = predictions.get(key, 0) if predictions else 0
                lines.append(
                    f'| {key} | {gt:.2f} | {pred:.2f} '
                    f'| {abs(pred - gt):.3f} |')
        else:
            lines.append('*Not available.*')
        lines.append('')

        # 4. Summary
        lines.extend(['## Summary', ''])
        lines.append(output.get('summary', '*Not available.*'))
        lines.append('')

        # 5. Score explanations (all 5)
        lines.extend(['## Score Explanations', ''])
        scores = output.get('scores', {})
        for factor in _FACTOR_NAMES:
            dn = _DISPLAY_NAMES.get(factor, factor)
            entry = scores.get(factor, {})
            score_val = entry.get('score', '?')
            level_str = entry.get('level', '?')
            level_vi = level_display(level_str, lang) if isinstance(
                level_str, str) else '?'
            lines.append(f'### {dn}: {score_val} ({level_vi})')
            lines.append('')
            lines.append(entry.get('explanation',
                                   '*No explanation generated.*'))
            lines.append('')

        # 6. Evidence completeness
        lines.extend(['## Evidence Completeness', ''])
        ec = output.get('evidence_completeness', {})
        if ec:
            for method in ['gradcam', 'attention', 'cross_attention',
                           'shap', 'lime']:
                avail = ec.get(method, False)
                mark = 'Available' if avail else 'Missing'
                lines.append(f'- **{method}**: {mark}')
            lines.append(f'- **Total**: {ec.get("total", "?")}')
        else:
            lines.append('*Not available.*')
        lines.append('')

        # 7. SHAP interpretation
        lines.extend(['## SHAP Modality Contribution', ''])
        mod = output.get('modality_contribution', {})
        if mod:
            lines.append(
                f'- Overall text-origin: '
                f'{mod.get("text_origin_pct", "?")}%')
            lines.append(
                f'- Overall image-origin: '
                f'{mod.get("image_origin_pct", "?")}%')
            per_tgt = mod.get('per_target', {})
            if per_tgt:
                lines.extend([
                    '', '| Target | Text-origin | Image-origin |',
                    '|--------|-------------|--------------|',
                ])
                for factor in _FACTOR_NAMES:
                    t = per_tgt.get(factor, {})
                    lines.append(
                        f'| {_DISPLAY_NAMES.get(factor, factor)} '
                        f'| {t.get("text_origin_pct", "?")}% '
                        f'| {t.get("image_origin_pct", "?")}% |')
            lines.append('')
            interp = mod.get('interpretation', '')
            if interp:
                lines.append(interp)
        else:
            lines.append('*Not available.*')
        lines.append('')

        # 8. Cross-modal insights
        lines.extend(['## Cross-Attention Insights', ''])
        cmi = output.get('cross_modal_insights', '')
        lines.append(cmi if cmi else '*Not available.*')
        lines.append('')

        # 9. Cross-method agreement
        lines.extend(['## Cross-Method Agreement', ''])
        ma = output.get('method_agreement', '')
        lines.append(ma if ma else '*Not available.*')
        lines.append('')

        # 10. Limitations
        lines.extend(['## Limitations', ''])
        lims = output.get('limitations', [])
        if lims:
            for lim in lims:
                lines.append(f'- {lim}')
        else:
            lines.append('*No limitations listed.*')
        lines.append('')

        # 11. Recommendations
        lines.extend(['## Recommendations', ''])
        recs = output.get('recommendations', [])
        if recs:
            for rec in recs:
                lines.append(f'- {rec}')
        else:
            lines.append('*No recommendations.*')
        lines.append('')

        # 12. Confidence reasoning
        lines.extend(['## Confidence', ''])
        lines.append(
            f'**Level:** {output.get("confidence", "?")}')
        cr = output.get('confidence_reasoning', '')
        if cr:
            lines.append(f'\n{cr}')
        lines.append('')

        # 13. Validation warnings
        warns = output.get('validation_warnings', [])
        if warns:
            lines.extend(['## Validation Warnings', ''])
            for w in warns:
                lines.append(f'- {w}')
            lines.append('')

        return '\n'.join(lines)
