"""
Generate Markdown and JSON report files from AI Agent output.
"""

import os
import json
import csv
import datetime
from typing import Dict, Any, List, Optional

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
        """Save JSON and Markdown reports for a single sample.

        Returns dict mapping format -> saved file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        sid = output.get('sample_id', 'unknown')
        lang = output.get('language', 'vi')
        paths: Dict[str, str] = {}

        # JSON report
        json_path = os.path.join(output_dir, f'{sid}_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        paths['json'] = json_path

        # Markdown report
        md = self._build_markdown(
            output, review_text, predictions, ground_truth)
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
        """Save batch summary CSV, JSON, and optional API log.

        Returns dict mapping format -> saved file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # batch_summary.json
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

        # batch_summary.csv
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        fieldnames = [
            'sample_id', 'confidence', 'summary',
            'food', 'price', 'atmos', 'service', 'overall',
            'status',
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
                                if isinstance(r.get('summary'), str) else ''),
                    'status': 'error' if 'error' in r else 'ok',
                }
                for f_name in _FACTOR_NAMES:
                    entry = scores.get(f_name, {})
                    row[f_name] = entry.get('score', '') if entry else ''
                writer.writerow(row)
        paths['csv'] = csv_path

        # batch_log.json (API usage)
        if api_log:
            log_path = os.path.join(output_dir, 'batch_log.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(api_log, f, indent=2, ensure_ascii=False)
            paths['log'] = log_path

        return paths

    # ── Private ──────────────────────────────────────────────────────

    def _build_markdown(
        self,
        output: Dict[str, Any],
        review_text: str,
        predictions: Optional[Dict[str, float]],
        ground_truth: Optional[Dict[str, float]],
    ) -> str:
        sid = output.get('sample_id', 'unknown')
        lang = output.get('language', 'vi')
        lines = [
            f'# AI Agent Report — {sid}', '',
            f'**Language:** {lang}  ',
            f'**Confidence:** {output.get("confidence", "?")}  ',
            f'**Generated:** {output.get("timestamp", "")}', '',
        ]

        # Review text
        if review_text:
            lines.extend([
                '## Review Text', '',
                f'> {review_text[:500]}', '',
            ])

        # Predictions table
        if predictions:
            lines.extend([
                '## Predictions', '',
                '| Target | Predicted |',
                '|--------|-----------|',
            ])
            for key in ['food_score', 'price_score', 'atmosphere_score',
                        'service_score', 'overall_satisfaction']:
                val = predictions.get(key, 0)
                lines.append(f'| {key} | {val:.2f} |')
            lines.append('')

        # Summary
        summary = output.get('summary', '')
        if summary:
            lines.extend(['## Summary', '', summary, ''])

        # Per-score explanations
        scores = output.get('scores', {})
        if scores:
            lines.extend(['## Score Explanations', ''])
            for factor in _FACTOR_NAMES:
                entry = scores.get(factor, {})
                if not entry:
                    continue
                dn = _DISPLAY_NAMES.get(factor, factor)
                lines.append(
                    f'### {dn}: {entry.get("score", "?")} '
                    f'({entry.get("level", "?")})')
                lines.append('')
                lines.append(entry.get('explanation', ''))
                lines.append('')

        # Modality contribution
        mod = output.get('modality_contribution', {})
        if mod:
            lines.extend([
                '## Modality Contribution', '',
                f'- Text-origin: {mod.get("text_origin_pct", "?")}%',
                f'- Image-origin: {mod.get("image_origin_pct", "?")}%',
                f'- {mod.get("interpretation", "")}', '',
            ])

        # Cross-modal insights
        cmi = output.get('cross_modal_insights', '')
        if cmi:
            lines.extend(['## Cross-Modal Insights', '', cmi, ''])

        # Limitations
        lims = output.get('limitations', [])
        if lims:
            lines.extend(['## Limitations', ''])
            for lim in lims:
                lines.append(f'- {lim}')
            lines.append('')

        # Recommendations
        recs = output.get('recommendations', [])
        if recs:
            lines.extend(['## Recommendations', ''])
            for rec in recs:
                lines.append(f'- {rec}')
            lines.append('')

        # Validation warnings
        warns = output.get('validation_warnings', [])
        if warns:
            lines.extend(['## Validation Warnings', ''])
            for w in warns:
                lines.append(f'- {w}')
            lines.append('')

        return '\n'.join(lines)
