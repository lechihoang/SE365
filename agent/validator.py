"""
Validate AI Agent output for schema compliance and evidence grounding.

Returns warnings instead of raising exceptions so partial outputs
can still be used.
"""

from typing import Dict, Any, List, Optional

from agent.output_schema import AGENT_OUTPUT_SCHEMA, score_to_level

_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']


class OutputValidator:
    """Validates agent output against schema and evidence grounding."""

    def validate(
        self,
        output: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Run all validation checks.

        Returns list of warning strings. Empty list = all checks passed.
        """
        warnings: List[str] = []

        if 'error' in output:
            warnings.append(f'Output contains error: {output["error"]}')
            return warnings

        warnings.extend(self._check_schema(output))
        warnings.extend(self._check_all_targets(output))
        warnings.extend(self._check_score_levels(output))
        warnings.extend(self._check_required_fields(output))
        warnings.extend(self._check_evidence_completeness(output))
        warnings.extend(self._check_limitations(output))
        if evidence:
            warnings.extend(self._check_shap_grounding(output, evidence))

        return warnings

    def _check_schema(self, output: Dict[str, Any]) -> List[str]:
        try:
            import jsonschema
            jsonschema.validate(output, AGENT_OUTPUT_SCHEMA)
            return []
        except ImportError:
            return ['jsonschema not installed — schema check skipped']
        except jsonschema.ValidationError as e:
            return [f'Schema violation: {e.message}']

    def _check_all_targets(self, output: Dict[str, Any]) -> List[str]:
        """Verify all 5 targets are explained."""
        warnings = []
        scores = output.get('scores', {})
        for factor in _FACTOR_NAMES:
            if factor not in scores:
                warnings.append(
                    f'Missing score explanation for "{factor}" — '
                    f'all 5 targets must be explained')
            elif not scores[factor].get('explanation'):
                warnings.append(
                    f'Empty explanation for "{factor}"')
        return warnings

    def _check_score_levels(self, output: Dict[str, Any]) -> List[str]:
        warnings = []
        scores = output.get('scores', {})
        for factor in _FACTOR_NAMES:
            entry = scores.get(factor)
            if not entry or 'score' not in entry or 'level' not in entry:
                continue
            expected = score_to_level(entry['score'])
            actual = entry['level']
            if actual != expected:
                warnings.append(
                    f'{factor}: level "{actual}" does not match '
                    f'score {entry["score"]:.1f} (expected "{expected}")')
        return warnings

    def _check_required_fields(self, output: Dict[str, Any]) -> List[str]:
        warnings = []
        for field in ['summary', 'confidence']:
            if not output.get(field):
                warnings.append(f'Missing or empty: {field}')
        if not output.get('method_agreement'):
            warnings.append('Missing "method_agreement" section')
        if not output.get('confidence_reasoning'):
            warnings.append('Missing "confidence_reasoning"')
        return warnings

    def _check_evidence_completeness(self, output: Dict[str, Any]) -> List[str]:
        ec = output.get('evidence_completeness')
        if not ec:
            return ['Missing "evidence_completeness" section']
        return []

    def _check_limitations(self, output: Dict[str, Any]) -> List[str]:
        lims = output.get('limitations', [])
        if not lims:
            return ['No limitations listed — always required']
        if len(lims) < 3:
            return [f'Only {len(lims)} limitation(s) — recommend at least 3']
        return []

    def _check_shap_grounding(
        self,
        output: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[str]:
        warnings = []
        shap_evidence = evidence.get('shap')
        if not shap_evidence:
            return warnings

        mod = output.get('modality_contribution', {})
        overall_shap = shap_evidence.get('overall', {})
        if mod and overall_shap:
            claimed = mod.get('text_origin_pct', 0)
            actual = overall_shap.get('text_pct', 0)
            if abs(claimed - actual) > 5.0:
                warnings.append(
                    f'Overall SHAP mismatch: claimed text-origin '
                    f'{claimed:.1f}% vs evidence {actual:.1f}%')

        return warnings
