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
        warnings.extend(self._check_score_levels(output))
        if evidence:
            warnings.extend(self._check_shap_grounding(output, evidence))
        warnings.extend(self._check_required_fields(output))

        return warnings

    def _check_schema(self, output: Dict[str, Any]) -> List[str]:
        """Validate against JSON schema using jsonschema if available."""
        try:
            import jsonschema
            jsonschema.validate(output, AGENT_OUTPUT_SCHEMA)
            return []
        except ImportError:
            return ['jsonschema package not installed — schema check skipped']
        except jsonschema.ValidationError as e:
            return [f'Schema violation: {e.message}']

    def _check_score_levels(self, output: Dict[str, Any]) -> List[str]:
        """Check that score levels match the score values."""
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

    def _check_shap_grounding(
        self,
        output: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[str]:
        """Check that SHAP percentages in output match input evidence."""
        warnings = []
        shap_evidence = evidence.get('shap')
        if not shap_evidence:
            return warnings

        mod = output.get('modality_contribution', {})
        overall_shap = shap_evidence.get('overall', {})
        if mod and overall_shap:
            claimed_text = mod.get('text_origin_pct', 0)
            actual_text = overall_shap.get('text_pct', 0)
            if abs(claimed_text - actual_text) > 5.0:
                warnings.append(
                    f'Overall SHAP text_origin_pct mismatch: '
                    f'claimed {claimed_text:.1f}% vs actual {actual_text:.1f}%')

        return warnings

    def _check_required_fields(self, output: Dict[str, Any]) -> List[str]:
        """Check that key fields are present and non-empty."""
        warnings = []
        for field in ['summary', 'confidence']:
            if not output.get(field):
                warnings.append(f'Missing or empty required field: {field}')
        if not output.get('limitations'):
            warnings.append('No limitations listed — agent should '
                            'always include XAI limitations')
        return warnings
