"""
JSON schema for AI Agent structured output and validation helpers.
"""

# Score level definitions — machine-readable enum + display labels
SCORE_LEVELS = {
    'very_poor':     {'range': (0, 2),  'vi': 'Rất kém',      'en': 'Very Poor'},
    'poor':          {'range': (2, 4),  'vi': 'Kém',           'en': 'Poor'},
    'average':       {'range': (4, 6),  'vi': 'Trung bình',    'en': 'Average'},
    'good':          {'range': (6, 8),  'vi': 'Khá',           'en': 'Good'},
    'excellent':     {'range': (8, 10), 'vi': 'Xuất sắc',      'en': 'Excellent'},
}

LEVEL_ENUM = list(SCORE_LEVELS.keys())


def score_to_level(score: float) -> str:
    """Map a 1-10 score to a machine-readable level string."""
    if score < 2:
        return 'very_poor'
    elif score < 4:
        return 'poor'
    elif score < 6:
        return 'average'
    elif score < 8:
        return 'good'
    return 'excellent'


def level_display(level: str, lang: str = 'vi') -> str:
    """Get human-readable display text for a level."""
    info = SCORE_LEVELS.get(level)
    if not info:
        return level
    return info.get(lang, info.get('vi', level))


# Allow string-or-null for optional text fields to prevent
# "None is not of type 'string'" schema violations
_STRING_OR_EMPTY = {'type': 'string', 'default': ''}

SCORE_EXPLANATION_SCHEMA = {
    'type': 'object',
    'required': ['score', 'level', 'explanation'],
    'properties': {
        'score': {'type': 'number'},
        'level': {'type': 'string', 'enum': LEVEL_ENUM},
        'explanation': {'type': 'string'},
        'evidence': {
            'type': 'object',
            'properties': {
                'gradcam': _STRING_OR_EMPTY,
                'attention': {'type': 'array', 'items': {'type': 'string'}},
                'cross_attention': _STRING_OR_EMPTY,
                'shap': {
                    'type': 'object',
                    'properties': {
                        'text_origin_pct': {'type': 'number'},
                        'image_origin_pct': {'type': 'number'},
                    },
                },
                'lime_text': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'word': {'type': 'string'},
                            'weight': {'type': 'number'},
                        },
                    },
                },
            },
        },
    },
}

AGENT_OUTPUT_SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'type': 'object',
    'required': ['sample_id', 'language', 'summary', 'scores', 'confidence'],
    'properties': {
        'sample_id': {'type': 'string'},
        'language': {'type': 'string', 'enum': ['vi', 'en']},
        'summary': {'type': 'string'},
        'scores': {
            'type': 'object',
            'required': ['food', 'price', 'atmos', 'service', 'overall'],
            'properties': {
                'food': SCORE_EXPLANATION_SCHEMA,
                'price': SCORE_EXPLANATION_SCHEMA,
                'atmos': SCORE_EXPLANATION_SCHEMA,
                'service': SCORE_EXPLANATION_SCHEMA,
                'overall': SCORE_EXPLANATION_SCHEMA,
            },
        },
        'modality_contribution': {
            'type': 'object',
            'properties': {
                'text_origin_pct': {'type': 'number'},
                'image_origin_pct': {'type': 'number'},
                'per_target': {'type': 'object'},
                'interpretation': {'type': 'string'},
            },
        },
        'evidence_completeness': {
            'type': 'object',
            'properties': {
                'gradcam': {'type': 'boolean'},
                'attention': {'type': 'boolean'},
                'cross_attention': {'type': 'boolean'},
                'shap': {'type': 'boolean'},
                'lime': {'type': 'boolean'},
                'total': {'type': 'string'},
            },
        },
        'visual_artifacts': {'type': 'object'},
        'reasoning_graph': {'type': 'object'},
        'agreement_matrix': {
            'type': 'array',
            'items': {'type': 'object'},
        },
        'cross_modal_insights': {'type': 'string'},
        'method_agreement': {'type': 'string'},
        'limitations': {'type': 'array', 'items': {'type': 'string'}},
        'recommendations': {'type': 'array', 'items': {'type': 'string'}},
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'confidence_reasoning': {'type': 'string'},
        'customer_view': {
            'type': 'object',
            'properties': {
                'summary': {'type': 'string'},
                'highlights': {'type': 'array', 'items': {'type': 'string'}},
                'recommendations': {
                    'type': 'array', 'items': {'type': 'string'}},
            },
        },
        'timestamp': {'type': 'string'},
        'validation_warnings': {
            'type': 'array', 'items': {'type': 'string'}},
    },
}
