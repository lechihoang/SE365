"""
JSON schema for AI Agent structured output and validation helpers.
"""

SCORE_EXPLANATION_SCHEMA = {
    'type': 'object',
    'required': ['score', 'level', 'explanation'],
    'properties': {
        'score': {'type': 'number'},
        'level': {
            'type': 'string',
            'enum': ['low', 'below_average', 'average', 'good', 'excellent'],
        },
        'explanation': {'type': 'string'},
        'evidence': {
            'type': 'object',
            'properties': {
                'gradcam': {'type': 'string'},
                'attention': {'type': 'array', 'items': {'type': 'string'}},
                'cross_attention': {'type': 'string'},
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
                'interpretation': {'type': 'string'},
            },
        },
        'cross_modal_insights': {'type': 'string'},
        'method_agreement': {'type': 'string'},
        'limitations': {'type': 'array', 'items': {'type': 'string'}},
        'recommendations': {'type': 'array', 'items': {'type': 'string'}},
        'confidence': {
            'type': 'string',
            'enum': ['low', 'medium', 'high'],
        },
        'timestamp': {'type': 'string'},
        'validation_warnings': {
            'type': 'array', 'items': {'type': 'string'},
        },
    },
}


def score_to_level(score: float) -> str:
    """Map a 1-10 score to a quality level string."""
    if score <= 3:
        return 'low'
    elif score <= 5:
        return 'below_average'
    elif score <= 7:
        return 'average'
    elif score <= 9:
        return 'good'
    return 'excellent'
