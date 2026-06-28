"""
OpenAI API client wrapper with retry, timeout, and structured JSON output.

Never logs or prints the API key.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List

from agent.config import AgentConfig

logger = logging.getLogger('agent')


class OpenAIClient:
    """Wrapper around the OpenAI chat completions API.

    Handles retries with exponential backoff, timeout, and JSON parsing.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy-init the OpenAI client on first use."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    'openai package not installed. '
                    'Install with: pip install openai>=1.0')
            api_key = self.config.require_api_key()
            self._client = OpenAI(api_key=api_key)
        return self._client

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call the OpenAI API and return parsed JSON.

        Args:
            messages: List of chat messages (system + user).
            model: Model override. Defaults to config.report_model.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            Parsed JSON dict from the model response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        import openai

        client = self._get_client()
        model = model or self.config.report_model
        temperature = temperature if temperature is not None \
            else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={'type': 'json_object'},
                    timeout=self.config.timeout,
                )
                content = response.choices[0].message.content

                # Log usage
                if response.usage:
                    logger.info(
                        'OpenAI usage: model=%s input=%d output=%d total=%d',
                        model,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        response.usage.total_tokens,
                    )

                return json.loads(content)

            except openai.RateLimitError as e:
                wait = (2 ** attempt) * 5
                logger.warning('Rate limited (attempt %d). Waiting %ds...',
                               attempt + 1, wait)
                time.sleep(wait)
                last_error = e

            except openai.APITimeoutError as e:
                logger.warning('Timeout on attempt %d', attempt + 1)
                last_error = e

            except json.JSONDecodeError as e:
                logger.error('JSON parse error: %s', e)
                return {
                    'error': 'Invalid JSON response from OpenAI',
                    'raw_content': content[:500] if content else '',
                }

            except openai.APIError as e:
                logger.warning('API error on attempt %d: %s',
                               attempt + 1, e)
                last_error = e
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f'OpenAI API failed after {self.config.max_retries} attempts. '
            f'Last error: {last_error}')
