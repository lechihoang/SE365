"""
Configuration for the AI Agent module.

Handles OpenAI API credentials, model selection, generation parameters,
and evidence extraction settings. API keys are read from environment
variables or .env files — never hardcoded.
"""

import os
from typing import Optional


class AgentConfig:
    """Configuration for the ExplanationAgent.

    Args:
        batch_model: Model for batch/cheap processing.
        report_model: Model for high-quality thesis reports.
        vision_model: Model for vision-enabled requests.
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Max output tokens per request.
        max_tokens_batch: Max output tokens in batch mode.
        timeout: API request timeout in seconds.
        max_retries: Max retry attempts on transient failures.
        language: Default output language ('vi' or 'en').
        top_k_tokens: Number of top attention tokens to include.
        top_k_pairs: Number of top cross-attention pairs to include.
        top_k_lime: Number of top LIME words per factor.
    """

    def __init__(
        self,
        batch_model: str = 'gpt-4o',
        report_model: str = 'gpt-4o',
        vision_model: str = 'gpt-4o',
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_tokens_batch: int = 800,
        timeout: int = 60,
        max_retries: int = 3,
        language: str = 'vi',
        top_k_tokens: int = 10,
        top_k_pairs: int = 10,
        top_k_lime: int = 5,
    ):
        self.api_key = self._resolve_api_key()
        self.batch_model = batch_model
        self.report_model = report_model
        self.vision_model = vision_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_batch = max_tokens_batch
        self.timeout = timeout
        self.max_retries = max_retries
        self.language = language
        self.top_k_tokens = top_k_tokens
        self.top_k_pairs = top_k_pairs
        self.top_k_lime = top_k_lime

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        """Resolve OpenAI API key from environment, .env, or Colab secrets."""
        key = os.environ.get('OPENAI_API_KEY')
        if key:
            return key

        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get('OPENAI_API_KEY')
            if key:
                return key
        except ImportError:
            pass

        try:
            from google.colab import userdata
            key = userdata.get('OPENAI_API_KEY')
            if key:
                return key
        except (ImportError, Exception):
            pass

        return None

    def require_api_key(self) -> str:
        """Return the API key or raise if not configured."""
        if not self.api_key:
            raise ValueError(
                'OPENAI_API_KEY not found. Set it via:\n'
                '  export OPENAI_API_KEY=sk-...\n'
                '  or add to .env file\n'
                '  or set in Colab secrets'
            )
        return self.api_key
