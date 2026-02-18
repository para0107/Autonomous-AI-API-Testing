"""
LLM client (Groq / any OpenAI-compatible cloud API)
"""

import logging
import asyncio
import aiohttp
import json
import re
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential

from config import LlamaConfig

logger = logging.getLogger(__name__)

llama_config = LlamaConfig()


class LlamaClient:
    """Client for interacting with an OpenAI-compatible LLM API (e.g. Groq)"""

    def __init__(self):
        self.base_url = llama_config.base_url
        self.model = llama_config.model_name
        self.api_key = llama_config.api_key
        self.default_params = {
            'temperature': llama_config.temperature,
            'max_tokens': llama_config.max_tokens,
            'top_p': llama_config.top_p,
        }
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_session(self) -> aiohttp.ClientSession:
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion.

        Args:
            prompt: Input prompt string.
            **kwargs: Override default parameters.

        Returns:
            Generated text string.
        """
        params = {**self.default_params, **kwargs}
        # Remove params not supported by cloud APIs
        params.pop('top_k', None)
        params.pop('frequency_penalty', None)
        params.pop('presence_penalty', None)
        params.pop('stop', None)

        payload = {
            'model': self.model,
            'messages': [{"role": "user", "content": prompt}],
            **params,
        }

        try:
            logger.debug(f"Sending request (prompt length: {len(prompt)} chars)")
            async with self._get_session().post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=llama_config.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
                return ''

        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {llama_config.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during generate: {e}")
            raise

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion with conversation history.

        Args:
            messages: List of {'role': ..., 'content': ...} dicts.
            **kwargs: Override default parameters.

        Returns:
            Generated response string.
        """
        params = {**self.default_params, **kwargs}
        params.pop('top_k', None)
        params.pop('frequency_penalty', None)
        params.pop('presence_penalty', None)
        params.pop('stop', None)

        payload = {
            'model': self.model,
            'messages': messages,
            **params,
        }

        try:
            async with self._get_session().post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=llama_config.timeout)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
                return ''
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

    async def generate_json(self, prompt: str, schema: Dict[str, Any] = None, **kwargs) -> Any:
        """
        Generate and parse a JSON response. Retries up to 3 times on parse failure.

        Args:
            prompt: The prompt requesting JSON output.
            schema: Optional schema hint appended to the prompt.
            **kwargs: Override generation parameters.

        Returns:
            Parsed Python object (dict or list).
        """
        json_prompt = f"""{prompt}

Respond with ONLY valid JSON — no markdown, no explanation, no extra text.
Start your response with {{ or [ and end with }} or ]."""

        if schema:
            json_prompt += f"\n\nExpected structure:\n{json.dumps(schema, indent=2)}"

        last_error = None
        for attempt in range(3):
            try:
                response = await self.generate(json_prompt, **kwargs)
                return self._parse_json(response)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(f"JSON parse attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    # Ask the model to fix its own output
                    json_prompt = f"""Your previous response was not valid JSON. Error: {e}

Original request:
{prompt}

Try again. Respond with ONLY valid JSON."""

        logger.error(f"All JSON parse attempts failed. Last error: {last_error}")
        raise ValueError(f"Could not obtain valid JSON after 3 attempts: {last_error}")

    def _parse_json(self, response: str) -> Any:
        """
        Extract and parse JSON from a model response.
        Handles markdown fences and leading/trailing text.

        Args:
            response: Raw model response string.

        Returns:
            Parsed Python object.

        Raises:
            ValueError: If no valid JSON is found.
        """
        # Strip markdown fences if present
        fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if fenced:
            candidate = fenced.group(1).strip()
            return json.loads(candidate)

        # Find the first { or [ and attempt to parse from there
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = response.find(start_char)
            if start == -1:
                continue
            # Walk backwards from the last closing brace to find the shortest valid JSON
            end = response.rfind(end_char)
            if end == -1:
                continue
            candidate = response[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No valid JSON found in response: {response[:200]}")

    async def check_connection(self) -> bool:
        """Check if the API server is reachable."""
        try:
            async with self._get_session().get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    async def stream_generate(self, prompt: str, callback=None, **kwargs):
        """
        Stream generation for real-time output.

        Args:
            prompt: Input prompt.
            callback: Optional async callback for each token chunk.
            **kwargs: Override parameters.
        """
        params = {**self.default_params, **kwargs}
        params.pop('top_k', None)
        params.pop('frequency_penalty', None)
        params.pop('presence_penalty', None)
        params['stream'] = True

        payload = {
            'model': self.model,
            'messages': [{"role": "user", "content": prompt}],
            **params,
        }

        async with self._get_session().post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == 'data: [DONE]':
                    continue
                if line.startswith('data: '):
                    line = line[6:]
                try:
                    data = json.loads(line)
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    token = delta.get('content', '')
                    if token:
                        if callback:
                            await callback(token)
                        else:
                            yield token
                except json.JSONDecodeError:
                    continue

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for text.

        Args:
            text: Input text.

        Returns:
            Embedding vector, or empty list if unsupported.
        """
        payload = {
            'model': self.model,
            'input': text
        }

        try:
            async with self._get_session().post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self._headers(),
            ) as response:
                result = await response.json()
                if 'data' in result and result['data']:
                    return result['data'][0].get('embedding', [])
                return []
        except Exception:
            logger.warning("Embeddings endpoint not supported or failed.")
            return []

    def get_config_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type."""
        return llama_config.get_agent_config(agent_type)