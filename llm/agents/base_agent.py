"""
Base agent class for LLM agents
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LLM agents"""

    def __init__(self, llama_client, agent_type: str = None):
        self.client = llama_client
        self.agent_type = agent_type or self.__class__.__name__
        self.config = self._get_config()

    def _get_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        raw_config = self.client.get_config_for_agent(self.agent_type)

        # LOG AT INFO LEVEL to see what we're getting
        logger.info(f"=== Config for agent '{self.agent_type}' ===")
        logger.info(f"Raw config type: {type(raw_config)}")
        for key, value in raw_config.items():
            logger.info(f"  {key}: type={type(value).__name__}, value={value if not isinstance(value, dict) else '<DICT>'}")

        # Validate and flatten config
        clean_config = {}
        for key, value in raw_config.items():
            if isinstance(value, (int, float, str, bool)):
                clean_config[key] = value
            elif isinstance(value, list):
                clean_config[key] = value
            elif isinstance(value, dict):
                logger.warning(f"Config key '{key}' contains nested dict, skipping")
            else:
                logger.warning(f"Config key '{key}' has unexpected type {type(value)}, skipping")

        logger.info(f"Clean config: {clean_config}")
        logger.info("=" * 50)
        return clean_config

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Any:
        """
        Execute agent task

        Args:
            input_data: Input data for the agent

        Returns:
            Agent output
        """
        pass

    async def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate with retry logic"""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.generate(prompt,  **self.config)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        return ""

    async def generate_json_with_retry(self, prompt: str, schema: Dict[str, Any] = None,
                                       max_retries: int = 3) -> Dict[str, Any]:
        """Generate JSON with retry logic - DON'T grow the prompt"""
        last_error = None
        original_prompt = prompt  # SAVE ORIGINAL

        logger.debug(f"Agent config type check:")
        for key, value in self.config.items():
            logger.debug(f"  {key}: {type(value)} = {value if not isinstance(value, dict) else '<dict>'}")

        for attempt in range(max_retries):
            try:
                # Use original prompt each time, not growing version
                current_prompt = original_prompt

                # Only add guidance on 2nd+ attempt
                if attempt > 0:
                    current_prompt = f"""RETRY ATTEMPT {attempt + 1}/{max_retries}

    {original_prompt}

    CRITICAL: Previous attempts failed. You MUST respond with ONLY valid JSON.
    Start with {{ or [ immediately. No other text."""

                response = await self.client.generate_json(current_prompt, schema, **self.config)

                if response:
                    # Validate it's not empty
                    if isinstance(response, (dict, list)) and response:
                        return response
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Empty response")
            except ValueError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} unexpected error: {e}")
                if attempt == max_retries - 1:
                    raise

        raise ValueError(f"Failed after {max_retries} attempts. Last error: {last_error}")
    def format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt"""
        formatted_parts = []

        if 'similar_tests' in context:
            formatted_parts.append("Similar Test Cases:")
            for test in context['similar_tests'][:3]:  # Limit to top 3
                formatted_parts.append(f"- {test.get('metadata', {}).get('name', 'Test')}")

        if 'edge_cases' in context:
            formatted_parts.append("\nRelevant Edge Cases:")
            for edge in context['edge_cases'][:3]:
                formatted_parts.append(f"- {edge.get('metadata', {}).get('description', 'Edge case')}")

        if 'validation_patterns' in context:
            formatted_parts.append("\nValidation Patterns:")
            for pattern in context['validation_patterns'][:3]:
                formatted_parts.append(f"- {pattern.get('metadata', {}).get('type', 'Validation')}")

        return "\n".join(formatted_parts)

    def validate_response(self, response: Any) -> bool:
        """Validate agent response"""
        return response is not None and response != ""