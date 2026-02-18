"""
Llama 3.2 configuration for LM Studio
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)

@dataclass
class LlamaConfig:
    """Configuration for qwen 2.5-7B-instruct via LM Studio"""

    # Connection settings
    base_url: str = os.getenv("LLAMA_BASE_URL", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")

    # Model parameters - OPTIMIZED FOR 3B MODEL
    temperature: float = 0.7
    max_tokens: int = 1500  # REDUCED from 2048
    top_p: float = 0.95
    top_k: int = 40
    frequency_penalty: float = 0.5  # INCREASED to reduce repetition
    presence_penalty: float = 0.5   # INCREASED to reduce repetition
    context_window: int = 8192

    # Agent-specific temperatures - LOWER FOR MORE RELIABILITY
    agent_temperatures: Dict[str, float] = field(default_factory=lambda: {
        'analyzer': 0.2,  # ✅ Just a float
        'test_designer': 0.3,  # ✅ Just a float
        'edge_case': 0.6,  # ✅ Just a float
        'data_generator': 0.4,  # ✅ Just a float
        'report_writer': 0.3  # ✅ Just a float
    })

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 60

    # Response settings
    stream: bool = False
    # CRITICAL FIX: Remove "\n\n\n" that causes premature stopping
    stop_sequences: list = field(default_factory=lambda: [])

    def __post_init__(self):
        """Debug: Check what agent_temperatures actually is"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"LlamaConfig initialized!")
        logger.info(f"  agent_temperatures type: {type(self.agent_temperatures)}")
        logger.info(f"  agent_temperatures value: {self.agent_temperatures}")

        # Verify each temperature is a float
        if isinstance(self.agent_temperatures, dict):
            for agent, temp in self.agent_temperatures.items():
                logger.info(f"    {agent}: {type(temp).__name__} = {temp}")

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Building config for agent type: '{agent_type}'")

        # DEFENSIVE: Explicitly extract temperature as float
        temp = self.temperature  # default fallback

        # Check if agent_temperatures exists and is a dict
        if hasattr(self, 'agent_temperatures') and isinstance(self.agent_temperatures, dict):
            if agent_type in self.agent_temperatures:
                temp_value = self.agent_temperatures[agent_type]
                # Ensure it's a number
                if isinstance(temp_value, (int, float)):
                    temp = float(temp_value)
                    logger.info(f"Using agent-specific temperature: {temp}")
                else:
                    logger.error(f"❌ Agent temperature for '{agent_type}' is NOT a number! Type: {type(temp_value)}")
                    logger.error(f"   Value: {temp_value}")
            else:
                logger.warning(f"Agent type '{agent_type}' not in agent_temperatures, using default: {temp}")
        else:
            logger.error(f"❌ agent_temperatures is not a dict! Type: {type(self.agent_temperatures)}")
            logger.error(f"   Value: {self.agent_temperatures}")

        config = {
            'temperature': temp,  # ✅ Now guaranteed to be a float
            'max_tokens': int(self.max_tokens),
            'top_p': float(self.top_p),
            'top_k': int(self.top_k),
            'frequency_penalty': float(self.frequency_penalty),
            'presence_penalty': float(self.presence_penalty),
            'stream': bool(self.stream),
            'stop': list(self.stop_sequences) if isinstance(self.stop_sequences, list) else []
        }

        logger.info(f"Config built successfully for '{agent_type}'")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'base_url': self.base_url,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'context_window': self.context_window
        }


# Global instance
llama_config = LlamaConfig()