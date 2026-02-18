"""
Configuration logging fix.

Original dumped every config value at INFO level in __post_init__,
get_agent_config, and _get_config. This floods logs.

Fix: Config values logged at DEBUG level. Only a brief summary at INFO.
"""

import logging
from dataclasses import dataclass, field, fields
from typing import Dict, Any

logger = logging.getLogger(__name__)


def log_config_summary(config_obj, config_name: str = "Config"):
    """
    Log a brief summary at INFO and full details at DEBUG.

    Use this instead of dumping every field at INFO.
    """
    all_fields = fields(config_obj)
    non_default = []

    for f in all_fields:
        value = getattr(config_obj, f.name)
        if value != f.default and f.default is not f.default_factory:
            non_default.append(f.name)

    logger.info(
        f"{config_name} loaded: {len(all_fields)} settings "
        f"({len(non_default)} non-default)"
    )

    # Full dump at DEBUG only
    if logger.isEnabledFor(logging.DEBUG):
        for f in all_fields:
            value = getattr(config_obj, f.name)
            # Mask sensitive values
            if any(s in f.name.lower() for s in ('key', 'token', 'secret', 'password')):
                display = f"{str(value)[:4]}..." if value else "NOT SET"
            else:
                display = value
            logger.debug(f"  {config_name}.{f.name} = {display}")


# Example usage pattern for LlamaConfig.__post_init__:
#
# def __post_init__(self):
#     log_config_summary(self, "LlamaConfig")
#
# Instead of:
#     logger.info(f"API Key: {self.api_key[:4]}...")
#     logger.info(f"Base URL: {self.base_url}")
#     logger.info(f"Model: {self.model}")
#     ... (20 more lines)