"""
LLM Client connection checking - Fixed version

Fix: check_connection no longer relies solely on /models endpoint.
Tries /models first, then falls back to a minimal completion call.
"""

import logging
import aiohttp
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


async def check_connection(base_url: str, api_key: str,
                           timeout: int = 10) -> bool:
    """
    Check if the LLM API is reachable and the API key is valid.

    Tries (in order):
      1. GET /models (works for OpenAI, Groq, most compatible APIs)
      2. POST /chat/completions with minimal payload (universal fallback)

    Args:
        base_url: API base URL (e.g., 'https://api.groq.com/openai/v1')
        api_key: API key
        timeout: Request timeout in seconds

    Returns:
        True if API is reachable and key is valid
    """
    if not api_key:
        logger.error("No API key provided")
        return False

    if not base_url:
        logger.error("No base URL provided")
        return False

    base_url = base_url.rstrip('/')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        # Method 1: GET /models
        try:
            async with session.get(
                f"{base_url}/models", headers=headers
            ) as resp:
                if resp.status == 200:
                    logger.info("Connection check passed via /models endpoint")
                    return True
                elif resp.status == 401:
                    logger.error("API key is invalid (401 Unauthorized)")
                    return False
                elif resp.status == 403:
                    logger.error("API key lacks permissions (403 Forbidden)")
                    return False
                else:
                    logger.info(
                        f"/models returned {resp.status}, "
                        f"trying fallback..."
                    )
        except asyncio.TimeoutError:
            logger.warning("/models endpoint timed out, trying fallback...")
        except aiohttp.ClientError as e:
            logger.warning(f"/models endpoint failed: {e}, trying fallback...")

        # Method 2: Minimal completion call
        try:
            payload = {
                "model": "llama-3.1-8b-instant",  # Small model, most providers have it
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status == 200:
                    logger.info("Connection check passed via completions endpoint")
                    return True
                elif resp.status == 401:
                    logger.error("API key is invalid (401)")
                    return False
                elif resp.status == 404:
                    # Model not found, but API is reachable and key works
                    logger.info("API reachable (model not found but auth OK)")
                    return True
                elif resp.status == 429:
                    # Rate limited but API is reachable and key works
                    logger.info("API reachable (rate limited but auth OK)")
                    return True
                else:
                    body = await resp.text()
                    logger.warning(
                        f"Completions check returned {resp.status}: {body[:200]}"
                    )
                    # If we got a response at all, the connection works
                    return resp.status < 500

        except asyncio.TimeoutError:
            logger.error("Connection check timed out on both endpoints")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Connection check failed: {e}")
            return False

    return False