"""
HTTP API client for test execution
"""

import logging
import aiohttp
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP client for API requests"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, endpoint: str, params: Dict = None,
                  headers: Dict = None) -> Dict[str, Any]:
        """GET request"""
        return await self.request('GET', endpoint, params=params, headers=headers)

    async def post(self, endpoint: str, data: Dict = None,
                   headers: Dict = None) -> Dict[str, Any]:
        """POST request"""
        return await self.request('POST', endpoint, json=data, headers=headers)

    async def put(self, endpoint: str, data: Dict = None,
                  headers: Dict = None) -> Dict[str, Any]:
        """PUT request"""
        return await self.request('PUT', endpoint, json=data, headers=headers)

    async def delete(self, endpoint: str, headers: Dict = None) -> Dict[str, Any]:
        """DELETE request"""
        return await self.request('DELETE', endpoint, headers=headers)

    async def patch(self, endpoint: str, data: Dict = None,
                    headers: Dict = None) -> Dict[str, Any]:
        """PATCH request"""
        return await self.request('PATCH', endpoint, json=data, headers=headers)

    async def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Generic request method"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        try:
            async with self.session.request(method, url, **kwargs) as response:
                status = response.status
                headers = dict(response.headers)

                try:
                    body = await response.json()
                except:
                    body = await response.text()

                return {
                    'status': status,
                    'headers': headers,
                    'body': body,
                    'ok': 200 <= status < 300
                }
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise