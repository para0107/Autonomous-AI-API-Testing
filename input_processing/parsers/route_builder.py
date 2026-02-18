"""
C# Parser route building utilities - Fixed version

Fix: _build_route no longer blindly prepends 'api/' to routes that already
have valid prefixes like 'v1/', 'v2/', or custom prefixes.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Known valid route prefixes that should NOT get 'api/' prepended
VALID_PREFIXES = (
    'api/', 'v1/', 'v2/', 'v3/', 'v4/',
    'api/v1/', 'api/v2/', 'api/v3/',
    'odata/', 'graphql/', 'rest/',
)


def build_route(controller_route: str, action_route: Optional[str] = None,
                controller_name: str = '') -> str:
    """
    Build a complete API route from controller and action routes.

    Args:
        controller_route: Route attribute on the controller class
        action_route: Route attribute on the action method (optional)
        controller_name: Controller class name for [controller] token replacement

    Returns:
        Complete route string like 'api/users/{id}'
    """
    # Clean controller route
    base_route = _clean_route(controller_route)

    # Replace [controller] token
    if '[controller]' in base_route:
        # Strip 'Controller' suffix: UsersController -> users
        name = controller_name
        if name.endswith('Controller'):
            name = name[:-len('Controller')]
        base_route = base_route.replace('[controller]', name.lower())

    # Replace [action] token if present (less common)
    if '[action]' in base_route:
        base_route = base_route.replace('[action]', '')

    # FIX: Only prepend 'api/' if route doesn't already have a valid prefix
    if base_route and not _has_valid_prefix(base_route):
        # Check if it's a bare controller name (e.g., 'users')
        # In ASP.NET, routes without a prefix are relative — we add api/ as convention
        # But only if it looks like it needs one
        if not base_route.startswith('/'):
            base_route = f"api/{base_route}"

    # Combine with action route
    if action_route:
        action_clean = _clean_route(action_route)
        if action_clean:
            # If action route starts with /, it's absolute (overrides controller route)
            if action_clean.startswith('/'):
                return _normalize_route(action_clean.lstrip('/'))
            else:
                return _normalize_route(f"{base_route}/{action_clean}")

    return _normalize_route(base_route)


def _clean_route(route: str) -> str:
    """Clean a route string."""
    if not route:
        return ''

    route = route.strip().strip('"').strip("'")

    # Remove leading/trailing slashes
    route = route.strip('/')

    # Remove ~ prefix (ASP.NET)
    if route.startswith('~'):
        route = route.lstrip('~').lstrip('/')

    return route


def _has_valid_prefix(route: str) -> bool:
    """Check if route already has a recognized prefix."""
    route_lower = route.lower()
    return any(route_lower.startswith(prefix) for prefix in VALID_PREFIXES)


def _normalize_route(route: str) -> str:
    """Normalize route: remove double slashes, ensure no leading/trailing slash."""
    # Remove double slashes
    while '//' in route:
        route = route.replace('//', '/')

    # Strip leading/trailing slashes
    route = route.strip('/')

    return route


def extract_route_parameters(route: str):
    """Extract route parameters like {id}, {userId:int}."""
    params = []
    matches = re.findall(r'\{(\w+)(?::(\w+))?\}', route)
    for name, constraint in matches:
        params.append({
            'name': name,
            'location': 'path',
            'required': True,
            'type': _map_constraint_type(constraint) if constraint else 'string',
        })
    return params


def _map_constraint_type(constraint: str) -> str:
    """Map ASP.NET route constraints to standard types."""
    mapping = {
        'int': 'integer',
        'long': 'integer',
        'float': 'number',
        'double': 'number',
        'decimal': 'number',
        'bool': 'boolean',
        'guid': 'string',
        'datetime': 'string',
        'alpha': 'string',
        'regex': 'string',
    }
    return mapping.get(constraint.lower(), 'string')