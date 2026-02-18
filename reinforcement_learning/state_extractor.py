"""
RL State Extraction - Fixed version

Original had 640-dimension state vectors with only ~20 populated (95% zeros).
This version uses a compact 64-dimension state that networks can actually learn from.

State vector layout (64 dimensions):
  [0-15]  API features (16 dims)
  [16-31] Test suite features (16 dims)
  [32-47] History/feedback features (16 dims)
  [48-63] Context features (16 dims)
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Fixed dimensions
API_DIM = 16
TEST_DIM = 16
HISTORY_DIM = 16
CONTEXT_DIM = 16
TOTAL_STATE_DIM = API_DIM + TEST_DIM + HISTORY_DIM + CONTEXT_DIM  # 64


def extract_state(test_cases: List[Dict], api_spec: Dict,
                  history: Optional[List[Dict]] = None,
                  context: Optional[Dict] = None) -> np.ndarray:
    """
    Build a compact 64-dim state vector from all available info.

    Args:
        test_cases: Current test cases generated
        api_spec: Parsed API specification
        history: Previous execution results (optional)
        context: RAG context (optional)

    Returns:
        np.ndarray of shape (64,), normalized to [-1, 1]
    """
    state = np.zeros(TOTAL_STATE_DIM, dtype=np.float32)

    state[0:API_DIM] = extract_api_features(api_spec)
    state[API_DIM:API_DIM + TEST_DIM] = extract_test_features(test_cases)
    state[API_DIM + TEST_DIM:API_DIM + TEST_DIM + HISTORY_DIM] = extract_history_features(history)
    state[API_DIM + TEST_DIM + HISTORY_DIM:] = extract_context_features(context)

    return state


def extract_api_features(api_spec: Dict) -> np.ndarray:
    """Extract 16 features from the API specification."""
    features = np.zeros(API_DIM, dtype=np.float32)

    if not api_spec:
        return features

    endpoints = api_spec.get('endpoints', [])
    models = api_spec.get('models', [])
    validation_rules = api_spec.get('validation_rules', [])

    # Basic counts (normalized by reasonable maximums)
    features[0] = min(len(endpoints) / 50.0, 1.0)          # endpoint count
    features[1] = min(len(models) / 20.0, 1.0)             # model count
    features[2] = min(len(validation_rules) / 50.0, 1.0)   # validation rule count

    # HTTP method distribution
    methods = [ep.get('http_method', '').upper() for ep in endpoints]
    total = max(len(methods), 1)
    features[3] = methods.count('GET') / total
    features[4] = methods.count('POST') / total
    features[5] = methods.count('PUT') / total
    features[6] = methods.count('DELETE') / total
    features[7] = methods.count('PATCH') / total

    # Parameter complexity
    total_params = 0
    total_path_params = 0
    total_query_params = 0
    total_body_params = 0
    auth_endpoints = 0

    for ep in endpoints:
        params = ep.get('parameters', [])
        total_params += len(params)
        for p in params:
            loc = p.get('location', p.get('in', ''))
            if loc == 'path':
                total_path_params += 1
            elif loc == 'query':
                total_query_params += 1
            elif loc in ('body', 'formData'):
                total_body_params += 1

        if ep.get('auth_required') or ep.get('requires_auth'):
            auth_endpoints += 1

    features[8] = min(total_params / 100.0, 1.0)           # total params
    features[9] = min(total_path_params / 30.0, 1.0)       # path params
    features[10] = min(total_query_params / 30.0, 1.0)     # query params
    features[11] = min(total_body_params / 50.0, 1.0)      # body params
    features[12] = auth_endpoints / total if endpoints else 0  # auth ratio

    # Model complexity
    total_fields = sum(
        len(m.get('properties', m.get('fields', [])))
        for m in models
    )
    features[13] = min(total_fields / 100.0, 1.0)

    # Has nested routes
    features[14] = 1.0 if any('/' in ep.get('path', '').strip('/') for ep in endpoints) else 0.0

    # Controller count
    controllers = api_spec.get('controllers', [])
    features[15] = min(len(controllers) / 10.0, 1.0)

    return features


def extract_test_features(test_cases: List[Dict]) -> np.ndarray:
    """Extract 16 features from the current test suite."""
    features = np.zeros(TEST_DIM, dtype=np.float32)

    if not test_cases:
        return features

    n = len(test_cases)
    features[0] = min(n / 100.0, 1.0)  # test count

    # Type distribution
    types = [tc.get('test_type', tc.get('type', 'unknown')).lower() for tc in test_cases]
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    # Map known types to features
    type_map = {
        'happy_path': 1, 'smoke': 1,
        'negative': 2, 'validation': 2,
        'edge_case': 3, 'boundary': 3,
        'security': 4,
        'auth': 5, 'authorization': 5,
        'performance': 6,
    }

    for test_type, count in type_counts.items():
        for key, idx in type_map.items():
            if key in test_type:
                features[idx] = min(features[idx] + count / n, 1.0)
                break

    # Priority distribution
    priorities = [tc.get('priority', 'medium').lower() for tc in test_cases]
    features[7] = priorities.count('high') / n if n else 0
    features[8] = priorities.count('medium') / n if n else 0
    features[9] = priorities.count('low') / n if n else 0

    # Method coverage
    methods = [tc.get('method', '').upper() for tc in test_cases]
    unique_methods = len(set(m for m in methods if m))
    features[10] = min(unique_methods / 5.0, 1.0)

    # Endpoint coverage
    endpoints = set(tc.get('endpoint', '') for tc in test_cases if tc.get('endpoint'))
    features[11] = min(len(endpoints) / 20.0, 1.0)

    # Has assertions
    has_assertions = sum(1 for tc in test_cases if tc.get('assertions'))
    features[12] = has_assertions / n if n else 0

    # Has test data
    has_data = sum(1 for tc in test_cases if tc.get('test_data'))
    features[13] = has_data / n if n else 0

    # Average steps per test
    avg_steps = np.mean([len(tc.get('steps', [])) for tc in test_cases]) if test_cases else 0
    features[14] = min(avg_steps / 10.0, 1.0)

    # Type diversity (entropy-like)
    if type_counts:
        probs = np.array(list(type_counts.values()), dtype=np.float32) / n
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features[15] = min(entropy / 2.0, 1.0)  # normalize

    return features


def extract_history_features(history: Optional[List[Dict]]) -> np.ndarray:
    """Extract 16 features from execution history."""
    features = np.zeros(HISTORY_DIM, dtype=np.float32)

    if not history:
        return features

    n = len(history)
    features[0] = min(n / 100.0, 1.0)  # history size

    # Pass/fail rates
    passed = sum(1 for r in history if r.get('passed'))
    failed = n - passed
    features[1] = passed / n if n else 0  # pass rate
    features[2] = failed / n if n else 0  # fail rate

    # Execution times
    times = [r.get('execution_time', 0) for r in history if r.get('execution_time')]
    if times:
        features[3] = min(np.mean(times) / 10.0, 1.0)   # avg time
        features[4] = min(np.max(times) / 30.0, 1.0)     # max time
        features[5] = min(np.std(times) / 5.0, 1.0)      # time variance

    # Error type distribution
    errors = [r.get('error', '') for r in history if r.get('error')]
    features[6] = min(len(errors) / max(n, 1), 1.0)  # error rate

    timeout_errors = sum(1 for e in errors if 'timeout' in str(e).lower())
    features[7] = timeout_errors / max(len(errors), 1)

    connection_errors = sum(1 for e in errors if 'connection' in str(e).lower())
    features[8] = connection_errors / max(len(errors), 1)

    validation_errors = sum(1 for e in errors if any(
        w in str(e).lower() for w in ['validation', '400', '422']
    ))
    features[9] = validation_errors / max(len(errors), 1)

    auth_errors = sum(1 for e in errors if any(
        w in str(e).lower() for w in ['401', '403', 'unauthorized', 'forbidden']
    ))
    features[10] = auth_errors / max(len(errors), 1)

    # Status code distribution
    statuses = [r.get('actual_status', 0) for r in history if r.get('actual_status')]
    if statuses:
        features[11] = sum(1 for s in statuses if 200 <= s < 300) / len(statuses)  # 2xx
        features[12] = sum(1 for s in statuses if 400 <= s < 500) / len(statuses)  # 4xx
        features[13] = sum(1 for s in statuses if 500 <= s < 600) / len(statuses)  # 5xx

    # Trend: recent pass rate vs overall (last 10 vs all)
    if n >= 10:
        recent = history[-10:]
        recent_pass = sum(1 for r in recent if r.get('passed')) / 10
        features[14] = recent_pass - features[1]  # trend (positive = improving)

    # Unique endpoints tested
    tested_eps = set(r.get('endpoint', '') for r in history if r.get('endpoint'))
    features[15] = min(len(tested_eps) / 20.0, 1.0)

    return features


def extract_context_features(context: Optional[Dict]) -> np.ndarray:
    """Extract 16 features from RAG context."""
    features = np.zeros(CONTEXT_DIM, dtype=np.float32)

    if not context:
        return features

    similar = context.get('similar_tests', [])
    edge_cases = context.get('edge_cases', [])
    validation = context.get('validation_patterns', [])

    features[0] = min(len(similar) / 20.0, 1.0)
    features[1] = min(len(edge_cases) / 20.0, 1.0)
    features[2] = min(len(validation) / 20.0, 1.0)

    # Average similarity scores
    def avg_score(items):
        scores = []
        for item in items:
            if isinstance(item, dict):
                scores.append(item.get('score', item.get('similarity', 0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                scores.append(float(item[1]) if not isinstance(item[1], str) else 0)
        return np.mean(scores) if scores else 0

    features[3] = avg_score(similar)
    features[4] = avg_score(edge_cases)
    features[5] = avg_score(validation)

    # Has high-quality matches (score > 0.8)
    high_quality = sum(1 for s in similar if isinstance(s, dict) and s.get('score', 0) > 0.8)
    features[6] = min(high_quality / 5.0, 1.0)

    # Total context items
    total = len(similar) + len(edge_cases) + len(validation)
    features[7] = min(total / 50.0, 1.0)

    return features