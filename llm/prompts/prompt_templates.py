"""
Prompt templates for different testing scenarios
"""


class PromptTemplates:
    """Collection of prompt templates"""

    # API Analysis Templates
    API_ANALYSIS = """Analyze the following API endpoint for testing requirements:

{api_specification}

Identify:
1. Critical test scenarios
2. Edge cases and boundary conditions
3. Security vulnerabilities
4. Performance considerations
5. Data validation requirements

Provide a structured analysis."""

    # Test Generation Templates
    HAPPY_PATH_TEST = """Generate a happy path test case for:
Endpoint: {endpoint}
Method: {method}
Parameters: {parameters}

Create a test that verifies successful operation with valid inputs."""

    EDGE_CASE_TEST = """Generate edge case tests for:
Parameter: {parameter_name}
Type: {parameter_type}
Constraints: {constraints}

Include boundary values, null/empty cases, and extreme values."""

    SECURITY_TEST = """Generate security test cases for:
Endpoint: {endpoint}
Authentication: {auth_type}

Test for:
- SQL Injection
- XSS attacks
- Authentication bypass
- Authorization elevation"""

    # Data Generation Templates
    VALID_DATA = """Generate valid test data for:
{parameter_specification}

Ensure all constraints are satisfied and data is realistic."""

    INVALID_DATA = """Generate invalid test data to test validation for:
{parameter_specification}

Include type mismatches, constraint violations, and malformed data."""

    # Report Generation Templates
    TEST_SUMMARY = """Summarize the test execution results:
Total: {total}
Passed: {passed}
Failed: {failed}

Provide insights and recommendations."""

    FAILURE_ANALYSIS = """Analyze this test failure:
Test: {test_name}
Expected: {expected}
Actual: {actual}
Error: {error}

Identify root cause and suggest fixes."""

    # RAG Enhancement Templates
    CONTEXT_INTEGRATION = """Given the following context from similar tests:
{context}

Adapt these patterns for testing:
{current_api}

Generate relevant test cases."""

    # Validation Templates
    VALIDATION_RULES = """Extract validation rules from:
{code_snippet}

Identify:
- Required fields
- Type constraints
- Format requirements
- Business rules"""

    # Custom Templates
    CUSTOM = """
{custom_prompt}
"""

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Get template by name"""
        return getattr(cls, template_name.upper(), cls.CUSTOM)