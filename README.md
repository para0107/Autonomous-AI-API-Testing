# API Testing Agent - Comprehensive Technical Documentation (Continuation from this repo: github.com/para0107/api-testing-agent )
## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Configuration System](#configuration-system)
4. [Input Processing Module](#input-processing-module)
5. [RAG (Retrieval-Augmented Generation) System](#rag-system)
6. [LLM Orchestration Layer](#llm-orchestration-layer)
7. [Reinforcement Learning Optimizer](#reinforcement-learning-optimizer)
8. [Core Orchestration Engine](#core-orchestration-engine)
9. [Test Execution & Output](#test-execution--output)
10. [Data Flow & Processing Pipeline](#data-flow--processing-pipeline)
11. [Scripts & Utilities](#scripts--utilities)
12. [Installation & Setup](#installation--setup)
13. [Usage Examples](#usage-examples)

---

## System Overview

The **API Testing Agent** is an advanced, AI-powered automated testing system designed to generate, optimize, and execute comprehensive test suites for RESTful APIs. The system leverages multiple cutting-edge technologies including:

- **Multi-Language Code Parsing**: Supports C#, Java, Python, and C++ API frameworks
- **RAG (Retrieval-Augmented Generation)**: FAISS-based vector database for contextual test generation
- **Multi-Agent LLM System**: Coordinated Llama 3.2 agents via LM Studio for specialized testing tasks
- **Reinforcement Learning**: PPO-based optimization for intelligent test case selection and prioritization
- **Automated Test Execution**: Complete test lifecycle management with detailed reporting

### Key Features

1. **Intelligent Code Analysis**: Automatically parses API code to extract endpoints, parameters, validation rules, and business logic
2. **Context-Aware Test Generation**: Uses historical test data and similar API patterns via RAG to generate relevant tests
3. **Multi-Agent Orchestration**: Specialized LLM agents handle different aspects (analysis, test design, edge cases, data generation, reporting)
4. **RL-Based Optimization**: Continuously learns optimal test strategies through reinforcement learning
5. **Comprehensive Test Coverage**: Generates happy path, validation, authentication, edge case, security, and performance tests
6. **QASE-Style Reporting**: Professional test reports with detailed execution results and recommendations

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│            (API Code Files + Configuration)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INPUT PROCESSING MODULE                        │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Parser Factory │→ │ Language Parser │→ │  Specification  │ │
│  │                │  │  (C#/Java/Py)   │  │    Builder      │ │
│  └────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (API Specification)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Embedding   │→ │ Vector Store │→ │  Retriever   │         │
│  │   Manager    │  │   (FAISS)    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Chunking   │  │ Knowledge    │                            │
│  │   Strategy   │  │    Base      │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (Retrieved Context)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               LLM MULTI-AGENT ORCHESTRATION                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Analyzer   │→ │    Test     │→ │  Edge Case  │            │
│  │   Agent     │  │  Designer   │  │    Agent    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │    Data     │  │   Report    │                              │
│  │  Generator  │  │   Writer    │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (Generated Test Cases)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│            REINFORCEMENT LEARNING OPTIMIZER                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Policy    │  │    Value    │  │ Experience  │            │
│  │  Network    │  │  Network    │  │   Buffer    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐                                                │
│  │   Reward    │                                                │
│  │ Calculator  │                                                │
│  └─────────────┘                                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (Optimized Test Suite)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TEST EXECUTION ENGINE                          │
│                (Execute Against Live API)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ (Execution Results)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEEDBACK LOOP & REPORTING                      │
│  - Update RAG with new patterns                                  │
│  - Train RL model on results                                     │
│  - Generate QASE-style reports                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Input Processing**: Parses API code → Extracts endpoints, parameters, validations
2. **RAG Context Retrieval**: Generates embeddings → Searches vector store → Returns similar patterns
3. **LLM Test Generation**: Multi-agent system generates comprehensive test cases
4. **RL Optimization**: Selects and prioritizes tests based on learned strategies
5. **Test Execution**: Runs tests against API → Collects results
6. **Feedback & Learning**: Updates knowledge base and RL model → Generates reports

---

## Configuration System

### Location: `config/`

The configuration system provides centralized settings for all system components.

### **settings.py**
Global application settings and paths.

**Key Classes:**
- `Settings`: Main application configuration
  - `APP_NAME`: Application identifier
  - `VERSION`: Current version
  - `DEBUG`: Debug mode flag
  - `MAX_WORKERS`: Parallel processing workers
  - `BATCH_SIZE`: Batch processing size
  - `SUPPORTED_LANGUAGES`: ["csharp", "python", "java", "cpp"]
  - `MAX_TESTS_PER_ENDPOINT`: Test generation limit (default: 50)
  - Test flags: `INCLUDE_EDGE_CASES`, `INCLUDE_NEGATIVE_TESTS`, `INCLUDE_SECURITY_TESTS`

- `PathConfig`: Directory structure management
  - `BASE_DIR`: Project root
  - `DATA_DIR`: Data storage root
  - `TRAINING_DATA_DIR`: RL training data
  - `VECTOR_STORE_DIR`: FAISS indices
  - `MODELS_DIR`: Trained models
  - `REPORTS_DIR`: Generated reports

**Usage:**
```python
from config import settings, paths

max_workers = settings.MAX_WORKERS
vector_dir = paths.VECTOR_STORE_DIR
```

### **llama_config.py**
Configuration for Llama 3.2 LLM via LM Studio.

**LlamaConfig Class:**
```python
@dataclass
class LlamaConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "not-needed"
    model_name: str = "llama-3.2-3b-instruct"
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    frequency_penalty: float = 0.3
    presence_penalty: float = 0.3
    context_window: int = 8192
```

**Agent-Specific Temperatures:**
- `analyzer`: 0.3 (low - deterministic analysis)
- `test_designer`: 0.5 (moderate - balanced creativity)
- `edge_case`: 0.8 (high - creative edge case generation)
- `data_generator`: 0.6 (moderate-high - diverse data)
- `report_writer`: 0.4 (low-moderate - consistent reports)

**Key Methods:**
- `get_agent_config(agent_type)`: Returns optimized config for specific agent
- `to_dict()`: Exports configuration as dictionary

### **rag_config.py**
RAG system configuration with FAISS.

**RAGConfig Class:**
```python
@dataclass
class RAGConfig:
    # Embedding models
    text_embedding_model: str = "all-mini-lm-l6-v2"
    code_embedding_model: str = "microsoft/codebert-base"
    embedding_dimension: int = 768
    
    # FAISS settings
    index_type: str = "IVF"  # IVF, HNSW, or Flat
    nlist: int = 100  # Number of clusters
    nprobe: int = 10  # Clusters to search
    metric: str = "cosine"
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 100
    
    # Retrieval
    top_k: int = 10
    similarity_threshold: float = 0.7
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

**Knowledge Levels:**
- `global`: General API patterns and REST principles
- `domain`: Business domain specific knowledge
- `service`: Service-specific patterns
- `endpoint`: Endpoint-specific test cases
- `edge_cases`: Edge cases and bug patterns

**Vector Store Indices:**
- `test_patterns`: Common test patterns
- `edge_cases`: Edge case scenarios
- `validation_rules`: Validation patterns
- `api_specifications`: API specs
- `bug_patterns`: Known bugs
- `successful_tests`: Passing tests

### **rl_config.py**
Reinforcement Learning (PPO) configuration.

**RLConfig Class:**
```python
@dataclass
class RLConfig:
    # PPO Algorithm
    algorithm: str = "PPO"
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    
    # Networks
    policy_hidden_sizes: List[int] = [256, 128, 64]
    value_hidden_sizes: List[int] = [256, 128, 64]
    
    # Learning rate schedule
    initial_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    lr_schedule: str = "cosine_annealing"
    
    # Training
    batch_size: int = 64
    mini_batch_size: int = 32
    n_epochs: int = 10
    max_grad_norm: float = 0.5
    
    # Experience buffer
    buffer_size: int = 10000
    prioritized_replay: bool = True
```

**State Space Dimensions:**
- `api_complexity`: 128 features
- `test_coverage`: 64 features
- `historical_bugs`: 256 features
- `execution_results`: 128 features
- `parameter_space`: 64 features

**Action Types (Test Categories):**
- `happy_path`, `boundary_value`, `null_empty`, `type_mismatch`
- `format_violation`, `business_logic`, `security_test`
- `concurrent_access`, `state_transition`, `integration_test`

**Reward Weights:**
```python
reward_weights = {
    'bug_found': 10.0,
    'code_coverage': 5.0,
    'edge_case_covered': 8.0,
    'unique_scenario': 6.0,
    'false_positive': -3.0,
    'redundant_test': -2.0,
    'test_failed': -1.0,
    'api_error': -5.0
}
```

**Key Methods:**
- `get_learning_rate(step)`: Calculates LR based on schedule (linear, exponential, cosine)
- `calculate_reward(metrics)`: Computes total reward from execution metrics

---

## Input Processing Module

### Location: `input_processing/`

Responsible for parsing API code in multiple languages and building unified API specifications.

### **Architecture:**

```
parser_factory.py → Language-Specific Parser → endpoint_extractor.py
                                              → validator_extractor.py
                                              → specification_builder.py
                                                        ↓
                                            OpenAPI Specification
```

### **parser_factory.py**

**ParserFactory Class:**
Factory pattern for creating language-specific parsers.

**Methods:**
- `get_parser(language: str) -> BaseParser`: Returns appropriate parser instance
  - Supported: `csharp`, `python`, `java`, `cpp`
  - Raises `ValueError` for unsupported languages

- `register_parser(language: str, parser_class: type)`: Dynamically registers new parser
  - Validates inheritance from `BaseParser`

- `get_supported_languages() -> list`: Returns list of supported languages

**Usage:**
```python
factory = ParserFactory()
parser = factory.get_parser('csharp')
parsed_data = parser.parse(['Controller.cs'])
```

### **parsers/base_parser.py**

**BaseParser (Abstract Class):**
Defines interface for all language parsers.

**Abstract Methods:**
- `parse(code_files: List[str]) -> Dict[str, Any]`: Main parsing entry point
- `extract_endpoints(code: str) -> List[Dict]`: Extract API endpoints
- `extract_methods(code: str) -> List[Dict]`: Extract all methods
- `extract_parameters(method_code: str) -> List[Dict]`: Extract method parameters
- `extract_validation_rules(code: str) -> List[Dict]`: Extract validation logic

**Concrete Methods:**
- `read_file(file_path: str) -> str`: Reads source file with UTF-8 encoding
- `combine_results(results: List[Dict]) -> Dict`: Merges multi-file parsing results
- `extract_dependencies(code: str) -> List[str]`: Extracts imports/using statements
- `extract_models(code: str) -> List[Dict]`: Extracts DTOs and data models

**Return Structure:**
```python
{
    'endpoints': [...],
    'services': [...],
    'validators': [...],
    'methods': [...],
    'models': [...],
    'dependencies': [...]
}
```

### **parsers/csharp_parser.py**

**CSharpParser Class:**
Parses C# ASP.NET Core API code.

**Regex Patterns:**
```python
patterns = {
    'class': r'public\s+class\s+(\w+)',
    'route': r'\[Route\("([^"]+)"\)\]',
    'http_method': r'\[Http(Get|Post|Put|Delete|Patch)(?:\("([^"]*)"\))?\]',
    'authorize': r'\[Authorize(?:\(([^)]*)\))?\]',
    'from_body': r'\[FromBody\]\s*(\w+)\s+(\w+)',
    'from_query': r'\[FromQuery\]\s*(\w+)\s+(\w+)',
    'from_route': r'\[FromRoute\]\s*(\w+)\s+(\w+)',
    'service_injection': r'private\s+readonly\s+(\w+)\s+_(\w+);',
    'validation': r'RuleFor\(.*?\)\..*?;'
}
```

**Key Methods:**

1. **extract_endpoints(code: str)**
   - Finds controller class and base route attribute
   - Locates HTTP method attributes (HttpGet, HttpPost, etc.)
   - Extracts method signatures and parameters
   - Parses authorization requirements
   - Returns list of endpoint definitions with:
     - `controller`, `method_name`, `http_method`, `route`
     - `return_type`, `parameters`, `authorization`

2. **extract_parameters(method_code: str)**
   - Identifies `[FromBody]`, `[FromQuery]`, `[FromRoute]` attributes
   - Parses parameter types and names
   - Determines required vs optional (nullable types)
   - Returns: `name`, `type`, `source`, `required`

3. **extract_validation_rules(code: str)**
   - Finds FluentValidation classes (`AbstractValidator<T>`)
   - Parses validation rules: `NotEmpty()`, `NotNull()`, `Length()`, `EmailAddress()`, `Matches()`
   - Extracts custom validation messages
   - Returns validator definitions with parsed rules

4. **extract_services(code: str)**
   - Finds dependency injection patterns
   - Extracts service types and field names
   - Returns: `type`, `name`

5. **extract_models(code: str)**
   - Finds DTO/Model classes (suffixed with DTO, Model, Request, Response)
   - Extracts properties with types
   - Returns: `name`, `properties[]`

**Type Normalization:**
- `String` → `string`
- `Int32/Int64` → `integer`
- `Double/Float` → `number`
- `Boolean` → `boolean`
- `DateTime/DateOnly` → `string`

### **parsers/java_parser.py**

**JavaParser Class:**
Parses Spring Boot and JAX-RS API code.

**Regex Patterns:**
```python
patterns = {
    'rest_controller': r'@RestController',
    'request_mapping': r'@RequestMapping\([\'"]([^\'"]+)[\'"]\)',
    'get_mapping': r'@GetMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
    'post_mapping': r'@PostMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
    'path_variable': r'@PathVariable(?:\([\'"](\w+)[\'"\)])?\s+(\w+)\s+(\w+)',
    'request_param': r'@RequestParam(?:\([\'"](\w+)[\'"].*?\))?\s+(\w+)\s+(\w+)',
    'request_body': r'@RequestBody\s+(\w+)\s+(\w+)',
    'autowired': r'@Autowired\s+(?:private\s+)?(\w+)\s+(\w+)',
    'validation': r'@(NotNull|NotEmpty|NotBlank|Size|Min|Max|Pattern|Email)'
}
```

**Key Features:**
- Detects `@RestController` and `@Controller` annotations
- Parses Spring Boot mapping annotations (`@GetMapping`, `@PostMapping`, etc.)
- Extracts `@PathVariable`, `@RequestParam`, `@RequestBody` parameters
- Identifies Bean Validation annotations on model fields
- Handles `@Autowired` service dependencies

**Bean Validation Support:**
- `@NotNull`, `@NotEmpty`, `@NotBlank`
- `@Size(min=, max=)`, `@Min`, `@Max`
- `@Pattern(regexp=)`, `@Email`

### **parsers/python_parser.py**

**PythonParser Class:**
Parses Flask, FastAPI, and Django API code using AST.

**Dual Parsing Approach:**
1. **AST (Abstract Syntax Tree)**: For precise code structure
2. **Regex**: For framework-specific decorators

**Framework Detection:**
- Flask: `from flask import` or `@app.route`
- FastAPI: `from fastapi import` or `@app.get`
- Django: `from django` or `path()`

**AST Parsing:**
```python
def parse_ast(tree: ast.AST, source_code: str):
    # Walks AST nodes
    # ClassDef → parse_class()
    # FunctionDef/AsyncFunctionDef → parse_function()
    # Import/ImportFrom → parse_import()
```

**Key Methods:**

1. **extract_flask_endpoints(code: str)**
   - Pattern: `@app.route('/path', methods=['GET', 'POST'])`
   - Extracts route path, HTTP methods, function signature
   - Parses function parameters

2. **extract_fastapi_endpoints(code: str)**
   - Pattern: `@app.get('/path')` or `@router.post('/path')`
   - Parses type-hinted parameters: `param: str = Query(...)`
   - Identifies parameter sources: `Query()`, `Path()`, `Body()`, `Header()`
   - Handles async functions

3. **extract_models(code: str)**
   - Pydantic: `class Model(BaseModel):`
   - Dataclasses: `@dataclass class Model:`
   - Extracts field annotations, defaults, Field() definitions
   - Parses Pydantic validators (`@validator`, `@root_validator`)

4. **extract_validation_rules(code: str)**
   - Pydantic Field() parameters: `min_length`, `max_length`, `ge`, `le`, `regex`
   - Custom validators with `@validator` decorator
   - Returns structured validation rules

**Type Hint Parsing:**
- Handles complex types: `Optional[str]`, `List[int]`, `Dict[str, Any]`
- Extracts default values and required/optional status

### **parsers/cpp_parser.py**

**CppParser Class:**
Parses C++ REST frameworks (Crow, Pistache, Restbed, C++ REST SDK).

**Supported Frameworks:**
- **Crow**: `CROW_ROUTE(app, "/path")`
- **Pistache**: `Routes::Get(router, "/path")`
- **Restbed**: `resource->set_path("/path")`
- **cpprest**: `listener.support(methods::GET, "/path")`

**Framework Detection:**
```python
def detect_framework(code: str):
    if 'crow.h' or 'CROW_ROUTE' in code: return 'crow'
    elif 'pistache' in code: return 'pistache'
    elif 'restbed' in code: return 'restbed'
    elif 'cpprest' or 'http_listener' in code: return 'cpprest'
```

**Key Challenges:**
- Template metaprogramming syntax
- Nested angle brackets in type definitions
- Manual string parsing (no standard C++ parser)

**Methods:**
- `split_parameters(params_str)`: Handles nested templates in parameter lists
- `extract_class_members(code, class_name)`: Parses public/private/protected sections
- `extract_structs(code)`: Extracts POD structs for data models

### **endpoint_extractor.py**

**EndpointExtractor Class:**
Normalizes endpoints across different languages to unified format.

**Key Methods:**

1. **normalize_endpoint(endpoint: Dict) -> Dict**
   - Standardizes field names across languages
   - Maps language-specific terms to OpenAPI equivalents
   - Returns unified structure:
```python
{
    'path': '/api/users/{id}',
    'method': 'GET',
    'name': 'getUserById',
    'parameters': [...],
    'authentication': {...},
    'return_type': 'User',
    'description': '...',
    'tags': [...]
}
```

2. **normalize_path(path: str) -> str**
   - Ensures path starts with `/`
   - Removes trailing slashes
   - Converts path parameter formats:
     - Express: `:id` → `{id}`
     - Flask: `<id>` → `{id}`
     - Result: OpenAPI style `{id}`

3. **normalize_parameters(parameters: List) -> List**
   - Standardizes parameter definitions
   - Maps parameter locations: `body`, `query`, `path`, `header`, `formData`
   - Extracts constraints: `min`, `max`, `minLength`, `maxLength`, `pattern`, `enum`

4. **normalize_type(type_str: str) -> str**
   - Cross-language type mapping:
     - C#: `String` → `string`, `Int32` → `integer`
     - Python: `str` → `string`, `int` → `integer`
     - Java: `String` → `string`, `Integer` → `integer`
   - Handles generic types: `List<String>` → `array`

5. **extract_constraints(param: Dict) -> Dict**
   - Parses validation constraints
   - Extracts from validation attributes/decorators
   - Returns: `min`, `max`, `minLength`, `maxLength`, `pattern`, `enum`, `format`, `multipleOf`

6. **extract_auth_requirements(endpoint: Dict) -> Dict**
   - Identifies authentication needs
   - Determines auth type: `bearer`, `apiKey`, `oauth2`
   - Extracts required scopes
   - Returns:
```python
{
    'required': True/False,
    'type': 'bearer|apiKey|oauth2',
    'scopes': ['read', 'write']
}
```

7. **group_endpoints(endpoints: List) -> Dict**
   - Groups endpoints by resource (first path segment)
   - Skips version prefixes (`/v1`, `/v2`)
   - Returns: `{'users': [...], 'products': [...]}`

### **validator_extractor.py**

**ValidatorExtractor Class:**
Extracts and analyzes validation rules from multiple frameworks.

**Supported Frameworks:**
- FluentValidation (C#)
- Bean Validation (Java)
- Pydantic (Python)
- Custom inline validations

**Validation Patterns:**
```python
validation_patterns = {
    'fluent_validation': {
        'not_empty': r'\.NotEmpty\(\)',
        'not_null': r'\.NotNull\(\)',
        'length': r'\.Length\((\d+)(?:,\s*(\d+))?\)',
        'email': r'\.EmailAddress\(\)',
        'matches': r'\.Matches\("([^"]+)"\)'
    },
    'bean_validation': {
        'not_null': r'@NotNull',
        'size': r'@Size\((?:min=(\d+))?(?:,?\s*max=(\d+))?\)',
        'pattern': r'@Pattern\(regexp="([^"]+)"'
    },
    'pydantic': {
        'field': r'Field\(([^)]+)\)',
        'validator': r'@validator\([\'"](\w+)[\'"]'
    }
}
```

**Key Methods:**

1. **extract(parsed_data: Dict) -> List[Dict]**
   - Extracts validators from parsed code
   - Enhances with additional metadata
   - Extracts inline validations from method bodies
   - Returns list of validator definitions

2. **enhance_validator(validator: Dict) -> Dict**
   - Adds `category`: format, size, presence, pattern, range, custom
   - Determines `strength`: strict, moderate, lenient
   - Generates `test_cases` for each validation rule

3. **categorize_validation(validator: Dict) -> str**
   - Analyzes validation rules
   - Categories:
     - `format`: email, url, phone patterns
     - `size`: length, min/max constraints
     - `presence`: required, not null
     - `pattern`: regex matching
     - `range`: between values
     - `custom`: business logic

4. **determine_validation_strength(validator: Dict) -> str**
   - Counts validation rules
   - Checks for strict validations (required + pattern)
   - Returns: `strict` (3+ rules or required+pattern), `moderate` (2 rules or required), `lenient` (1 rule)

5. **generate_validation_test_cases(validator: Dict) -> List**
   - Creates test cases for each validation
   - Examples:
     - `NotEmpty`: empty string (fail), non-empty (pass)
     - `Length(5,10)`: 4 chars (fail), 5 chars (pass), 10 chars (pass), 11 chars (fail)
     - `EmailAddress`: "invalid" (fail), "test@example.com" (pass)
   - Returns: `input`, `expected` (pass/fail), `reason`

6. **extract_inline_validations(methods: List) -> List**
   - Scans method bodies for validation patterns:
     - `if (x == null)` → null_check
     - `if (string.IsEmpty)` → empty_check
     - `if (x > 100)` → range_check
     - `if (Regex.Match)` → regex_check
   - Returns inline validation definitions

### **specification_builder.py**

**SpecificationBuilder Class:**
Builds complete OpenAPI 3.0 specification from parsed components.

**Output Format:** OpenAPI 3.0 compliant JSON

**Key Methods:**

1. **build(parsed_data: Dict) -> Dict**
   - Main specification builder
   - Returns complete OpenAPI spec:
```python
{
    'openapi': '3.0.0',
    'info': {...},
    'servers': [...],
    'paths': {...},
    'components': {...},
    'security': [...],
    'tags': [...],
    'x-test-metadata': {...}
}
```

2. **build_info(parsed_data: Dict) -> Dict**
   - Creates API info section
   - Returns: `title`, `version`, `description`

3. **build_servers(parsed_data: Dict) -> List**
   - Defines API servers
   - Returns development and production server URLs

4. **build_paths(parsed_data: Dict) -> Dict**
   - Builds paths section from endpoints
   - Groups by path and HTTP method
   - Calls `build_operation()` for each endpoint

5. **build_operation(endpoint: Dict) -> Dict**
   - Creates operation object:
     - `summary`, `description`, `operationId`
     - `parameters` (via `build_parameters()`)
     - `requestBody` (for POST/PUT/PATCH)
     - `responses` (via `build_responses()`)
     - `security` (if auth required)
     - `tags` (for categorization)

6. **build_parameters(endpoint: Dict) -> List**
   - Converts parameters to OpenAPI format
   - Filters out body parameters (handled in requestBody)
   - Each parameter:
```python
{
    'name': 'id',
    'in': 'path|query|header',
    'required': True/False,
    'description': '...',
    'schema': {'type': '...', 'constraints': {...}}
}
```

7. **build_schema(param: Dict) -> Dict**
   - Creates JSON Schema for parameter
   - Includes: `type`, `minLength`, `maxLength`, `minimum`, `maximum`, `pattern`, `enum`, `format`, `default`

8. **build_request_body(body_params: List) -> Dict**
   - Defines request body for POST/PUT/PATCH
   - Returns:
```python
{
    'required': True/False,
    'content': {
        'application/json': {
            'schema': {...}
        }
    }
}
```

9. **build_responses(endpoint: Dict) -> Dict**
   - Defines expected responses
   - Default responses: 200, 400, 401, 404, 500

10. **build_components(parsed_data: Dict) -> Dict**
    - Builds reusable components:
      - `schemas`: Data models
      - `securitySchemes`: Auth definitions (bearerAuth, apiKey)

11. **build_model_schema(model: Dict) -> Dict**
    - Creates JSON Schema from model definition
    - Lists properties and required fields

12. **extract_business_logic(parsed_data: Dict) -> Dict**
    - Extracts business logic patterns:
      - `validations`: Validation rules
      - `exceptions`: Exception handling
      - `workflows`: Business workflows
      - `dependencies`: Service dependencies

13. **extract_exceptions(method_body: str) -> List**
    - Finds exception types in code
    - Patterns: `throw new XException`, `raise XError`

### **__init__.py (Input Processing)**

**InputProcessor Class:**
Main façade for input processing module.

**Initialization:**
```python
def __init__(self):
    self.parser_factory = ParserFactory()
    self.endpoint_extractor = EndpointExtractor()
    self.specification_builder = SpecificationBuilder()
    self.validator_extractor = ValidatorExtractor()
```

**Public Methods:**
- `parse_code(code_files, language)`: Parses code files → returns parsed data
- `build_specification(parsed_data)`: Builds OpenAPI spec → returns specification
- `extract_business_logic(parsed_data)`: Extracts business logic → returns logic patterns
- `extract_validation_rules(parsed_data)`: Extracts validations → returns validator list

**Usage Flow:**
```python
processor = InputProcessor()
parsed = processor.parse_code(['api.cs'], 'csharp')
spec = processor.build_specification(parsed)
validations = processor.extract_validation_rules(parsed)
```

---

## RAG System

### Location: `rag/`

Implements Retrieval-Augmented Generation using FAISS vector database for contextual test generation.

### **Architecture:**

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐
│  Chunking   │───→│  Embeddings  │
│  Strategy   │    │   Manager    │
└─────────────┘    └──────┬───────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Indexer   │
                   └──────┬──────┘
                          │
                          ▼
┌─────────────┐    ┌─────────────┐
│  Knowledge  │←──→│   Vector    │
│    Base     │    │   Store     │
└─────────────┘    │   (FAISS)   │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  Retriever  │
                   └─────────────┘
```

### **vector_store.py**

**VectorStore Class:**
FAISS-based vector database for similarity search.

**Supported Index Types:**
- **IVF (Inverted File)**: Accuracy-focused, requires training
- **HNSW (Hierarchical Navigable Small World)**: Fast search
- **Flat**: Simple brute-force (exact but slow)

**Initialization:**
```python
def __init__(self):
    self.dimension = 768  # Embedding dimension
    self.indices = {}  # Index name → FAISS index
    self.metadata_stores = {}  # Index name → {id → metadata}
    self.index_configs = {}  # Index name → config
```

**Creates Indices:**
- `test_patterns`, `edge_cases`, `validation_rules`
- `api_specifications`, `bug_patterns`, `successful_tests`

**Key Methods:**

1. **_create_index(index_name: str)**
   - Creates FAISS index based on config
   - IVF: `IndexIVFFlat` with quantizer
   - HNSW: `IndexHNSWFlat` with M=32
   - Flat: `IndexFlatL2`
   - Wraps with `IndexIDMap` for ID tracking

2. **add(index_name, embeddings, metadata, ids)**
   - Trains index if needed (IVF)
   - Adds embeddings with IDs
   - Stores metadata separately
   - Usage:
```python
embeddings = np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])
metadata = [{'text': 'test 1'}, {'text': 'test 2'}]
store.add('test_patterns', embeddings, metadata)
```

3. **search(index_name, query_embedding, k=10)**
   - Sets IVF nprobe for search
   - Searches for k nearest neighbors
   - Returns: (ids, distances, metadata)
   - Distance → similarity score: `1 / (1 + distance)`

4. **search_multiple_indices(query_embedding, indices, k)**
   - Searches across multiple indices
   - Returns results per index
   - Used for hybrid retrieval

5. **save_index(index_name) / load_index(index_name)**
   - Persists to disk:
     - `index.faiss`: FAISS index
     - `metadata.pkl`: Metadata store
     - `config.json`: Index configuration
   - Loads from disk on initialization

6. **get_index_stats(index_name) -> Dict**
   - Returns:
     - `total_embeddings`: Count
     - `dimension`: Embedding size
     - `index_type`: FAISS index type
     - `metadata_count`: Metadata entries

**Index Management:**
- `clear_index(index_name)`: Resets specific index
- `clear_all()`: Resets all indices
- `save_all()`: Persists all indices
- `get_all_stats()`: Stats for all indices

### **embeddings.py**

**EmbeddingManager Class:**
Generates embeddings for text and code using transformer models.

**Models:**
- **Text**: `sentence-transformers/all-mini-lm-l6-v2` (384-768 dim)
- **Code**: `microsoft/codebert-base` (768 dim)

**Initialization:**
```python
def __init__(self):
    self.text_model = SentenceTransformer('all-mini-lm-l6-v2')
    self.code_model = AutoModel.from_pretrained('microsoft/codebert-base')
    self.cache_dir = paths.VECTOR_STORE_DIR / "embedding_cache"
    self.cache = {}  # In-memory cache
```

**Key Methods:**

1. **embed_text(text: str, use_cache=True) -> np.ndarray**
   - Generates text embedding
   - Checks cache (MD5 hash key)
   - Uses SentenceTransformer
   - Normalizes vector (L2 norm)
   - Saves to cache (pickle)
   - Returns: 768-dim numpy array

2. **embed_code(code: str, language=None) -> np.ndarray**
   - Generates code embedding
   - Uses CodeBERT if available
   - Preprocesses code (removes comments)
   - Tokenizes with AutoTokenizer
   - Extracts last hidden state mean
   - Fallback: `embed_text()` with preprocessing

3. **_embed_with_codebert(code: str) -> np.ndarray**
   - Tokenizes code (max 512 tokens)
   - Forward pass through CodeBERT
   - Mean pooling of hidden states
   - Returns normalized embedding

4. **embed_structured(data: Dict) -> np.ndarray**
   - Converts structured data to text
   - Formats: `"Path: /api/users | Method: GET | Parameters: ..."`
   - Embeds combined text
   - Useful for API specs

5. **embed_batch(texts: List[str]) -> np.ndarray**
   - Batch embedding for efficiency
   - Processes in batches of 32
   - Normalizes each embedding
   - Returns stacked array

6. **combine_embeddings(embeddings: List, weights=None) -> np.ndarray**
   - Weighted average of multiple embeddings
   - Default: equal weights
   - Normalizes result
   - Use case: Multi-field embeddings

**Preprocessing:**
- `_preprocess_code(code, language)`: Removes comments, normalizes whitespace
- `_remove_python_comments(code)`: Removes `#` and `"""` docstrings
- `_remove_c_style_comments(code)`: Removes `//` and `/* */`

**Caching:**
- MD5 hash of text as cache key
- Saves as pickle: `{hash}.pkl`
- `clear_cache()`: Removes all cached embeddings

### **chunking.py**

**ChunkingStrategy Class:**
Splits documents into chunks for efficient embedding and retrieval.

**Chunk Dataclass:**
```python
@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int
    chunk_id: str
```

**Strategies:**
1. **Sliding Window** (default)
2. **Semantic** (paragraph-based)
3. **Code** (function/class-based)
4. **Test** (test case-based)

**Configuration:**
```python
chunk_size: int = 512  # Characters per chunk
chunk_overlap: int = 50  # Overlap for context preservation
```

**Key Methods:**

1. **chunk_document(document: str, metadata, strategy) -> List[Chunk]**
   - Routes to appropriate strategy
   - Returns list of `Chunk` objects

2. **sliding_window_chunk(document: str, metadata) -> List[Chunk]**
   - Splits into sentences
   - Accumulates until chunk_size reached
   - Adds overlap from previous chunk
   - Preserves sentence boundaries
   - Best for: General text, documentation

3. **semantic_chunk(document: str, metadata) -> List[Chunk]**
   - Splits by paragraphs (`\n\n`)
   - Respects semantic boundaries
   - Groups paragraphs until chunk_size
   - Best for: Documentation, articles

4. **code_chunk(document: str, metadata) -> List[Chunk]**
   - Detects programming language
   - Splits by function/class definitions
   - Patterns:
     - Python: `def`, `class`, `async def`
     - C#/Java: Method boundaries with braces
   - Fallback: Line-based chunking (50 lines)
   - Metadata: `block_type` (class, function, test)
   - Best for: Source code

5. **test_chunk(document: str, metadata) -> List[Chunk]**
   - Identifies test function patterns:
     - Python: `def test_*`
     - Java/C#: `*Test` methods
     - JavaScript: `it('...')`
     - C++: `TEST(...)`
   - Splits at test boundaries
   - Metadata: `test_name`
   - Best for: Test files

**Helper Methods:**
- `_split_sentences(text)`: Regex-based sentence splitting
- `_split_code_blocks(code, language)`: Language-specific code splitting
- `_split_by_lines(text, lines_per_chunk)`: Simple line-based chunking
- `_detect_block_type(code_block)`: Identifies code block type
- `_extract_test_name(test_block)`: Extracts test function name

**Example:**
```python
chunker = ChunkingStrategy(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_document(
    document=code_text,
    metadata={'file': 'api.py', 'language': 'python'},
    strategy='code'
)
```

### **indexer.py**

**Indexer Class:**
Manages indexing of documents into vector store.

**Dependencies:**
- `VectorStore`: For storing embeddings
- `EmbeddingManager`: For generating embeddings
- `ChunkingStrategy`: For splitting documents

**Initialization:**
```python
def __init__(self, vector_store, embedding_manager, chunking_strategy):
    self.vector_store = vector_store
    self.embedding_manager = embedding_manager
    self.chunking_strategy = chunking_strategy
    self.indexed_documents = set()  # Track indexed doc IDs
    self.index_metadata_file = paths.VECTOR_STORE_DIR / "indexed_docs.json"
```

**Key Methods:**

1. **index_document(document: Dict, index_name=None) -> Dict**
   - Checks if already indexed (skip duplicates)
   - Determines appropriate index
   - Extracts content from document
   - Chunks document
   - Generates embeddings for chunks
   - Adds to vector store with metadata
   - Marks as indexed
   - Returns: `{'status': 'indexed', 'chunks': count}`

2. **index_documents(documents: List, index_name=None) -> Dict**
   - Batch indexes multiple documents
   - Returns statistics:
```python
{
    'total_documents': int,
    'indexed': int,
    'skipped': int,
    'failed': int,
    'chunks_created': int
}
```

3. **index_test_cases(test_cases: List)**
   - Specialized indexing for test cases
   - Formats test case as document
   - Metadata includes: test_type, endpoint, method
   - Indexes into `test_patterns`

4. **index_api_specifications(api_specs: List)**
   - Indexes OpenAPI specs
   - Creates document per endpoint
   - Metadata: path, method, operationId, tags
   - Indexes into `api_specifications`

**Helper Methods:**

- `_determine_index(document: Dict) -> str**
  - Routes document to appropriate index
  - Based on document type:
    - `test_case` → `test_patterns`
    - `edge_case` → `edge_cases`
    - `validation` → `validation_rules`
    - `api_specification` → `api_specifications`
    - `bug` → `bug_patterns`

- `_extract_content(document: Dict) -> str**
  - Extracts textual content
  - Checks fields: `content`, `text`, `code`
  - Constructs from: `name`, `description`, `summary`

- `_format_test_case(test_case: Dict) -> str**
  - Formats for indexing:
```
Test: {name}
Type: {test_type}
Endpoint: {endpoint}
Method: {method}
Description: {description}
Steps: ...
Assertions: ...
Test Data: ...
```

- `_format_api_operation(path, method, operation) -> str**
  - Formats OpenAPI operation:
```
Endpoint: GET /api/users
Summary: ...
Parameters: ...
Request Body: ...
Responses: ...
```

**Persistence:**
- `_load_indexed_documents()`: Loads indexed doc IDs from JSON
- `_save_indexed_documents()`: Saves indexed doc IDs to JSON
- Prevents duplicate indexing across sessions

**Update Operations:**
- `update_index(document, index_name)`: Forces reindex with `force_reindex` flag
- `clear_index(index_name)`: Clears index and metadata

### **retriever.py**

**Retriever Class:**
Handles retrieval of relevant documents from vector store.

**Features:**
- Vector similarity search
- Re-ranking with cross-encoder
- MMR (Maximal Marginal Relevance) for diversity
- Multi-index hybrid search

**Initialization:**
```python
def __init__(self, vector_store, embedding_manager):
    self.vector_store = vector_store
    self.embedding_manager = embedding_manager
    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
```

**Key Methods:**

1. **retrieve(query: Union[str, np.ndarray], index_name, k=10, rerank=None) -> List[Dict]**
   - Main retrieval method
   - Generates embedding if query is text
   - Searches vector store
   - Optionally reranks results
   - Returns top-k results:
```python
[{
    'id': int,
    'score': float,  # Similarity score (0-1)
    'metadata': {...},
    'rank': int
}, ...]
```

2. **retrieve_similar_tests(query_embedding, k=10) -> List**
   - Searches `test_patterns` index
   - Enhances with test classification
   - Returns similar test cases

3. **retrieve_edge_cases(query_embedding, k=5) -> List**
   - Searches `edge_cases` index
   - Filters by similarity threshold
   - Returns relevant edge cases

4. **retrieve_validation_patterns(query_embedding, k=5) -> List**
   - Searches `validation_rules` index
   - Returns validation patterns
   - Used for validation test generation

5. **hybrid_search(query: str, indices=None, k=10) -> List**
   - Searches across multiple indices
   - Combines results from all indices
   - Sorts by score
   - Applies reranking
   - Applies MMR for diversity
   - Returns unified result list

**Reranking:**

**_rerank_results(query: str, results: List) -> List**
- Uses cross-encoder model
- Creates query-document pairs
- Scores each pair (0-1)
- Combines with vector similarity: `(vector_score + rerank_score) / 2`
- Re-sorts by final score
- More accurate but slower than vector search

**Diversity (MMR):**

**_apply_mmr(query_embedding, results, k, lambda_param=0.7) -> List**
- Maximal Marginal Relevance algorithm
- Balances relevance and diversity
- Formula: `MMR = λ * relevance - (1-λ) * max_similarity_to_selected`
- Selects first result (highest relevance)
- Iteratively selects results that are relevant but dissimilar to already selected
- Prevents redundant results

**Classification:**

**_classify_test_type(test_code: str) -> str**
- Heuristic classification based on keywords
- Categories: `edge_case`, `null_check`, `happy_path`, `error_handling`, `security`, `general`

### **knowledge_base.py**

**KnowledgeBase Class:**
Manages structured knowledge for API testing.

**Knowledge Types:**
- `test_patterns`: Common test patterns and strategies
- `edge_cases`: Edge cases and boundary conditions
- `validation_rules`: Validation patterns
- `api_patterns`: Common API design patterns
- `bug_patterns`: Common bugs and issues
- `best_practices`: Testing best practices

**Initialization:**
```python
def __init__(self):
    self.knowledge_dir = paths.DATA_DIR / "knowledge_base"
    self.knowledge = {}  # Type → List[items]
    self._initialize_knowledge_base()
```

**Default Test Patterns:**
```python
{
    'name': 'Happy Path Testing',
    'description': 'Test with valid inputs',
    'applicable_to': ['GET', 'POST', 'PUT', 'DELETE'],
    'test_data': {
        'strategy': 'Use realistic, valid data',
        'examples': ['Valid IDs', 'Complete objects']
    }
}
```

**Default Edge Cases:**
```python
{
    'type': 'string',
    'cases': [
        {'value': '', 'description': 'Empty string'},
        {'value': ' ', 'description': 'Single space'},
        {'value': 'A' * 10000, 'description': 'Very long string'},
        {'value': '<script>alert("xss")</script>', 'description': 'XSS attempt'},
        {'value': "'; DROP TABLE users; --", 'description': 'SQL injection'}
    ]
}
```

**Key Methods:**

1. **add_knowledge(knowledge_type: str, item: Dict)**
   - Adds new knowledge item
   - Adds metadata: `added_at`, `id` (hash)
   - Saves to disk

2. **get_knowledge(knowledge_type: str, filters=None) -> List**
   - Retrieves knowledge items
   - Optionally filters by criteria
   - Example: `get_knowledge('test_patterns', {'applicable_to': 'POST'})`

3. **update_knowledge(knowledge_type, item_id, updates)**
   - Updates existing item
   - Adds `updated_at` timestamp
   - Saves to disk

4. **remove_knowledge(knowledge_type, item_id)**
   - Removes item by ID
   - Saves to disk

5. **search_knowledge(query: str, knowledge_types=None) -> List**
   - Full-text search across knowledge base
   - Searches in JSON string representation
   - Returns matching items with type

6. **get_test_pattern_for_endpoint(method: str, endpoint_type=None) -> List**
   - Returns relevant test patterns for HTTP method
   - Filters by `applicable_to` field

7. **get_edge_cases_for_type(data_type: str) -> List**
   - Returns edge cases for specific data type
   - Types: `string`, `integer`, `array`, `date`

8. **add_test_result(test_case, result)**
   - Learns from execution results
   - Successful tests → `test_patterns`
   - Failed tests → `bug_patterns`

**Persistence:**
- Each knowledge type saved as JSON file
- Location: `data/knowledge_base/{type}.json`
- `export_knowledge(output_path)`: Exports all knowledge
- `import_knowledge(import_path, merge=True)`: Imports knowledge

**Statistics:**
- `get_statistics()`: Returns counts per type, last updated

---

## LLM Orchestration Layer

### Location: `llm/`

Multi-agent system using Llama 3.2 via LM Studio for specialized testing tasks.

### **Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│                   Agent Manager                          │
│              (Coordinates all agents)                     │
└───────┬──────────────────────────────────────────────────┘
        │
        ├──→ Analyzer Agent (API analysis)
        ├──→ Test Designer Agent (Test case design)
        ├──→ Edge Case Agent (Edge case generation)
        ├──→ Data Generator Agent (Test data creation)
        └──→ Report Writer Agent (Report generation)
```

### **llama_client.py**

**LlamaClient Class:**
Client for interacting with Llama 3.2 via LM Studio.

**Connection:**
- Base URL: `http://127.0.0.1:1234/v1` (LM Studio local server)
- Compatible with OpenAI API format
- No API key required

**Key Methods:**

1. **generate(prompt: str, **kwargs) -> str**
   - Sends completion request
   - Endpoint: `/v1/completions`
   - Uses async/await with aiohttp
   - Retry logic with exponential backoff (3 attempts)
   - Returns generated text

2. **chat(messages: List[Dict], **kwargs) -> str**
   - Chat-style completion
   - Formats messages as prompt:
```python
System: {system_message}
User: {user_message}
Assistant: {assistant_message}
Assistant:
```
   - Returns assistant response

3. **generate_json(prompt: str, schema=None, **kwargs) -> Dict**
   - Generates structured JSON response
   - Adds instruction: "Respond with valid JSON only"
   - Strips markdown code blocks
   - Parses JSON
   - Validates against schema if provided
   - Returns parsed JSON

4. **stream_generate(prompt: str, callback=None, **kwargs)**
   - Streaming generation for real-time output
   - Yields tokens as they're generated
   - Calls async callback for each token

5. **get_embeddings(text: str) -> List[float]**
   - Attempts to get embeddings from LM Studio
   - Endpoint: `/v1/embeddings`
   - Returns embedding vector if supported
   - Warning if not available

6. **get_config_for_agent(agent_type: str) -> Dict**
   - Returns agent-specific configuration
   - Different temperatures per agent

**Context Manager:**
```python
async with LlamaClient() as client:
    response = await client.generate("prompt")
```

### **prompt_templates.py**

**PromptTemplates Class:**
Collection of reusable prompt templates.

**Templates:**

1. **API_ANALYSIS**
   - Analyzes API endpoint for testing requirements
   - Identifies: critical scenarios, edge cases, security vulnerabilities, performance considerations, validation requirements

2. **HAPPY_PATH_TEST**
   - Generates happy path test case
   - Inputs: endpoint, method, parameters
   - Creates test verifying successful operation

3. **EDGE_CASE_TEST**
   - Generates edge case tests
   - Inputs: parameter name, type, constraints
   - Includes: boundary values, null/empty cases, extreme values

4. **SECURITY_TEST**
   - Generates security test cases
   - Tests: SQL injection, XSS, auth bypass, authorization elevation

5. **VALID_DATA / INVALID_DATA**
   - Generates valid/invalid test data
   - Ensures constraints are satisfied/violated

6. **TEST_SUMMARY**
   - Summarizes test execution results
   - Inputs: total, passed, failed
   - Provides insights and recommendations

7. **FAILURE_ANALYSIS**
   - Analyzes test failure
   - Identifies root cause
   - Suggests fixes

8. **CONTEXT_INTEGRATION**
   - Integrates RAG context
   - Adapts patterns from similar tests

9. **VALIDATION_RULES**
   - Extracts validation rules from code
   - Identifies: required fields, type constraints, format requirements, business rules

**Usage:**
```python
template = PromptTemplates.get_template('api_analysis')
prompt = template.format(api_specification=spec)
```

### **prompt_builder.py**

**PromptBuilder Class:**
Dynamically constructs prompts for different scenarios.

**Key Methods:**

1. **build_analysis_prompt(api_spec, context=None) -> str**
   - Formats API specification
   - Includes RAG context
   - Requests structured JSON response
   - Sections: parameters, business logic, security, edge cases, performance

2. **build_test_generation_prompt(test_type, api_spec, context=None, examples=None) -> str**
   - Generates test case prompt
   - Includes test-type specific instructions
   - Provides test case schema
   - Examples for reference

3. **build_data_generation_prompt(parameters, test_type='valid', constraints=None) -> str**
   - Generates test data prompt
   - Valid: realistic data satisfying constraints
   - Invalid: data violating constraints
   - Edge: boundary values and extreme cases

4. **build_report_prompt(results, report_type='summary') -> str**
   - Report generation prompt
   - Types: `summary`, `detailed`, `recommendations`
   - Includes execution statistics

**Helper Methods:**

- `_format_api_spec(api_spec)`: Formats spec for prompt
- `_format_context(context)`: Formats RAG context
- `_format_examples(examples)`: Formats example test cases
- `_get_test_type_instructions(test_type)`: Returns test-type specific instructions
- `_get_test_case_schema(test_type)`: Returns JSON schema for test case

**Test Type Instructions:**
```python
'happy_path': """
- Cover basic successful operations
- Use valid, realistic data
- Verify expected responses
- Test different valid parameter combinations
"""

'security': """
- Test SQL injection
- Check XSS vulnerabilities
- Test command injection
- Verify path traversal protection
- Test authentication bypass
"""
```

### **response_parser.py**

**ResponseParser Class:**
Parses and validates LLM responses.

**Parsers:**
- `json`: Structured JSON
- `list`: List of items
- `code`: Code blocks
- `text`: Plain text
- `structured`: Structured text with sections

**Key Methods:**

1. **parse(response: str, expected_format='json') -> Any**
   - Routes to appropriate parser
   - Returns parsed response

2. **parse_json(response: str) -> Union[Dict, List]**
   - Cleans JSON response (removes markdown)
   - Attempts JSON parsing
   - Extracts JSON from text if direct parsing fails
   - Fallback to structured parsing

3. **parse_list(response: str) -> List[str]**
   - Tries JSON parsing first
   - Falls back to text list parsing
   - Removes list markers (-, *, numbers)

4. **parse_code(response: str) -> Dict**
   - Extracts code blocks with triple backticks
   - Returns: `language`, `code`, `description`
   - Handles inline code with single backticks

5. **parse_structured(response: str) -> Dict**
   - Parses text with headers
   - Splits by numbered lists, headers, bold text
   - Groups content by section
   - Returns: `{section_name: content}`

**Validators:**

- `validate_test_case(test_case)`: Checks required fields (name, endpoint, method)
- `validate_analysis(analysis)`: Checks required fields (endpoint, method, critical_parameters)

**Cleaning:**

- `_clean_json_response(response)`: Removes markdown, finds JSON boundaries
- `_extract_description(full_response, code)`: Extracts description from code response

### **agents/base_agent.py**

**BaseAgent (Abstract Class):**
Base class for all LLM agents.

**Initialization:**
```python
def __init__(self, llama_client, agent_type: str = None):
    self.client = llama_client
    self.agent_type = agent_type
    self.config = self._get_config()  # Agent-specific config
```

**Abstract Method:**
- `execute(input_data: Dict) -> Any`: Must be implemented by subclasses

**Retry Logic:**

- `generate_with_retry(prompt, max_retries=3)`: Retries on failure
- `generate_json_with_retry(prompt, schema=None, max_retries=3)`: JSON with retry

**Helper Methods:**

- `format_context(context: Dict) -> str`: Formats RAG context for prompt
- `validate_response(response) -> bool`: Basic validation

### **agents/analyzer_agent.py**

**AnalyzerAgent Class:**
Analyzes API specifications and identifies testing requirements.

**Extends:** BaseAgent

**Purpose:** First agent in pipeline - analyzes API to guide test generation

**Key Method:**

**analyze(api_spec: Dict, context: Dict) -> Dict**

Input: API specification, RAG context
Output: Comprehensive analysis

```python
{
    'endpoint': '/api/users/{id}',
    'method': 'GET',
    'critical_parameters': ['id', 'includeDeleted'],
    'auth_requirements': {
        'required': True,
        'type': 'bearer',
        'scopes': ['users:read']
    },
    'business_logic': ['User must exist', 'Soft delete handling'],
    'failure_points': ['Invalid ID', 'Unauthorized access'],
    'dependencies': ['UserService', 'AuthService'],
    'validation_rules': ['ID must be positive integer'],
    'error_scenarios': ['404 if not found', '401 if unauthorized'],
    'performance': {
        'expected_latency': '<100ms',
        'throughput': '1000 req/s'
    }
}
```

**Prompt Construction:**
- Formats API specification with parameters, request body, responses
- Includes RAG context (similar tests, edge cases, validation patterns)
- Requests structured JSON analysis

**Analysis Schema:**
Provides JSON schema for consistent output structure

**Risk Assessment:**
- `_assess_risks(api_spec)`: Categorizes risks (high, medium, low)
  - High: No auth, DELETE/PUT operations, file uploads
  - Medium: No validation rules
  - Low: Other considerations

**Complexity Assessment:**
- `_assess_complexity(api_spec)`: Returns complexity level
  - Score factors: parameter count, nested structures, response codes, dependencies
  - Levels: low (<5), medium (5-15), high (>15)

### **agents/test_designer.py**

**TestDesignerAgent Class:**
Designs comprehensive test cases based on analysis.

**Extends:** BaseAgent

**Purpose:** Generates various test types based on analyzer output

**Key Method:**

**design_tests(analysis: Dict, context: Dict, config: Dict) -> List[Dict]**

Generates test cases for multiple categories:
- Happy path
- Validation
- Authentication
- Error handling
- Boundary
- Performance

**Test Types:**

1. **_generate_happy_path_tests(analysis, context)**
   - 3-5 basic successful operations
   - Valid data with different combinations
   - Expected successful responses
   - Uses RAG context for similar successful tests

2. **_generate_validation_tests(analysis, context)**
   - Tests each validation rule
   - Required field validation
   - Data type validation
   - Format validation (email, phone)
   - Length/size constraints
   - Pattern matching

3. **_generate_auth_tests(analysis, context)**
   - No authentication
   - Invalid token/credentials
   - Expired token
   - Insufficient permissions/scopes
   - Valid authentication

4. **_generate_error_tests(analysis, context)**
   - 400 Bad Request scenarios
   - 404 Not Found scenarios
   - 409 Conflict scenarios
   - 500 Internal Server Error scenarios

5. **_generate_boundary_tests(analysis, context)**
   - Minimum valid values
   - Maximum valid values
   - Just below minimum (invalid)
   - Just above maximum (invalid)
   - Edge cases (0, -1, empty, null)

6. **_generate_performance_tests(analysis, context)**
   - Response time validation
   - Concurrent request handling
   - Large payload handling
   - Rate limiting validation

**Test Case Structure:**
```python
{
    'name': 'Descriptive test name',
    'description': 'What this test validates',
    'test_type': 'happy_path|validation|...',
    'endpoint': '/api/users',
    'method': 'GET',
    'input': {'parameter': 'value'},
    'expected_status': 200,
    'expected_response': {'key': 'expected value'},
    'assertions': ['List of assertions to verify']
}
```

**Prioritization:**

**_prioritize_tests(test_cases, max_tests) -> List**
- Scores each test based on type and importance
- Priority order: authentication > validation > happy_path > error_handling > boundary > performance
- Bonus for specific assertions
- Returns top N tests

### **agents/edge_case_agent.py**

**EdgeCaseAgent Class:**
Generates creative edge case and security test scenarios.

**Extends:** BaseAgent

**Purpose:** High-temperature agent for creative edge case discovery

**Key Method:**

**generate_edge_cases(api_spec: Dict, analysis: Dict) -> List[Dict]**

Generates 4 categories of edge cases:
1. Parameter edge cases
2. Combination edge cases
3. Security edge cases
4. State-based edge cases

**1. Parameter Edge Cases:**

**_generate_parameter_edge_cases(param, api_spec)**

For each parameter, generates:
- **Boundary values**: Min, max, just outside bounds
- **Type mismatches**: Wrong data types
- **Special characters**: Encoding issues
- **Null/empty/undefined**: Missing values
- **Extreme values**: Very large, very small, infinity, NaN
- **Format violations**: Invalid formats
- **Injection attempts**: SQL, XSS, command injection

Example:
```python
{
    'name': 'SQL Injection via username parameter',
    'description': 'Tests SQL injection protection',
    'parameter': 'username',
    'value': "admin' OR '1'='1",
    'expected_behavior': 'Request rejected or sanitized',
    'risk_level': 'high'
}
```

**2. Combination Edge Cases:**

**_generate_combination_edge_cases(parameters, api_spec)**

Tests parameter interactions:
- All parameters at minimum values
- All parameters at maximum values
- Mix of min and max values
- One valid, others invalid
- Conflicting values violating business logic

**3. Security Edge Cases:**

**_generate_security_edge_cases(api_spec, analysis)**

Security-focused tests:
- SQL Injection attempts
- XSS (Cross-Site Scripting) attempts
- Command injection
- Path traversal attacks (`../../etc/passwd`)
- Authentication bypass attempts
- Authorization elevation attempts
- CSRF attacks
- XXE injection (XML endpoints)
- Buffer overflow attempts
- Rate limiting bypass

**4. State-Based Edge Cases:**

**_generate_state_edge_cases(api_spec, analysis)**

Tests state transitions:
- Resource doesn't exist
- Resource already exists (for creation)
- Concurrent modifications
- Stale data scenarios
- Transaction rollback scenarios
- Partial state updates
- Race conditions

**Risk Levels:**
- `high`: Security vulnerabilities
- `medium`: Data integrity issues
- `low`: Minor edge cases

### **agents/data_generator.py**

**DataGeneratorAgent Class:**
Generates realistic test data for test cases.

**Extends:** BaseAgent

**Purpose:** Creates valid, invalid, and edge case test data

**Faker Patterns:**
```python
faker_patterns = {
    'email': lambda: f"test_{random.randint(1000, 9999)}@example.com",
    'phone': lambda: f"+1{random.randint(1000000000, 9999999999)}",
    'uuid': lambda: "{uuid format}",
    'date': lambda: (datetime.now() + timedelta(days=random.randint(-365, 365))).isoformat(),
    'url': lambda: f"https://example.com/{random.choice(['api', 'test'])}/{random.randint(1, 100)}",
    'ip': lambda: f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
}
```

**Key Method:**

**generate_data(test_cases: List, api_spec: Dict) -> Dict**

Generates data for each test case based on test type:
- `happy_path` → valid data
- `validation` → invalid data
- `boundary` → boundary values
- `edge_case` → edge values

**Data Generation Methods:**

1. **_generate_valid_data(test_case, api_spec)**
   - Realistic, valid data satisfying all constraints
   - Uses LLM for complex generation
   - Applies faker patterns for common fields
   - Returns: `{param_name: value}`

2. **_generate_validation_data(test_case, api_spec)**
   - Invalid data based on test description
   - Missing required fields
   - Wrong data types
   - Invalid lengths/sizes
   - Invalid formats
   - Returns: `{param_name: invalid_value}`

3. **_generate_boundary_data(test_case, api_spec)**
   - Boundary values from constraints
   - Min values, max values
   - Edge strings (min length, max length)
   - Returns: `{param_name: boundary_value}`

4. **_generate_edge_data(test_case, api_spec)**
   - Edge case values:
     - Integer: 0, -1, MAX_INT, MIN_INT
     - Number: 0.0, -0.0, infinity, -infinity
     - String: '', ' ', '\n', '<script>', SQL injection
     - Boolean: null
     - Array: [], [null], deeply nested
     - Object: {}, null, {'': ''}

**Value Generators:**

- `_generate_valid_value(param)`: Generates valid value based on type and constraints
- `_generate_wrong_type(correct_type)`: Generates wrong type (e.g., string for integer)
- `_generate_invalid_length(param)`: Generates string violating length constraints
- `_generate_invalid_format(param_type)`: Generates invalid format (e.g., "invalid-email")
- `_generate_boundary_value(constraints)`: Generates min or max value
- `_generate_edge_value(param_type)`: Generates edge case value

**Pattern Application:**

`_apply_faker_patterns(data)`: Applies faker patterns to fields based on name (email, phone, uuid, etc.)

### **agents/report_writer.py**

**ReportWriterAgent Class:**
Generates professional QASE-style test reports.

**Extends:** BaseAgent

**Purpose:** Final agent - produces comprehensive test reports

**Key Method:**

**generate_report(execution_results: List, session: Dict) -> Dict**

Generates complete test report:
- Summary statistics
- Individual test case reports
- Recommendations

**Report Structure:**
```python
{
    'title': 'API Test Report - {session_id}',
    'generated_at': '2024-01-01T00:00:00',
    'summary': {...},
    'test_cases': [...],
    'recommendations': [...],
    'metadata': {
        'session_id': '...',
        'api_endpoint': '...',
        'total_duration': 123.45
    }
}
```

**Summary Generation:**

`_generate_summary(execution_results) -> Dict`

```python
{
    'total_tests': 50,
    'passed': 45,
    'failed': 5,
    'pass_rate': 90.0,
    'by_test_type': {
        'happy_path': {'total': 10, 'passed': 10, 'failed': 0},
        'validation': {'total': 15, 'passed': 13, 'failed': 2},
        ...
    },
    'critical_failures': 2,
    'execution_time': 123.45
}
```

**Test Case Reports:**

`_generate_test_case_report(result) -> Dict`

QASE-style format:
```python
{
    'title': 'Test name',
    'status': 'PASSED|FAILED',
    'severity': 'critical|major|normal|minor',
    'priority': 'high|medium|low',
    'test_type': '...',
    'preconditions': [...],
    'steps': [
        {'action': '...', 'expected': '...'},
        ...
    ],
    'expected_result': '...',
    'actual_result': '...',
    'execution_time': 1.23,
    'error': '...' if failed,
    'failure_analysis': '...' if failed,
    'attachments': {
        'request': {...},
        'response': {...},
        'logs': [...]
    }
}
```

**Preconditions Generation:**
- API endpoint availability
- Valid authentication (if not auth test)
- Test data prepared
- Service dependencies available

**Steps Generation:**
1. Setup test environment
2. Prepare test data
3. Send HTTP request
4. Validate assertions
5. Clean up test data

**Failure Analysis:**

`_analyze_failure(result) -> str`
- Uses LLM to analyze failure
- Identifies root cause
- Determines severity
- Recommends action
- Concise explanation (<100 words)

**Recommendations Generation:**

`_generate_recommendations(execution_results) -> List[str]`
- Analyzes failure patterns
- Groups by test type
- Uses LLM to generate 3-5 actionable recommendations:
  - Critical issues needing immediate attention
  - Systematic problems
  - Areas needing additional coverage
  - Performance/security concerns

**Severity/Priority Determination:**

- **Severity**: Based on test type
  - `critical`: authentication, security tests
  - `major`: validation, error handling
  - `normal`: boundary, edge cases
  - `minor`: other

- **Priority**: Based on severity and failure status
  - Failed critical → `high`
  - Failed major → `medium`
  - Failed normal/minor → `low`
  - Passed → `low`

### **agent_manager.py**

**AgentManager Class:**
Coordinates multiple agents with dependency management.

**Purpose:** Orchestrates agent execution with proper task dependencies

**Agent Types Enum:**
```python
class AgentType(Enum):
    ANALYZER = "analyzer"
    TEST_DESIGNER = "test_designer"
    EDGE_CASE = "edge_case"
    DATA_GENERATOR = "data_generator"
    REPORT_WRITER = "report_writer"
```

**AgentTask Dataclass:**
```python
@dataclass
class AgentTask:
    agent_type: AgentType
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1
```

**Initialization:**
```python
def __init__(self):
    self.agents = {
        AgentType.ANALYZER: AnalyzerAgent(),
        AgentType.TEST_DESIGNER: TestDesignerAgent(),
        AgentType.EDGE_CASE: EdgeCaseAgent(),
        AgentType.DATA_GENERATOR: DataGeneratorAgent(),
        AgentType.REPORT_WRITER: ReportWriterAgent()
    }
    self.task_queue = asyncio.Queue()
    self.results = {}
    self.running_tasks = {}
```

**Key Method:**

**orchestrate(api_spec: Dict, context: Dict) -> Dict**

Main orchestration flow:

1. Create task workflow
2. Execute tasks with dependency resolution
3. Combine results

**Task Workflow:**
```python
[
    # Priority 1: Analyze API
    AgentTask(ANALYZER, {...}, dependencies=None, priority=1),
    
    # Priority 2: Parallel test generation
    AgentTask(TEST_DESIGNER, {...}, dependencies=['analyzer'], priority=2),
    AgentTask(EDGE_CASE, {...}, dependencies=['analyzer'], priority=2),
    
    # Priority 3: Generate test data
    AgentTask(DATA_GENERATOR, {...}, 
              dependencies=['test_designer', 'edge_case'], 
              priority=3)
]
```

**Dependency Resolution:**

`_wait_for_dependencies(dependencies: List[str])`
- Waits until all dependent tasks complete
- Checks `self.results` for completed tasks
- Async polling with sleep(0.1)

**Task Execution:**

`_execute_task(task: AgentTask) -> Any`
- Retrieves appropriate agent
- Injects dependency results into input data
- Executes agent
- Stores result in `self.results`

**Result Combination:**

`_combine_results(results: Dict) -> Dict`
```python
{
    'analysis': results.get('analyzer'),
    'test_cases': results.get('test_designer'),
    'edge_cases': results.get('edge_case'),
    'test_data': results.get('data_generator'),
    'metadata': {
        'total_tests': count,
        'agents_used': list(results.keys())
    }
}
```

**Single Agent Execution:**

`execute_single_agent(agent_type: AgentType, input_data: Dict) -> Any`
- Executes agent independently
- Bypasses orchestration
- Returns agent result directly

---

## Reinforcement Learning Optimizer

### Location: `reinforcement_learning/`

PPO (Proximal Policy Optimization) based system for intelligent test selection and prioritization.

### **Architecture:**

```
┌─────────────────────────────────────────────────┐
│          RL Optimizer (Main Coordinator)         │
└───────┬─────────────────────────────────────────┘
        │
        ├──→ Policy Network (Action Selection)
        ├──→ Value Network (State Evaluation)
        ├──→ Experience Buffer (Replay Memory)
        └──→ Reward Calculator (Reward Function)
```

### **policy_network.py**

**PolicyNetwork Class:**
Neural network for selecting test actions (test types/priorities).

**Architecture:**
```python
Input (state_dim) → Linear(256) → ReLU → Dropout(0.1)
                  → Linear(128) → ReLU → Dropout(0.1)
                  → Linear(64) → ReLU → Dropout(0.1)
                  → Linear(action_dim) → Softmax → Action Probabilities
```

**Key Methods:**

1. **forward(state: Tensor) -> Tensor**
   - Forward pass through network
   - Returns action probabilities (softmax)

2. **get_action(state: Tensor, deterministic=False) -> Tensor**
   - Selects action from policy
   - Deterministic: argmax (exploitation)
   - Stochastic: sample from distribution (exploration)
   - Returns action index

3. **evaluate_actions(states: Tensor, actions: Tensor) -> tuple**
   - For PPO training
   - Returns: (log_probabilities, entropy)
   - Used in policy gradient calculation

**Weight Initialization:**
- Xavier uniform initialization
- Small positive bias (0.01)

### **value_network.py**

**ValueNetwork Class:**
Neural network for evaluating state value (V-function).

**Architecture:**
```python
Input (state_dim) → Linear(256) → ReLU → Dropout(0.1)
                  → Linear(128) → ReLU → Dropout(0.1)
                  → Linear(64) → ReLU → Dropout(0.1)
                  → Linear(1) → State Value
```

**Key Methods:**

1. **forward(state: Tensor) -> Tensor**
   - Forward pass
   - Returns single value (state value)

2. **get_value(state: Tensor) -> float**
   - Gets value for single state
   - Returns float value

**Purpose:**
- Estimates expected return from state
- Used in advantage calculation (A = Q - V)
- Trained with TD(λ) error

### **experience_buffer.py**

**ExperienceBuffer Class:**
Replay buffer with prioritized experience replay.

**Experience Dataclass:**
```python
@dataclass
class Experience:
    state: Tensor
    action: Tensor
    reward: float
    next_state: Tensor
    done: bool
    priority: float = 1.0
```

**Initialization:**
```python
def __init__(self, capacity: int):
    self.buffer = deque(maxlen=capacity)  # Circular buffer
    self.priorities = deque(maxlen=capacity)
    self.alpha = 0.6  # Priority exponent
    self.beta = 0.4  # Importance sampling
```

**Key Methods:**

1. **add(state, action, reward, next_state, done)**
   - Adds experience to buffer
   - Assigns initial priority (max existing priority)
   - Automatic eviction when full (FIFO)

2. **sample(batch_size: int) -> List[Experience]**
   - Samples batch of experiences
   - Uniform or prioritized sampling

3. **_uniform_sample(batch_size: int)**
   - Random uniform sampling
   - Simple baseline

4. **_prioritized_sample(batch_size: int)**
   - Prioritized Experience Replay (PER)
   - Sampling probability: `P(i) = p_i^α / Σ p_j^α`
   - Higher priority → higher probability
   - Samples without replacement

5. **update_priorities(indices: List[int], td_errors)**
   - Updates priorities after training
   - Priority = `|TD-error| + ε`
   - Higher error → higher priority
   - Focuses learning on surprising transitions

**Statistics:**
- `total_added`: Total experiences added
- `total_sampled`: Total samples drawn

### **reward_calculator.py**

**RewardCalculator Class:**
Calculates rewards based on test execution outcomes.

**Reward Weights (from config):**
```python
{
    'bug_found': 10.0,           # Major reward for finding bugs
    'code_coverage': 5.0,        # Coverage improvement
    'edge_case_covered': 8.0,    # Edge case discovery
    'unique_scenario': 6.0,      # Novel test scenarios
    'false_positive': -3.0,      # Penalty for false positives
    'redundant_test': -2.0,      # Penalty for redundancy
    'test_failed': -1.0,         # Minor penalty for test failure
    'api_error': -5.0'           # Penalty for API errors
}
```

**Key Methods:**

1. **calculate_reward(test_results: List, metrics: Dict) -> float**
   - Main reward calculation
   - Aggregates rewards from metrics
   - Applies weights
   - Normalizes result
   - Stores in history

2. **calculate_intermediate_reward(state, action) -> float**
   - Reward before execution (shaping)
   - Based on test diversity (edge_case=2.0, validation=1.5, happy_path=1.0)
   - Parameter coverage bonus (×3.0)
   - Redundancy penalty (-2.0)

3. **_count_unique_scenarios(test_results) -> int**
   - Creates signature: (endpoint, method, test_type, parameters)
   - Counts unique combinations

4. **_count_redundant_tests(test_results) -> int**
   - Creates test signature
   - Counts duplicates

5. **_normalize_reward(reward: float) -> float**
   - Clips to [-100, 100]
   - Applies sigmoid: `2 / (1 + exp(-r/10)) - 1`
   - Smooths extreme values

**Statistics:**

`get_reward_statistics() -> Dict`
```python
{
    'mean_reward': float,
    'std_reward': float,
    'max_reward': float,
    'min_reward': float,
    'total_episodes': int,
    'recent_trend': 'improving|declining|stable'
}
```

**Trend Calculation:**
- Linear regression on recent rewards
- Slope > 0.1 → improving
- Slope < -0.1 → declining
- Otherwise → stable

### **rl_optimizer.py**

**RLOptimizer Class:**
Main PPO-based reinforcement learning optimizer.

**State Representation:**

Concatenates multiple feature vectors:
- **API complexity** (128 features): parameter count, types, method, auth, responses
- **Test coverage** (64 features): test type distribution, total tests, assertions
- **Historical performance** (256 features): past bug patterns
- **Current test set** (128 features): diversity, complexity, priority

Total: 576 features

**Action Space:**
- Dimension: Number of test types (10)
- Actions: Probabilities for each test type
- Continuous action space (softmax output)

**Initialization:**
```python
def __init__(self):
    self.policy_net = PolicyNetwork(state_dim, action_dim)
    self.value_net = ValueNetwork(state_dim)
    self.policy_optimizer = Adam(policy_net.parameters(), lr=1e-3)
    self.value_optimizer = Adam(value_net.parameters(), lr=1e-3)
    self.experience_buffer = ExperienceBuffer(10000)
    self.reward_calculator = RewardCalculator()
    self.training_step = 0
    self.exploration_rate = 1.0
```

**Key Methods:**

1. **optimize(state: Dict, test_cases: List) -> List**
   - Main optimization entry point
   - Creates state tensor
   - Gets action probabilities from policy
   - Selects and reorders test cases
   - Adds exploration
   - Returns optimized test cases

2. **create_state(test_cases: List, api_spec: Dict) -> Tensor**
   - Extracts features
   - Concatenates feature vectors
   - Returns state tensor (1 × 576)

**Feature Extractors:**

3. **extract_api_features(api_spec) -> ndarray (128)**
   - Parameter count (normalized)
   - Type distribution (string, integer, boolean, object, array)
   - HTTP method encoding
   - Authentication flag
   - Response code count
   - Validation rule count

4. **extract_coverage_features(test_cases) -> ndarray (64)**
   - Test type distribution
   - Total test count
   - Assertion coverage
   - Diversity metrics

5. **extract_history_features(api_spec) -> ndarray (256)**
   - Historical bug patterns
   - Past failure rates
   - Common edge cases
   - (Currently placeholder - would use historical data)

6. **extract_test_features(test_cases) -> ndarray (128)**
   - Diversity score (unique test types)
   - Complexity score (average assertions)
   - Priority score (weighted by test type)

**Test Selection:**

7. **select_test_cases(test_cases: List, action_probs: Tensor) -> List**
   - Scores each test case
   - Maps test type to action probability
   - Adds priority bonus (auth=1.5×, validation=1.5×)
   - Sorts by score (descending)
   - Returns sorted tests

8. **add_exploration(test_cases: List, all_tests: List) -> List**
   - Adds random tests (20% of current selection)
   - Prevents local optima
   - Promotes diversity

**Learning:**

9. **update_from_feedback(state, action, reward, next_state, done)**
   - Stores experience in buffer
   - Triggers training when buffer ready (min 1000 experiences)

10. **train()**
    - Main PPO training loop
    - Samples batch from experience buffer
    - Calculates GAE (Generalized Advantage Estimation)
    - Updates policy with clipped objective
    - Updates value network with MSE loss
    - Updates learning rate schedule
    - Decays exploration rate

**PPO Training Loop:**

For n_epochs (10):
  1. Get current policy predictions
  2. Calculate probability ratio: `ratio = exp(log_π_new - log_π_old)`
  3. Compute surrogate objectives:
     - `L1 = ratio × advantage`
     - `L2 = clip(ratio, 1-ε, 1+ε) × advantage`
  4. Policy loss: `-min(L1, L2) - entropy_bonus`
  5. Value loss: `MSE(V_pred, returns)`
  6. Backpropagate and update

11. **calculate_gae(rewards, values, next_values, dones) -> Tensor**
    - Generalized Advantage Estimation
    - Formula: `A_t = δ_t + γλδ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}`
    - Where: `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
    - Balances bias-variance tradeoff
    - Returns normalized advantages

**Learning Rate Schedule:**

12. **update_learning_rate()**
    - Calls `rl_config.get_learning_rate(step)`
    - Schedules: linear, exponential, cosine annealing
    - Updates both optimizers

**Checkpointing:**

13. **save_checkpoint(path: str)**
    - Saves:
      - Policy network state
      - Value network state
      - Optimizer states
      - Training step
      - Exploration rate

14. **load_checkpoint(path: str)**
    - Restores all saved states
    - Resumes training

**Exploration Strategy:**
- Initial: 100% exploration
- Decay: 0.995 per step
- Minimum: 1% exploration
- ε-greedy with probability-based exploration

---

## Core Orchestration Engine

### Location: `core/`

Coordinates the entire test generation and execution pipeline.

### **engine.py**

**CoreEngine Class:**
Main orchestrator coordinating all system components.

**Components:**
```python
self.input_processor = InputProcessor()
self.rag_system = RAGSystem()
self.llama_orchestrator = LlamaOrchestrator()
self.rl_optimizer = RLOptimizer()
self.test_executor = TestExecutor()
self.feedback_loop = FeedbackLoop()
self.report_generator = ReportGenerator()
```

**APITestRequest Dataclass:**
```python
@dataclass
class APITestRequest:
    code_files: List[str]
    language: str
    endpoint_url: str
    test_types: List[str] = None
    max_tests: int = 50
    include_edge_cases: bool = True
```

**Main Processing Pipeline:**

**process_api(request: APITestRequest) -> Dict**

7-step pipeline:

1. **_analyze_api(request)**
   - Parses code files
   - Builds API specification
   - Extracts business logic
   - Extracts validation rules
   - Returns enriched API spec

2. **_retrieve_context(api_spec)**
   - Generates embeddings for API spec
   - Retrieves similar test cases from vector store
   - Retrieves relevant edge cases
   - Retrieves validation patterns
   - Returns context dictionary

3. **_generate_tests(api_spec, context)**
   - Orchestrates LLM agents
   - Generates comprehensive test suite
   - Returns test cases list

4. **_optimize_tests(test_cases, api_spec)**
   - Creates RL state representation
   - Gets optimal test selection and ordering
   - Returns optimized test cases

5. **_execute_tests(test_cases, endpoint_url)**
   - Executes tests in parallel
   - Collects results
   - Returns execution results

6. **_process_feedback(execution_results)**
   - Updates RAG system with new patterns
   - Updates RL model with rewards
   - Detects API drift
   - Logs warnings if drift detected

7. **_generate_report(execution_results)**
   - Generates QASE-style report
   - Returns formatted report

**Session Management:**

`_create_session(request) -> Dict`
```python
{
    'id': 'session_20240101_120000',
    'request': request,
    'started_at': datetime.now(),
    'status': 'in_progress'
}
```

**Metrics Tracking:**

`_update_metrics(execution_results)`
```python
{
    'total_tests': int,
    'passed_tests': int,
    'failed_tests': int,
    'pass_rate': float,
    'bugs_found': int,
    'edge_cases_covered': int
}
```

**Error Handling:**
- Try-catch around entire pipeline
- Returns error status with session ID
- Logs all errors

**Return Structure:**
```python
{
    'status': 'success|error',
    'session_id': '...',
    'api_specification': {...},
    'test_cases': [...],
    'execution_results': [...],
    'report': {...},
    'metrics': {...}
}
```

### **pipeline.py**

**TestGenerationPipeline Class:**
Manages complete test generation pipeline with stage-based execution.

**PipelineStage Dataclass:**
```python
@dataclass
class PipelineStage:
    name: str
    function: callable
    required: bool = True
    timeout: int = 60
```

**Pipeline Stages:**

1. **validation** (10s timeout)
   - Validates input requirements
   - Checks file existence
   - Validates language support

2. **parsing** (30s timeout)
   - Parses API code
   - Extracts components

3. **analysis** (60s timeout)
   - Analyzes API specification
   - Validates API spec

4. **retrieval** (30s timeout)
   - Retrieves context from RAG

5. **generation** (120s timeout)
   - Generates test cases
   - Validates generated tests

6. **optimization** (60s timeout, optional)
   - Optimizes with RL
   - Can skip if fails

7. **execution** (300s timeout)
   - Executes test cases
   - Collects results

8. **feedback** (30s timeout, optional)
   - Processes feedback
   - Updates knowledge base

9. **reporting** (30s timeout)
   - Generates final report

**Key Methods:**

**run(request: Dict) -> Dict**
- Executes all stages sequentially
- Handles timeouts with asyncio.wait_for
- Skips optional stages on failure
- Calculates pipeline metrics
- Returns comprehensive results

**_execute_stage(stage: PipelineStage, request: Dict)**
- Executes single stage with timeout
- Stores result in stage_results
- Logs duration
- Raises on timeout or error (if required)

**Stage Functions:**

Each stage function (e.g., _validate_input, _parse_code) calls corresponding CoreEngine methods.

**Metrics:**
```python
{
    'total_duration': float,
    'stages_completed': int,
    'total_stages': int,
    'success_rate': float,
    'tests_generated': int,
    'tests_executed': int,
    'test_pass_rate': float
}
```

---

## Test Execution & Output

### Location: `test_execution/`, `output/`

Executes tests against live APIs and generates reports.

### **Test Execution (Not Fully Implemented)**

Expected TestExecutor class would:
- Execute HTTP requests
- Validate responses
- Collect results
- Handle retries and timeouts

### **Report Generation (Not Fully Implemented)**

Expected ReportGenerator class would:
- Format execution results
- Generate HTML/PDF reports
- Create QASE-compatible output
- Include charts and statistics

---

# Data Flow & Processing Pipeline

### Complete System Flow:

```
1. USER INPUT
   ├── API Code Files (.cs, .py, .java, .cpp)
   ├── Configuration (language, endpoint_url, test types)
   └── Submit to CoreEngine.process_api()

2. INPUT PROCESSING
   ├── ParserFactory → Language-specific parser
   ├── Parse code → Extract endpoints, validations, models
   ├── EndpointExtractor → Normalize across languages
   ├── SpecificationBuilder → Build OpenAPI spec
   └── Output: API Specification (OpenAPI 3.0)

3. RAG CONTEXT RETRIEVAL
   ├── EmbeddingManager → Generate embeddings for API spec
   ├── VectorStore → Search FAISS indices
   │   ├── test_patterns
   │   ├── edge_cases
   │   └── validation_rules
   ├── Retriever → Retrieve top-k similar documents
   ├── Optional: Cross-encoder reranking
   ├── Optional: MMR for diversity
   └── Output: Retrieved Context

4. LLM TEST GENERATION
   ├── Agent Manager orchestration
   ├── AnalyzerAgent → Analyze API (priority 1)
   │   └── Output: Analysis (critical params, auth, business logic)
   ├── Parallel execution (priority 2):
   │   ├── TestDesignerAgent → Generate test cases
   │   │   ├── Happy path tests
   │   │   ├── Validation tests
   │   │   ├── Auth tests
   │   │   ├── Error tests
   │   │   ├── Boundary tests
   │   │   └── Performance tests
   │   └── EdgeCaseAgent → Generate edge cases
   │       ├── Parameter edge cases
   │       ├── Combination edge cases
   │       ├── Security edge cases
   │       └── State edge cases
   ├── DataGeneratorAgent → Generate test data (priority 3)
   │   ├── Valid data
   │   ├── Invalid data
   │   ├── Boundary data
   │   └── Edge data
   └── Output: Complete Test Suite

5. RL OPTIMIZATION
   ├── Extract features → Create state representation
   │   ├── API complexity features
   │   ├── Test coverage features
   │   ├── Historical performance features
   │   └── Current test set features
   ├── PolicyNetwork → Predict action probabilities
   ├── Score and rank test cases
   ├── Add exploration (ε-greedy)
   └── Output: Optimized Test Suite

6. TEST EXECUTION
   ├── Execute tests against live API
   ├── Parallel execution (configurable workers)
   ├── Collect results (passed/failed, errors, timing)
   └── Output: Execution Results

7. FEEDBACK & LEARNING
   ├── Update RAG System
   │   ├── Index successful test patterns
   │   ├── Index discovered edge cases
   │   └── Index bug patterns
   ├── Update RL Model
   │   ├── Calculate rewards
   │   ├── Store experience in buffer
   │   ├── Train policy and value networks (PPO)
   │   └── Update exploration rate
   └── Detect API drift (changes in behavior)

8. REPORT GENERATION
   ├── ReportWriterAgent → Generate QASE-style report
   ├── Summary statistics
   ├── Individual test case reports
   │   ├── Preconditions
   │   ├── Steps
   │   ├── Expected vs Actual
   │   ├── Failure analysis (if failed)
   │   └── Attachments (requests/responses)
   ├── Recommendations based on failures
   └── Output: Comprehensive Test Report

9. USER OUTPUT
   ├── Test execution results
   ├── Generated test cases
   ├── API specification
   ├── Metrics and statistics
   └── Professional test report
```

### State Management:

**Session State:**
- Tracked in CoreEngine
- Includes: session_id, request, start time, status
- Persisted execution history
- Metrics accumulation

**RAG State:**
- Vector indices persisted to disk
- Indexed document tracking
- Knowledge base files (JSON)
- Embedding cache (pickle files)

**RL State:**
- Policy/Value network weights
- Experience buffer
- Training step counter
- Exploration rate
- Checkpointed periodically

---

## Scripts & Utilities

### Location: `scripts/`

### **train.py**

**RLTrainer Class:**
Standalone RL training script.

**Purpose:** Train policy and value networks on historical data

**Key Methods:**

1. **__init__(state_dim, action_dim, device=None, save_dir=None)**
   - Initializes networks and optimizer
   - Creates experience buffer
   - Sets up checkpoint directory

2. **add_experience(exp: Experience)**
   - Adds experience to buffer
   - Used to populate buffer before training

3. **train_step() -> Dict**
   - Runs single optimizer step
   - Returns training metrics

4. **fit(steps: int)**
   - Runs multiple training steps
   - Logs progress every 50 steps
   - Prints final duration

5. **save_checkpoint(name='checkpoint.pt') -> Path**
   - Saves model states
   - Saves optimizer states
   - Saves dimensions

6. **load_checkpoint(path: Path)**
   - Loads model states
   - Loads optimizer states
   - Allows resuming training

**Usage:**
```python
trainer = RLTrainer(state_dim=576, action_dim=10)
# Add experiences
for exp in experiences:
    trainer.add_experience(exp)
# Train
trainer.fit(steps=1000)
# Save
trainer.save_checkpoint('model.pt')
```

### **evaluate.py**

**Evaluator Class:**
Runs test executor and generates reports.

**Purpose:** Evaluate trained RL model on test set

**Key Methods:**

1. **__init__(state_dim, action_dim, checkpoint=None, device=None)**
   - Loads policy network
   - Loads checkpoint if provided
   - Initializes TestExecutor

2. **evaluate(**kwargs) -> Dict**
   - Executes tests through TestExecutor
   - Compatible with multiple method names (run, execute, run_tests)
   - Returns execution results

3. **generate_report(results, output_path=None) -> Optional[Path]**
   - Generates report using ReportGenerator
   - Tries common method names (generate, write, render, save)
   - Returns path to generated report

**Usage:**
```python
evaluator = Evaluator(
    state_dim=576,
    action_dim=10,
    checkpoint=Path('model.pt')
)
results = evaluator.evaluate()
report_path = evaluator.generate_report(results)
```

### **index_knowledge.py**

**KnowledgeIndexerRunner Class:**
Indexes knowledge base documents into vector store.

**Purpose:** Build/update vector indices for RAG system

**Key Methods:**

1. **__init__(out_dir=None)**
   - Dynamically imports RAG modules
   - Sets output directory for indices

2. **_resolve(module, candidates: List[str])**
   - Resolves class/function names from module
   - Tries multiple candidate names
   - Provides flexibility for evolving APIs

3. **index_paths(paths: Iterable[Path], namespace='default') -> Path**
   - Indexes files into vector store
   - Steps:
     1. Validates file paths
     2. Resolves RAG components (Chunker, Embedder, Indexer, VectorStore)
     3. Instantiates components
     4. Reads and chunks documents
     5. Adds chunks to index
     6. Persists index
   - Returns path to created index

**Usage:**
```python
indexer = KnowledgeIndexerRunner(out_dir='data/vectors')
index_path = indexer.index_paths(
    paths=[Path('test1.py'), Path('test2.py')],
    namespace='test_patterns'
)
```

**Component Resolution:**
- Tries multiple naming conventions
- Handles callable factories or classes
- Fallback strategies for missing components
- Graceful error handling

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- LM Studio with Llama 3.2 model
- CUDA-capable GPU (optional, for RL training)

### Dependencies

**Core Dependencies:**
```txt
# LLM & NLP
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
aiohttp>=3.8.0
tenacity>=8.2.0

# Vector Database
faiss-cpu>=1.7.4  # or faiss-gpu for GPU
numpy>=1.24.0

# API Parsing
tree-sitter>=0.20.0  # For code parsing

# Testing
requests>=2.31.0
pytest>=7.4.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

**Install:**
```bash
pip install -r requirements.txt
```

### LM Studio Setup

1. **Download LM Studio:**
   - Visit https://lmstudio.ai/
   - Download for your platform

2. **Download Llama 3.2 Model:**
   - Open LM Studio
   - Search for "llama-3.2-3b-instruct"
   - Download model

3. **Start Server:**
   - Load model in LM Studio
   - Click "Start Server"
   - Verify running on http://127.0.0.1:1234

4. **Test Connection:**
```python
from llm.llama_client import LlamaClient

async with LlamaClient() as client:
    response = await client.generate("Hello!")
    print(response)
```

### Directory Structure Setup

```bash
api-testing-agent/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── llama_config.py
│   ├── rag_config.py
│   └── rl_config.py
├── input_processing/
│   ├── __init__.py
│   ├── parser_factory.py
│   ├── parsers/
│   │   ├── base_parser.py
│   │   ├── csharp_parser.py
│   │   ├── python_parser.py
│   │   ├── java_parser.py
│   │   └── cpp_parser.py
│   ├── endpoint_extractor.py
│   ├── validator_extractor.py
│   └── specification_builder.py
├── rag/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── chunking.py
│   ├── retriever.py
│   ├── indexer.py
│   └── knowledge_base.py
├── llm/
│   ├── __init__.py
│   ├── llama_client.py
│   ├── prompts/
│   │   ├── prompt_templates.py
│   │   └── prompt_builder.py
│   ├── response_parser.py
│   └── agents/
│       ├── base_agent.py
│       ├── analyzer_agent.py
│       ├── test_designer.py
│       ├── edge_case_agent.py
│       ├── data_generator.py
│       └── report_writer.py
├── reinforcement_learning/
│   ├── __init__.py
│   ├── policy_network.py
│   ├── value_network.py
│   ├── experience_buffer.py
│   ├── reward_calculator.py
│   └── rl_optimizer.py
├── core/
│   ├── __init__.py
│   ├── engine.py
│   └── pipeline.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── index_knowledge.py
├── data/
│   ├── training/
│   ├── vectors/
│   ├── models/
│   ├── reports/
│   └── knowledge_base/
├── logs/
├── requirements.txt
└── .env
```

### Environment Configuration

Create `.env` file:
```env
# LM Studio
LLAMA_BASE_URL=http://127.0.0.1:1234/v1
LLAMA_MODEL=llama-3.2-3b-instruct

# Application
DEBUG=False
MAX_WORKERS=10
BATCH_SIZE=32
MAX_TESTS_PER_ENDPOINT=50

# API Testing
API_TIMEOUT=30
MAX_RETRIES=3

# Logging
LOG_LEVEL=INFO
```

### Initialize System

```python
# Initialize RAG system
from rag import RAGSystem
from rag.knowledge_base import KnowledgeBase

rag = RAGSystem()
kb = KnowledgeBase()

# Knowledge base is auto-initialized with defaults
print(kb.get_statistics())

# Initialize RL system (optional - for advanced usage)
from reinforcement_learning import RLOptimizer

rl = RLOptimizer()
```

---

## Usage Examples

### Example 1: Basic API Testing

```python
import asyncio
from core.engine import CoreEngine, APITestRequest

async def test_api():
    # Create engine
    engine = CoreEngine()
    
    # Create request
    request = APITestRequest(
        code_files=['Controllers/UserController.cs'],
        language='csharp',
        endpoint_url='http://localhost:5000',
        max_tests=30,
        include_edge_cases=True
    )
    
    # Process API
    result = await engine.process_api(request)
    
    # Check results
    if result['status'] == 'success':
        print(f"Session ID: {result['session_id']}")
        print(f"Total tests: {result['metrics']['total_tests']}")
        print(f"Pass rate: {result['metrics']['pass_rate']:.2%}")
        print(f"Bugs found: {result['metrics']['bugs_found']}")
        
        # Save report
        with open('report.json', 'w') as f:
            json.dump(result['report'], f, indent=2)
    else:
        print(f"Error: {result['error']}")

asyncio.run(test_api())
```

### Example 2: Using Pipeline with Custom Stages

```python
from core.pipeline import TestGenerationPipeline

async def run_pipeline():
    pipeline = TestGenerationPipeline()
    
    request = {
        'code_files': ['api.py'],
        'language': 'python',
        'endpoint_url': 'http://localhost:8000',
        'test_types': ['happy_path', 'validation', 'security']
    }
    
    result = await pipeline.run(request)
    
    print(f"Pipeline completed: {result['status']}")
    print(f"Stages completed: {result['stages_completed']}")
    print(f"Duration: {result['metrics']['total_duration']:.2f}s")
    
    # Access results
    test_cases = result['results']['test_cases']
    report = result['results']['report']

asyncio.run(run_pipeline())
```

### Example 3: Direct Agent Usage

```python
from llm.llama_client import LlamaClient
from llm.agents import AnalyzerAgent, TestDesignerAgent

async def use_agents():
    async with LlamaClient() as client:
        # Analyze API
        analyzer = AnalyzerAgent(client)
        analysis = await analyzer.analyze(
            api_spec={
                'path': '/api/users/{id}',
                'method': 'GET',
                'parameters': [
                    {'name': 'id', 'type': 'integer', 'in': 'path', 'required': True}
                ]
            },
            context={}
        )
        
        print("Analysis:", analysis)
        
        # Design tests
        designer = TestDesignerAgent(client)
        tests = await designer.design_tests(
            analysis=analysis,
            context={},
            config={'max_tests': 10}
        )
        
        print(f"Generated {len(tests)} test cases")
        for test in tests[:3]:
            print(f"- {test['name']}: {test['description']}")

asyncio.run(use_agents())
```

### Example 4: RAG System Usage

```python
from rag import RAGSystem
from input_processing import InputProcessor

async def use_rag():
    # Parse API code
    processor = InputProcessor()
    parsed_data = processor.parse_code(['api.cs'], 'csharp')
    api_spec = processor.build_specification(parsed_data)
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Generate embeddings
    embeddings = await rag.generate_embeddings(api_spec)
    
    # Retrieve similar tests
    similar_tests = await rag.retrieve_similar_tests(embeddings, k=5)
    print(f"Found {len(similar_tests)} similar tests")
    
    # Retrieve edge cases
    edge_cases = await rag.retrieve_edge_cases(embeddings, k=5)
    print(f"Found {len(edge_cases)} relevant edge cases")
    
    # Index new test case
    test_case = {
        'name': 'Test user creation',
        'test_type': 'happy_path',
        'endpoint': '/api/users',
        'method': 'POST'
    }
    await rag.index_test_cases([test_case])

asyncio.run(use_rag())
```

### Example 5: RL Training

```python
from scripts.train import RLTrainer
from reinforcement_learning.experience_buffer import Experience
import torch

# Initialize trainer
trainer = RLTrainer(
    state_dim=576,
    action_dim=10,
    save_dir='data/models'
)

# Add training experiences
# (In practice, these come from test execution)
for i in range(1000):
    state = torch.randn(576)
    action = torch.randint(0, 10, (1,))
    reward = np.random.random()
    next_state = torch.randn(576)
    done = False
    
    exp = Experience(state, action, reward, next_state, done)
    trainer.add_experience(exp)

# Train
trainer.fit(steps=500)

# Save checkpoint
checkpoint_path = trainer.save_checkpoint('trained_model.pt')
print(f"Model saved to {checkpoint_path}")
```

### Example 6: Knowledge Base Management

```python
from rag.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

# Add test pattern
kb.add_knowledge('test_patterns', {
    'name': 'Rate Limiting Test',
    'description': 'Test API rate limiting',
    'applicable_to': ['GET', 'POST'],
    'test_data': {
        'strategy': 'Send multiple rapid requests',
        'examples': ['100 requests in 1 second']
    }
})

# Get patterns for endpoint
patterns = kb.get_test_pattern_for_endpoint('POST')
print(f"Found {len(patterns)} patterns for POST")

# Get edge cases for data type
string_edge_cases = kb.get_edge_cases_for_type('string')
for case in string_edge_cases[:3]:
    print(f"- {case['description']}: {case['value']}")

# Search knowledge base
results = kb.search_knowledge('sql injection')
print(f"Found {len(results)} items about SQL injection")

# Export knowledge
export_path = kb.export_knowledge()
print(f"Knowledge exported to {export_path}")

# Statistics
stats = kb.get_statistics()
print(f"Total items: {stats['total_items']}")
print(f"By type: {stats['by_type']}")
```

### Example 7: Custom Parser Registration

```python
from input_processing import ParserFactory
from input_processing.parsers import BaseParser

# Create custom parser
class RustParser(BaseParser):
    def parse(self, code_files):
        # Implementation
        pass
    
    def extract_endpoints(self, code):
        # Implementation
        pass
    
    # ... other methods

# Register parser
factory = ParserFactory()
factory.register_parser('rust', RustParser)

# Use parser
parser = factory.get_parser('rust')
parsed_data = parser.parse(['api.rs'])
```

### Example 8: Indexing Custom Knowledge

```python
from scripts.index_knowledge import KnowledgeIndexerRunner
from pathlib import Path

# Initialize indexer
indexer = KnowledgeIndexerRunner(out_dir='data/vectors')

# Index test files
test_files = [
    Path('tests/test_users.py'),
    Path('tests/test_products.py'),
    Path('tests/test_orders.py')
]

index_path = indexer.index_paths(
    paths=test_files,
    namespace='custom_tests'
)

print(f"Indexed {len(test_files)} files to {index_path}")
```

---

## Advanced Topics

### Custom Agent Creation

```python
from llm.agents.base_agent import BaseAgent

class CustomSecurityAgent(BaseAgent):
    def __init__(self, llama_client):
        super().__init__(llama_client, 'custom_security')
    
    async def execute(self, input_data):
        api_spec = input_data['api_spec']
        
        prompt = f"""
        Generate advanced security test cases for:
        {api_spec}
        
        Focus on:
        - OWASP Top 10 vulnerabilities
        - Zero-day attack patterns
        - Advanced injection techniques
        """
        
        response = await self.generate_json_with_retry(prompt)
        return response

# Use custom agent
async with LlamaClient() as client:
    agent = CustomSecurityAgent(client)
    security_tests = await agent.execute({'api_spec': spec})
```

### Custom Reward Function

```python
from reinforcement_learning.reward_calculator import RewardCalculator

class CustomRewardCalculator(RewardCalculator):
    def calculate_reward(self, test_results, metrics):
        reward = super().calculate_reward(test_results, metrics)
        
        # Add custom rewards
        if metrics.get('critical_bug_found'):
            reward += 50.0
        
        if metrics.get('security_vulnerability_found'):
            reward += 30.0
        
        # Custom penalty
        if metrics.get('slow_test'):
            reward -= 5.0
        
        return reward

# Use in RL optimizer
rl_optimizer = RLOptimizer()
rl_optimizer.reward_calculator = CustomRewardCalculator()
```

### Custom Chunking Strategy

```python
from rag.chunking import ChunkingStrategy, Chunk

class CustomChunker(ChunkingStrategy):
    def chunk_document(self, document, metadata=None, strategy='custom'):
        # Custom chunking logic
        chunks = []
        
        # Example: Split by API endpoint definitions
        endpoints = self.extract_endpoints(document)
        
        for i, endpoint in enumerate(endpoints):
            chunk = Chunk(
                text=endpoint,
                metadata={'endpoint_index': i, **metadata},
                start_idx=0,
                end_idx=len(endpoint),
                chunk_id=f"endpoint_{i}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def extract_endpoints(self, document):
        # Implementation
        pass

# Use custom chunker
chunker = CustomChunker()
chunks = chunker.chunk_document(api_code, strategy='custom')
```

---

## Troubleshooting

### Common Issues

**1. LM Studio Connection Error**
```
Error: Connection refused to http://127.0.0.1:1234
```
Solution:
- Ensure LM Studio is running
- Verify server is started in LM Studio
- Check firewall settings
- Verify base URL in config

**2. FAISS Index Error**
```
RuntimeError: Index not trained
```
Solution:
- IVF indices require training
- Ensure at least nlist (100) vectors before searching
- Or use Flat index for small datasets

**3. Out of Memory (GPU)**
```
RuntimeError: CUDA out of memory
```
Solution:
- Reduce batch_size in config
- Use CPU instead: `torch.device('cpu')`
- Reduce network sizes
- Use gradient accumulation

**4. Parser Not Found**
```
ValueError: Unsupported language: go
```
Solution:
- Check SUPPORTED_LANGUAGES in config
- Implement parser for new language
- Register parser with factory

**5. JSON Parsing Error**
```
JSONDecodeError: Expecting value
```
Solution:
- LLM response not properly formatted
- Adjust temperature (lower for more deterministic)
- Improve prompt instructions
- Use retry logic (already implemented)

### Performance Optimization

**1. Faster Retrieval:**
```python
# Use HNSW index instead of IVF
rag_config.index_type = "HNSW"

# Reduce top_k
rag_config.top_k = 5

# Disable reranking for speed
rag_config.rerank = False
```

**2. Faster Generation:**
```python
# Reduce max_tokens
llama_config.max_tokens = 1024

# Increase temperature for faster sampling
llama_config.temperature = 0.8

# Disable streaming
llama_config.stream = False
```

**3. Parallel Processing:**
```python
# Increase workers
settings.MAX_WORKERS = 20

# Use async/await throughout
async def process_multiple():
    tasks = [engine.process_api(req) for req in requests]
    results = await asyncio.gather(*tasks)
```

**4. Caching:**
```python
# Enable embedding cache
rag_config.enable_cache = True
rag_config.cache_ttl = 7200

# Pre-generate embeddings
embeddings = await rag.embedding_manager.embed_batch(texts)
```

---

## Testing & Validation

### Unit Tests

```python
# Test parser
def test_csharp_parser():
    parser = CSharpParser()
    parsed = parser.parse(['test_controller.cs'])
    assert 'endpoints' in parsed
    assert len(parsed['endpoints']) > 0

# Test embedding
async def test_embeddings():
    manager = EmbeddingManager()
    embedding = await manager.embed_text("test")
    assert embedding.shape == (768,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0)

# Test RL components
def test_policy_network():
    net = PolicyNetwork(state_dim=576, action_dim=10)
    state = torch.randn(1, 576)
    probs = net(state)
    assert probs.shape == (1, 10)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
```

### Integration Tests

```python
async def test_full_pipeline():
    engine = CoreEngine()
    request = APITestRequest(
        code_files=['test_api.cs'],
        language='csharp',
        endpoint_url='http://localhost:5000'
    )
    
    result = await engine.process_api(request)
    
    assert result['status'] == 'success'
    assert 'test_cases' in result
    assert 'report' in result
    assert result['metrics']['total_tests'] > 0
```

---

## Contributing & Extension

### Adding New Language Support

1. **Create Parser:**
```python
# input_processing/parsers/go_parser.py
class GoParser(BaseParser):
    def parse(self, code_files):
        # Implement Go parsing
        pass
```

2. **Register Parser:**
```python
# input_processing/parser_factory.py
self.parsers = {
    # ...
    'go': GoParser
}
```

3. **Update Config:**
```python
# config/settings.py
SUPPORTED_LANGUAGES = [..., "go"]
```

### Adding New Agent

1. **Create Agent:**
```python
# llm/agents/performance_agent.py
class PerformanceAgent(BaseAgent):
    def __init__(self, llama_client):
        super().__init__(llama_client, 'performance')
    
    async def execute(self, input_data):
        # Implement performance testing logic
        pass
```

2. **Register in Manager:**
```python
# llm/agent_manager.py
self.agents = {
    # ...
    AgentType.PERFORMANCE: PerformanceAgent()
}
```

### Adding New Knowledge Type

```python
# rag/knowledge_base.py
self.knowledge_types = {
    # ...
    'performance_patterns': 'Performance testing patterns'
}

# Add default knowledge
def _add_default_performance_patterns(self):
    patterns = [...]
    self.knowledge['performance_patterns'] = patterns
```

---

## API Reference Summary

### Core Classes

**CoreEngine**
- `process_api(request: APITestRequest) -> Dict`

**TestGenerationPipeline**
- `run(request: Dict) -> Dict`

**InputProcessor**
- `parse_code(code_files: List[str], language: str) -> Dict`
- `build_specification(parsed_data: Dict) -> Dict`

**RAGSystem**
- `generate_embeddings(data: Any) -> np.ndarray`
- `retrieve_similar_tests(embedding, k) -> List`
- `index_test_cases(test_cases: List)`

**RLOptimizer**
- `optimize(state: Dict, test_cases: List) -> List`
- `train()`

**LlamaClient**
- `generate(prompt: str) -> str`
- `generate_json(prompt: str, schema: Dict) -> Dict`
- `chat(messages: List[Dict]) -> str`

**Agents**
- `AnalyzerAgent.analyze(api_spec, context) -> Dict`
- `TestDesignerAgent.design_tests(analysis, context, config) -> List`
- `EdgeCaseAgent.generate_edge_cases(api_spec, analysis) -> List`
- `DataGeneratorAgent.generate_data(test_cases, api_spec) -> Dict`
- `ReportWriterAgent.generate_report(results, session) -> Dict`

---

## Performance Metrics

### Expected Performance

**Parsing:**
- C# file (1000 LOC): ~2-5 seconds
- Python file (1000 LOC): ~3-7 seconds (AST parsing)

**RAG Retrieval:**
- Vector search (10k documents): ~50-200ms
- With reranking: ~500-1000ms

**LLM Generation:**
- Analysis: ~3-10 seconds
- Test case generation: ~5-15 seconds per type
- Total for 50 tests: ~2-5 minutes

**RL Optimization:**
- State creation: ~100-500ms
- Action selection: ~50-200ms
- Training step: ~500-2000ms

**End-to-End:**
- Simple API (5 endpoints): ~3-8 minutes
- Complex API (20+ endpoints): ~15-30 minutes

### Scalability

**Vector Store:**
- Tested up to: 100k documents
- Search latency: O(log n) with IVF
- Memory: ~4 bytes per dimension per vector

**RL Training:**
- Buffer capacity: 10k experiences
- Training: ~1-2 hours for 10k steps
- Convergence: typically 50k-100k steps

---

## Future Enhancements

### Planned Features

1. **Additional Language Support:**
   - Go
   - Ruby
   - TypeScript/Node.js
   - Kotlin

2. **Enhanced Test Execution:**
   - Parallel test execution
   - Test result caching
   - Retry mechanisms
   - Mock server integration

3. **Advanced RL:**
   - Multi-agent RL
   - Meta-learning for rapid adaptation
   - Transfer learning across APIs

4. **Improved RAG:**
   - Hybrid retrieval (dense + sparse)
   - Query expansion
   - Adaptive chunking

5. **UI/Dashboard:**
   - Web interface
   - Real-time monitoring
   - Interactive test editing
   - Visualization of coverage

6. **Integration:**
   - CI/CD pipelines
   - JIRA/Azure DevOps
   - Postman collections
   - Swagger/OpenAPI import

---

## License & Acknowledgments

### Technologies Used

- **Llama 3.2**: Meta AI's language model
- **FAISS**: Facebook AI Similarity Search
- **PyTorch**: Deep learning framework
- **Sentence Transformers**: Embedding models
- **LM Studio**: Local LLM runtime

### Architecture Inspiration

- **RAG**: Retrieval-Augmented Generation (Lewis et al., 2020)
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **Multi-Agent Systems**: Cooperative AI agents
- **Test Generation**: Automated software testing research

---

## Contact & Support

For issues, questions, or contributions, please refer to the project repository.

---

**End of Documentation**

This comprehensive README covers every aspect of this API Testing Agent project, from high-level architecture to implementation details of each module, class, and method. The documentation is designed to be both a reference manual and a learning resource for understanding the system's complex interactions between code parsing, RAG, LLM agents, and reinforcement learning.
