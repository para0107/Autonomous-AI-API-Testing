>  **Continuation from:** [github.com/para0107/api-testing-agent](https://github.com/para0107/api-testing-agent)

# API Testing Agent - Comprehensive Technical Documentation

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
Configuration for Llama 3.2 LLM via Groq

**LlamaConfig Class:**
```python
@dataclass
class LlamaConfig:
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ".env file"
    model_name: str = "llama-3.3-70b-versatile"
    
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

**State Space (TOTAL_STATE_DIM = 64):**

The state vector is a compact 64-dimensional representation built from:
- API complexity features (endpoint metadata, parameter counts, types)
- Test coverage features (test type distribution, assertion coverage)
- Historical performance features (past bug patterns, failure rates)
- Current test set features (diversity, complexity, priority scores)

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

**Key Methods:**
1. **extract_flask_endpoints(code)**: Parses `@app.route` decorators
2. **extract_fastapi_endpoints(code)**: Parses `@app.get`/`@router.post` decorators with type hints
3. **extract_models(code)**: Extracts Pydantic and dataclass models
4. **extract_validation_rules(code)**: Extracts Pydantic Field() parameters and `@validator` decorators

### **parsers/cpp_parser.py**

**CppParser Class:**
Parses C++ REST frameworks (Crow, Pistache, Restbed, C++ REST SDK).

**Supported Frameworks:**
- **Crow**: `CROW_ROUTE(app, "/path")`
- **Pistache**: `Routes::Get(router, "/path")`
- **Restbed**: `resource->set_path("/path")`
- **cpprest**: `listener.support(methods::GET, "/path")`

### **endpoint_extractor.py**

**EndpointExtractor Class:**
Normalizes endpoints across different languages to unified format.

**Key Methods:**
1. **normalize_endpoint(endpoint)**: Standardizes field names to OpenAPI equivalents
2. **normalize_path(path)**: Converts path parameter formats (`:id`, `<id>` → `{id}`)
3. **normalize_parameters(parameters)**: Standardizes parameter definitions and constraints
4. **normalize_type(type_str)**: Cross-language type mapping
5. **extract_constraints(param)**: Parses validation constraints
6. **extract_auth_requirements(endpoint)**: Identifies authentication needs
7. **group_endpoints(endpoints)**: Groups endpoints by resource

### **validator_extractor.py**

**ValidatorExtractor Class:**
Extracts and analyzes validation rules from multiple frameworks.

**Supported Frameworks:**
- FluentValidation (C#)
- Bean Validation (Java)
- Pydantic (Python)
- Custom inline validations

**Key Methods:**
1. **extract(parsed_data)**: Extracts validators from parsed code
2. **enhance_validator(validator)**: Adds category, strength, and test_cases
3. **categorize_validation(validator)**: Categories: format, size, presence, pattern, range, custom
4. **determine_validation_strength(validator)**: Returns strict, moderate, or lenient
5. **generate_validation_test_cases(validator)**: Creates test cases per validation rule
6. **extract_inline_validations(methods)**: Scans method bodies for validation patterns

### **specification_builder.py**

**SpecificationBuilder Class:**
Builds complete OpenAPI 3.0 specification from parsed components.

**Key Methods:**
1. **build(parsed_data)**: Main builder → returns OpenAPI 3.0 spec
2. **build_paths(parsed_data)**: Builds paths section from endpoints
3. **build_operation(endpoint)**: Creates operation object with parameters, requestBody, responses
4. **build_parameters(endpoint)**: Converts parameters to OpenAPI format
5. **build_schema(param)**: Creates JSON Schema for parameter
6. **build_request_body(body_params)**: Defines request body for POST/PUT/PATCH
7. **build_responses(endpoint)**: Defines expected responses (200, 400, 401, 404, 500)
8. **build_components(parsed_data)**: Builds reusable schemas and securitySchemes
9. **extract_business_logic(parsed_data)**: Extracts validations, exceptions, workflows, dependencies

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

**Index Creation Strategy:**

The VectorStore always starts with a **Flat index** (IndexFlatL2) wrapped in an IndexIDMap for ID tracking. This avoids the common FAISS error where IVF indices require training data before they can be used. When sufficient data accumulates, the index can be upgraded to IVF for better performance at scale.

**Automatic IVF Fallback:**

When adding embeddings to an untrained IVF index, the system checks if enough vectors are available for training (must be ≥ `nlist`). If insufficient data is present, the system automatically falls back to a Flat index instead of failing:

```python
if n < nlist:
    # Not enough data for IVF — switch to Flat index
    flat_index = faiss.IndexFlatL2(self.dimension)
    index = faiss.IndexIDMap(flat_index)
```

**Initialization:**
```python
def __init__(self):
    self.dimension = 768  # Embedding dimension from rag_config
    self.indices = {}  # Index name → FAISS index
    self.metadata_stores = {}  # Index name → {id → metadata}
    self.index_configs = {}  # Index name → config
```

**Creates Indices:**
- `test_patterns`, `edge_cases`, `validation_rules`
- `api_specifications`, `bug_patterns`, `successful_tests`

**Key Methods:**

1. **_create_index(index_name: str)**
   - Creates a Flat (IndexFlatL2) index by default
   - Wraps with `IndexIDMap` for ID tracking
   - Avoids requiring training before first use

2. **add(index_name, embeddings, metadata, ids)**
   - Ensures proper dtypes: `float32` for embeddings, `int64` for IDs
   - Ensures contiguous arrays via `np.ascontiguousarray()`
   - Handles dimension mismatches (padding or truncation)
   - Trains IVF index if needed, or falls back to Flat
   - Calls `index.add_with_ids(emb, ids_array)` where:
     - `emb`: `np.ndarray` of shape `(n, d)` dtype `float32`
     - `ids_array`: `np.ndarray` of shape `(n,)` dtype `int64`
   - Stores metadata keyed by ID

3. **search(index_name, query_embedding, k=10)**
   - Returns empty results if index has no data
   - Clamps `k` to available vectors
   - Sets IVF nprobe for search
   - Returns: `(ids, distances, metadata)`

4. **search_multiple_indices(query_embedding, indices, k)**
   - Searches across multiple indices
   - Returns results per index

5. **save_index(index_name) / load_index(index_name)**
   - Persists to disk:
     - `index.faiss`: FAISS index
     - `metadata.pkl`: Metadata store
     - `config.json`: Index configuration

6. **get_index_stats(index_name) -> Dict**
   - Returns: `total_embeddings`, `dimension`, `index_type`, `metadata_count`

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

**Key Methods:**
1. **embed_text(text, use_cache=True)**: Generates text embedding (768-dim, L2-normalized)
2. **embed_code(code, language=None)**: Generates code embedding via CodeBERT
3. **embed_structured(data)**: Converts structured data to text and embeds
4. **embed_batch(texts)**: Batch embedding for efficiency (batches of 32)
5. **combine_embeddings(embeddings, weights=None)**: Weighted average of multiple embeddings

**Caching:**
- MD5 hash of text as cache key
- Saves as pickle: `{hash}.pkl`
- `clear_cache()`: Removes all cached embeddings

### **chunking.py**

**ChunkingStrategy Class:**
Splits documents into chunks for efficient embedding and retrieval.

**Strategies:**
1. **Sliding Window** (default): Sentence-based with overlap
2. **Semantic**: Paragraph-based
3. **Code**: Function/class-based splitting
4. **Test**: Test case boundary splitting

**Key Methods:**
1. **chunk_document(document, metadata, strategy)**: Routes to appropriate strategy
2. **sliding_window_chunk(document, metadata)**: Preserves sentence boundaries
3. **semantic_chunk(document, metadata)**: Respects paragraph boundaries
4. **code_chunk(document, metadata)**: Splits by function/class definitions
5. **test_chunk(document, metadata)**: Splits at test function boundaries

### **indexer.py**

**Indexer Class:**
Manages indexing of documents into vector store.

**Key Methods:**
1. **index_document(document, index_name=None)**: Chunks, embeds, and adds single document
2. **index_documents(documents, index_name=None)**: Batch indexes with statistics
3. **index_test_cases(test_cases)**: Specialized indexing for test cases
4. **index_api_specifications(api_specs)**: Indexes OpenAPI specs

**Persistence:**
- Tracks indexed document IDs in JSON to prevent duplicates across sessions

### **retriever.py**

**Retriever Class:**
Handles retrieval of relevant documents from vector store.

**Features:**
- Vector similarity search
- Re-ranking with cross-encoder
- MMR (Maximal Marginal Relevance) for diversity
- Multi-index hybrid search

**Key Methods:**

1. **retrieve(query, index_name, k=10, rerank=None)**
   - Main retrieval method
   - Generates embedding if query is text
   - Optionally reranks results

2. **retrieve_similar_tests(query_embedding, k=10)**
   - Searches `test_patterns` index
   - Parameter name: `query_embedding` (not `embeddings`)

3. **retrieve_edge_cases(query_embedding, k=5)**
   - Searches `edge_cases` index
   - Parameter name: `query_embedding` (not `embeddings`)

4. **retrieve_validation_patterns(query_embedding, k=5)**
   - Searches `validation_rules` index
   - Parameter name: `query_embedding` (not `embeddings`)

5. **hybrid_search(query, indices=None, k=10)**
   - Searches across multiple indices
   - Combines, reranks, and applies MMR for diversity

**Reranking:**
- Uses cross-encoder model for more accurate scoring
- Combines with vector similarity: `(vector_score + rerank_score) / 2`

**Diversity (MMR):**
- Formula: `MMR = λ * relevance - (1-λ) * max_similarity_to_selected`
- Prevents redundant results

### **knowledge_base.py**

**KnowledgeBase Class:**
Manages structured knowledge for API testing.

**Knowledge Types:**
- `test_patterns`, `edge_cases`, `validation_rules`
- `api_patterns`, `bug_patterns`, `best_practices`

**Key Methods:**
1. **add_knowledge(knowledge_type, item)**: Adds new knowledge item
2. **get_knowledge(knowledge_type, filters=None)**: Retrieves with optional filtering
3. **update_knowledge(knowledge_type, item_id, updates)**: Updates existing item
4. **search_knowledge(query, knowledge_types=None)**: Full-text search
5. **get_test_pattern_for_endpoint(method, endpoint_type=None)**: Returns relevant patterns for HTTP method
6. **get_edge_cases_for_type(data_type)**: Returns edge cases for data type
7. **add_test_result(test_case, result)**: Learns from execution results

**Persistence:**
- Each knowledge type saved as JSON file in `data/knowledge_base/`
- `export_knowledge(output_path)` / `import_knowledge(import_path, merge=True)`

---

## LLM Orchestration Layer

### Location: `llm/`

Multi-agent system using Llama 3.2 via LM Studio for specialized testing tasks.

### **Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│                  LlamaOrchestrator                        │
│          (Async Context Manager wrapper)                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │                Agent Manager                        │  │
│  │            (Coordinates all agents)                  │  │
│  └───────┬────────────────────────────────────────────┘  │
│          │                                                │
│          ├──→ Analyzer Agent (API analysis)               │
│          ├──→ Test Designer Agent (Test case design)      │
│          ├──→ Edge Case Agent (Edge case generation)      │
│          ├──→ Data Generator Agent (Test data creation)   │
│          └──→ Report Writer Agent (Report generation)     │
└──────────────────────────────────────────────────────────┘
```

### **llama_client.py**

**LlamaClient Class:**
Client for interacting with Llama 3.2 via LM Studio.

**Connection:**
- Base URL: `http://127.0.0.1:1234/v1` (LM Studio local server)
- Compatible with OpenAI API format
- No API key required

**Async Context Manager:**

LlamaClient uses the async context manager protocol (`__aenter__`/`__aexit__`) for lifecycle management. It does **not** expose `initialize()` or `close()` methods directly.

```python
async with LlamaClient() as client:
    response = await client.generate("Hello!")
```

**Key Methods:**

1. **generate(prompt, **kwargs) -> str**: Sends completion request with retry logic
2. **chat(messages, **kwargs) -> str**: Chat-style completion
3. **generate_json(prompt, schema=None, **kwargs) -> Dict**: Generates structured JSON response
4. **stream_generate(prompt, callback=None, **kwargs)**: Streaming generation
5. **get_embeddings(text) -> List[float]**: Gets embeddings from LM Studio if supported
6. **get_config_for_agent(agent_type) -> Dict**: Returns agent-specific configuration

### **orchestrator.py**

**LlamaOrchestrator Class:**
Wraps LlamaClient and AgentManager in an async context manager.

**Lifecycle Management:**

LlamaOrchestrator delegates to LlamaClient's async context manager protocol:

```python
async def __aenter__(self):
    await self.client.__aenter__()
    self.agent_manager = AgentManager(llama_client=self.client)
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.client:
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
```

**Usage:**
```python
async with LlamaOrchestrator() as orchestrator:
    result = await orchestrator.generate_test_suite(
        api_spec=spec,
        context=context
    )
```

**Key Methods:**

- **generate_test_suite(api_spec, context, config=None)**: Generates a complete test suite using coordinated agents. Delegates to `AgentManager.orchestrate()`. Applies test limits and priority sorting (high > medium > low).

### **connection_check.py**

Utility for verifying LM Studio connectivity before running the pipeline.

### **prompt_templates.py** (in `llm/prompts/`)

**PromptTemplates Class:**
Collection of reusable prompt templates for API analysis, test generation, security testing, data generation, reporting, failure analysis, context integration, and validation rules.

### **prompt_builder.py** (in `llm/prompts/`)

**PromptBuilder Class:**
Dynamically constructs prompts for different scenarios.

**Key Methods:**
1. **build_analysis_prompt(api_spec, context=None)**: Formats API spec with RAG context
2. **build_test_generation_prompt(test_type, api_spec, context=None, examples=None)**: Test case prompt
3. **build_data_generation_prompt(parameters, test_type, constraints=None)**: Test data prompt
4. **build_report_prompt(results, report_type)**: Report generation prompt

### **response_parser.py**

**ResponseParser Class:**
Parses and validates LLM responses.

**Parsers:** `json`, `list`, `code`, `text`, `structured`

**Validators:**
- `validate_test_case(test_case)`: Checks required fields (name, endpoint, method)
- `validate_analysis(analysis)`: Checks required fields (endpoint, method, critical_parameters)

### **agents/base_agent.py**

**BaseAgent (Abstract Class):**

```python
def __init__(self, llama_client, agent_type: str = None):
    self.client = llama_client
    self.agent_type = agent_type
    self.config = self._get_config()
```

**Abstract Method:** `execute(input_data: Dict) -> Any`

**Retry Logic:**
- `generate_with_retry(prompt, max_retries=3)`
- `generate_json_with_retry(prompt, schema=None, max_retries=3)`

### **agents/analyzer_agent.py**

**AnalyzerAgent Class:**
Analyzes API specifications and identifies testing requirements.

**Key Method:** `analyze(api_spec, context) -> Dict`

Returns comprehensive analysis including critical_parameters, auth_requirements, business_logic, failure_points, dependencies, validation_rules, error_scenarios, and performance expectations.

### **agents/test_designer.py**

**TestDesignerAgent Class:**
Designs comprehensive test cases based on analysis.

**Key Method:** `design_tests(analysis, context, config) -> List[Dict]`

**Test Types Generated:**
1. Happy path tests
2. Validation tests
3. Authentication tests
4. Error handling tests
5. Boundary tests
6. Performance tests

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

### **agents/edge_case_agent.py**

**EdgeCaseAgent Class:**
Generates creative edge case and security test scenarios.

**Key Method:** `generate_edge_cases(api_spec, analysis) -> List[Dict]`

**Categories:**
1. Parameter edge cases (boundary values, type mismatches, injection attempts)
2. Combination edge cases (parameter interactions)
3. Security edge cases (SQL injection, XSS, CSRF, path traversal)
4. State-based edge cases (race conditions, concurrent modifications)

### **agents/data_generator.py**

**DataGeneratorAgent Class:**
Generates realistic test data for test cases.

**Key Method:** `generate_data(test_cases, api_spec) -> Dict`

**Data Generation Methods:**
- `_generate_valid_data()`: Realistic data satisfying constraints
- `_generate_validation_data()`: Invalid data for validation testing
- `_generate_boundary_data()`: Boundary values from constraints
- `_generate_edge_data()`: Edge case values (0, -1, MAX_INT, empty, null, injection strings)

### **agents/report_writer.py**

**ReportWriterAgent Class:**
Generates professional QASE-style test reports.

**Key Method:** `generate_report(execution_results, session) -> Dict`

**Report Structure:**
```python
{
    'title': 'API Test Report - {session_id}',
    'generated_at': '2024-01-01T00:00:00',
    'summary': {...},
    'test_cases': [...],  # QASE-style individual reports
    'recommendations': [...],
    'metadata': {...}
}
```

### **agent_manager.py**

**AgentManager Class:**
Coordinates multiple agents with dependency management.

**Initialization:**

AgentManager receives the LlamaClient instance and passes it to each agent:

```python
def __init__(self, llama_client):
    self.agents = {
        AgentType.ANALYZER: AnalyzerAgent(llama_client),
        AgentType.TEST_DESIGNER: TestDesignerAgent(llama_client),
        AgentType.EDGE_CASE: EdgeCaseAgent(llama_client),
        AgentType.DATA_GENERATOR: DataGeneratorAgent(llama_client),
        AgentType.REPORT_WRITER: ReportWriterAgent(llama_client)
    }
    self.task_queue = asyncio.Queue()
    self.results = {}
    self.running_tasks = {}
```

**Key Method:** `orchestrate(api_spec, context) -> Dict`

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

**Result Combination:**
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
        ├──→ Reward Calculator (Reward Function)
        └──→ State Extractor (Feature Engineering)
```

### **state_extractor.py**

**`extract_state()` Function:**
Builds the 64-dimensional state vector (`TOTAL_STATE_DIM = 64`) from test cases, API spec, history, and context.

```python
state = extract_state(
    test_cases=[test_case],
    api_spec=api_spec,
    history=[],
    context=None
)
# Returns: np.ndarray of shape (64,)
```

### **policy_network.py**

**PolicyNetwork Class:**
Neural network for selecting test actions.

**Architecture:**
```python
Input (state_dim) → Linear(256) → ReLU → Dropout(0.1)
                  → Linear(128) → ReLU → Dropout(0.1)
                  → Linear(64) → ReLU → Dropout(0.1)
                  → Linear(action_dim) → Softmax → Action Probabilities
```

**Key Methods:**
1. **forward(state)**: Forward pass → action probabilities
2. **get_action(state, deterministic=False)**: Selects action (argmax or sample)
3. **evaluate_actions(states, actions)**: Returns log_probabilities and entropy for PPO training

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
1. **forward(state)**: Returns single state value
2. **get_value(state)**: Gets value for single state as float

### **experience_buffer.py**

**ExperienceBuffer Class:**
Replay buffer with prioritized experience replay.

**Key Methods:**
1. **add(state, action, reward, next_state, done)**: Adds experience
2. **sample(batch_size)**: Samples batch (uniform or prioritized)
3. **update_priorities(indices, td_errors)**: Updates priorities after training

**Prioritized Experience Replay:**
- Sampling probability: `P(i) = p_i^α / Σ p_j^α`
- Priority = `|TD-error| + ε`

### **reward_calculator.py**

**RewardCalculator Class:**
Calculates rewards based on test execution outcomes.

**Key Methods:**
1. **calculate_reward(test_results, metrics)**: Main reward calculation
2. **calculate_intermediate_reward(state, action)**: Pre-execution reward shaping
3. **get_reward_statistics()**: Returns mean, std, trend analysis

### **rl_optimizer.py**

**RLOptimizer Class:**
Main PPO-based reinforcement learning optimizer.

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

1. **optimize(state, test_cases)**: Main entry point → returns optimized test cases
2. **get_action(state)**: Gets action from policy network (this is the correct method name — not `select_action`)
3. **create_state(test_cases, api_spec)**: Extracts features → returns state tensor
4. **select_test_cases(test_cases, action_probs)**: Scores and sorts tests by action probability
5. **add_exploration(test_cases, all_tests)**: Adds random tests (20%) for diversity
6. **update_from_feedback(state, action, reward, next_state, done)**: Stores experience, triggers training
7. **train()**: PPO training loop with GAE, clipped objective, value loss
8. **calculate_gae(rewards, values, next_values, dones)**: Generalized Advantage Estimation
9. **save_checkpoint(path)** / **load_checkpoint(path)**: Model persistence

**PPO Training Loop:**
```
For n_epochs (10):
  1. Get current policy predictions
  2. Calculate ratio = exp(log_π_new - log_π_old)
  3. Surrogate objectives: L1 = ratio × advantage, L2 = clip(ratio, 1-ε, 1+ε) × advantage
  4. Policy loss = -min(L1, L2) - entropy_bonus
  5. Value loss = MSE(V_pred, returns)
  6. Backpropagate and update
```

**Exploration Strategy:**
- Initial: 100% exploration
- Decay: 0.995 per step
- Minimum: 1% exploration

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

1. **_analyze_api(request)**: Parses code files, builds API specification, extracts business logic and validation rules
2. **_retrieve_context(api_spec)**: Generates embeddings, retrieves similar tests, edge cases, and validation patterns from vector store
3. **_generate_tests(api_spec, context)**: Orchestrates LLM agents to generate comprehensive test suite
4. **_optimize_tests(test_cases, api_spec)**: Creates RL state, gets optimal test selection and ordering
5. **_execute_tests(test_cases, endpoint_url)**: Executes tests in parallel, collects results
6. **_process_feedback(execution_results)**: Updates RAG system and RL model, detects API drift
7. **_generate_report(execution_results)**: Generates QASE-style report

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

**Pipeline Stages:**
1. **validation** (10s timeout): Validates input requirements
2. **parsing** (30s timeout): Parses API code
3. **analysis** (60s timeout): Analyzes API specification
4. **retrieval** (30s timeout): Retrieves context from RAG (uses `query_embedding=` parameter)
5. **generation** (120s timeout): Generates test cases
6. **optimization** (60s timeout, optional): Optimizes with RL
7. **execution** (300s timeout): Executes test cases
8. **feedback** (30s timeout, optional): Processes feedback
9. **reporting** (30s timeout): Generates final report

**RAG Retrieval Call (line ~233):**

The retrieval stage calls RAG retriever methods with the correct parameter name:

```python
retrieval_map = {
    'similar_tests': self.rag_system.retrieve_similar_tests,
    'edge_cases': self.rag_system.retrieve_edge_cases,
    'validation_patterns': self.rag_system.retrieve_validation_patterns,
}

for key, retriever_method in retrieval_map.items():
    results = await retriever_method(query_embedding=embeddings, k=10)
```

**Key Method:** `run(request: Dict) -> Dict`
- Executes all stages sequentially with timeouts
- Skips optional stages on failure
- Returns comprehensive results with pipeline metrics

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
   │   (uses query_embedding= parameter)
   ├── Optional: Cross-encoder reranking
   ├── Optional: MMR for diversity
   └── Output: Retrieved Context

4. LLM TEST GENERATION
   ├── LlamaOrchestrator (async context manager)
   │   ├── __aenter__() → initializes LlamaClient + AgentManager
   │   └── __aexit__() → cleans up client
   ├── AgentManager receives llama_client, passes to all agents
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
   ├── extract_state() → Create 64-dim state vector
   │   ├── API complexity features
   │   ├── Test coverage features
   │   ├── Historical performance features
   │   └── Current test set features
   ├── PolicyNetwork → Predict action probabilities
   ├── RLOptimizer.get_action(state) → Select action
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

**Key Methods:**
1. **__init__(state_dim, action_dim, device=None, save_dir=None)**: Initializes networks and optimizer
2. **add_experience(exp)**: Adds experience to buffer
3. **train_step()**: Runs single optimizer step
4. **fit(steps)**: Runs multiple training steps with logging
5. **save_checkpoint(name)** / **load_checkpoint(path)**: Model persistence

### **evaluate.py**

**Evaluator Class:**
Runs test executor and generates reports.

**Key Methods:**
1. **__init__(state_dim, action_dim, checkpoint=None, device=None)**: Loads policy network
2. **evaluate(**kwargs)**: Executes tests through TestExecutor
3. **generate_report(results, output_path=None)**: Generates report

**Note:** The Evaluator uses `RLOptimizer.get_action(state)` (not `select_action`).

### **index_knowledge.py**

**KnowledgeIndexerRunner Class:**
Indexes knowledge base documents into vector store.

**Key Methods:**
1. **__init__(out_dir=None)**: Dynamically imports RAG modules
2. **index_paths(paths, namespace)**: Indexes files into vector store

### Location: `utils/`

Utility modules for logging, common helpers, and shared functions.

### Root-Level Scripts

- **`main.py`**: Main application entry point — orchestrates the full pipeline
- **`check_indices.py`**: Diagnostic script for inspecting FAISS vector store indices
- **`load_test_cases_to_rag.py`**: Utility for loading test case data into the RAG system
- **`rag_diagnosis.py`**: Diagnostic script for debugging RAG retrieval issues

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

1. **Download LM Studio:** Visit https://lmstudio.ai/
2. **Download Llama 3.2 Model:** Search for "llama-3.2-3b-instruct" and download
3. **Start Server:** Load model → Click "Start Server" → Verify running on http://127.0.0.1:1234
4. **Test Connection:**
```python
from llm.llama_client import LlamaClient

async with LlamaClient() as client:
    response = await client.generate("Hello!")
    print(response)
```

### Directory Structure

```bash
Autonomous-AI-API-Testing/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── llama_config.py
│   ├── rag_config.py
│   └── rl_config.py
├── input_processing/
│   ├── __init__.py              # InputProcessor façade
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
│   ├── vector_store.py          # FAISS vector database
│   ├── embeddings.py            # Embedding generation
│   ├── chunking.py              # Document chunking
│   ├── retriever.py             # Similarity search & retrieval
│   ├── indexer.py               # Document indexing
│   └── knowledge_base.py        # Structured knowledge management
├── llm/
│   ├── __init__.py
│   ├── llama_client.py          # LM Studio client (async context manager)
│   ├── orchestrator.py          # LlamaOrchestrator (async context manager)
│   ├── agent_manager.py         # Agent coordination
│   ├── connection_check.py      # LM Studio connectivity check
│   ├── response_parser.py       # LLM response parsing
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py
│   │   └── prompt_builder.py
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
│   ├── rl_optimizer.py          # Main PPO optimizer (get_action method)
│   └── state_extractor.py       # 64-dim state vector extraction
├── core/
│   ├── __init__.py
│   ├── engine.py                # CoreEngine orchestrator
│   └── pipeline.py              # Stage-based pipeline
├── test_execution/
│   └── (test executor — not fully implemented)
├── scripts/
│   ├── __init__.py
│   ├── train.py                 # RL training
│   ├── evaluate.py              # Model evaluation
│   └── index_knowledge.py       # Knowledge indexing
├── utils/
│   └── (utility modules)
├── output/
│   └── (generated reports)
├── logs/
│   └── (application logs)
├── data/
│   ├── training/
│   ├── vectors/
│   ├── models/
│   ├── reports/
│   └── knowledge_base/
├── main.py                      # Application entry point
├── check_indices.py             # FAISS diagnostic
├── load_test_cases_to_rag.py    # RAG data loading
├── rag_diagnosis.py             # RAG diagnostic
├── requirements.txt
├── .env
├── .gitignore
└── README.md
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
    engine = CoreEngine()
    
    request = APITestRequest(
        code_files=['Controllers/UserController.cs'],
        language='csharp',
        endpoint_url='http://localhost:5000',
        max_tests=30,
        include_edge_cases=True
    )
    
    result = await engine.process_api(request)
    
    if result['status'] == 'success':
        print(f"Session ID: {result['session_id']}")
        print(f"Total tests: {result['metrics']['total_tests']}")
        print(f"Pass rate: {result['metrics']['pass_rate']:.2%}")
        print(f"Bugs found: {result['metrics']['bugs_found']}")

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
        
        # Design tests
        designer = TestDesignerAgent(client)
        tests = await designer.design_tests(
            analysis=analysis,
            context={},
            config={'max_tests': 10}
        )
        
        print(f"Generated {len(tests)} test cases")

asyncio.run(use_agents())
```

### Example 4: Using the Orchestrator

```python
from llm.orchestrator import LlamaOrchestrator

async def use_orchestrator():
    async with LlamaOrchestrator() as orchestrator:
        result = await orchestrator.generate_test_suite(
            api_spec=spec,
            context=context,
            config={'max_tests': 50}
        )
        print(f"Generated {len(result.get('test_cases', []))} tests")

asyncio.run(use_orchestrator())
```

### Example 5: RAG System Usage

```python
from rag import RAGSystem
from input_processing import InputProcessor

async def use_rag():
    processor = InputProcessor()
    parsed_data = processor.parse_code(['api.cs'], 'csharp')
    api_spec = processor.build_specification(parsed_data)
    
    rag = RAGSystem()
    embeddings = await rag.generate_embeddings(api_spec)
    
    # Use query_embedding= parameter for retrieval
    similar_tests = await rag.retrieve_similar_tests(query_embedding=embeddings, k=5)
    edge_cases = await rag.retrieve_edge_cases(query_embedding=embeddings, k=5)

asyncio.run(use_rag())
```

### Example 6: RL Training

```python
from scripts.train import RLTrainer
from reinforcement_learning.experience_buffer import Experience
import torch

trainer = RLTrainer(state_dim=64, action_dim=10, save_dir='data/models')

# Add training experiences
for i in range(1000):
    state = torch.randn(64)
    action = torch.randint(0, 10, (1,))
    reward = np.random.random()
    next_state = torch.randn(64)
    done = False
    
    exp = Experience(state, action, reward, next_state, done)
    trainer.add_experience(exp)

trainer.fit(steps=500)
checkpoint_path = trainer.save_checkpoint('trained_model.pt')
```

### Example 7: Knowledge Base Management

```python
from rag.knowledge_base import KnowledgeBase

kb = KnowledgeBase()

kb.add_knowledge('test_patterns', {
    'name': 'Rate Limiting Test',
    'description': 'Test API rate limiting',
    'applicable_to': ['GET', 'POST'],
    'test_data': {
        'strategy': 'Send multiple rapid requests',
        'examples': ['100 requests in 1 second']
    }
})

patterns = kb.get_test_pattern_for_endpoint('POST')
string_edge_cases = kb.get_edge_cases_for_type('string')
results = kb.search_knowledge('sql injection')
stats = kb.get_statistics()
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
        prompt = f"Generate advanced security test cases for: {api_spec}"
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
        
        if metrics.get('critical_bug_found'):
            reward += 50.0
        if metrics.get('security_vulnerability_found'):
            reward += 30.0
        if metrics.get('slow_test'):
            reward -= 5.0
        
        return reward

rl_optimizer = RLOptimizer()
rl_optimizer.reward_calculator = CustomRewardCalculator()
```

---

## Troubleshooting

### Common Issues

**1. LM Studio Connection Error**
```
Error: Connection refused to http://127.0.0.1:1234
```
Solution: Ensure LM Studio is running with the server started. Verify base URL in config.

**2. FAISS Index Error**
```
RuntimeError: Index not trained
```
Solution: The updated VectorStore starts with a Flat index by default and automatically falls back from IVF when insufficient training data is present. If you still encounter this error, ensure you are using the updated `vector_store.py`.

**3. Out of Memory (GPU)**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch_size in config, use CPU (`torch.device('cpu')`), or reduce network sizes.

**4. Parser Not Found**
```
ValueError: Unsupported language: go
```
Solution: Check SUPPORTED_LANGUAGES in config. Implement and register a parser for the new language.

**5. JSON Parsing Error**
```
JSONDecodeError: Expecting value
```
Solution: LLM response not properly formatted. Adjust temperature (lower = more deterministic). Retry logic is built in.

**6. Unresolved Method Errors**

If your IDE reports unresolved method references, verify:
- `LlamaOrchestrator` uses `__aenter__`/`__aexit__` (not `initialize()`/`close()`)
- `RLOptimizer` uses `get_action()` (not `select_action()`)
- Retriever methods use `query_embedding=` parameter (not `embeddings`)
- `AgentManager` constructor takes `llama_client` parameter

### Performance Optimization

**Faster Retrieval:**
```python
rag_config.index_type = "HNSW"  # Use HNSW instead of IVF
rag_config.top_k = 5            # Reduce top_k
rag_config.rerank = False        # Disable reranking for speed
```

**Faster Generation:**
```python
llama_config.max_tokens = 1024  # Reduce max_tokens
llama_config.temperature = 0.8  # Increase temperature
```

**Parallel Processing:**
```python
settings.MAX_WORKERS = 20
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

# Test RL components
def test_policy_network():
    net = PolicyNetwork(state_dim=64, action_dim=10)
    state = torch.randn(1, 64)
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
        pass
```

2. **Register in Manager:**
```python
# llm/agent_manager.py
self.agents = {
    # ...
    AgentType.PERFORMANCE: PerformanceAgent(llama_client)
}
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
- `retrieve_similar_tests(query_embedding, k) -> List`
- `retrieve_edge_cases(query_embedding, k) -> List`
- `retrieve_validation_patterns(query_embedding, k) -> List`
- `index_test_cases(test_cases: List)`

**RLOptimizer**
- `optimize(state: Dict, test_cases: List) -> List`
- `get_action(state) -> action`
- `train()`

**LlamaClient** (async context manager)
- `generate(prompt: str) -> str`
- `generate_json(prompt: str, schema: Dict) -> Dict`
- `chat(messages: List[Dict]) -> str`

**LlamaOrchestrator** (async context manager)
- `generate_test_suite(api_spec, context, config) -> Dict`

**AgentManager**
- `__init__(llama_client)`: Receives LlamaClient, creates all agents
- `orchestrate(api_spec, context) -> Dict`

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

1. **Additional Language Support:** Go, Ruby, TypeScript/Node.js, Kotlin
2. **Enhanced Test Execution:** Parallel execution, result caching, retry mechanisms, mock server integration
3. **Advanced RL:** Multi-agent RL, meta-learning, transfer learning across APIs
4. **Improved RAG:** Hybrid retrieval (dense + sparse), query expansion, adaptive chunking
5. **UI/Dashboard:** Web interface, real-time monitoring, interactive test editing
6. **Integration:** CI/CD pipelines, JIRA/Azure DevOps, Postman collections, Swagger/OpenAPI import

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

This comprehensive README covers every aspect of the API Testing Agent project, from high-level architecture to implementation details of each module, class, and method. The documentation reflects the current state of the codebase including corrected method signatures, async context manager patterns, and FAISS integration fixes.