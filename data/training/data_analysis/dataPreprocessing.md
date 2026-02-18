# Dataset Preparation for LLM Training

**Summary:** This document describes the three-stage process applied to a raw Qase.io export to produce a balanced, machine-learning-ready dataset suitable for LLM training: Data Simplification, Data Analysis \& Visualization, and Data Augmentation.

## 1. Introduction
The goal of this data engineering process was to transform a proprietary, manual test export into a balanced dataset for training an AI agent. The work was treated as a two-phase experiment: Baseline (Raw \/\ Simplified) vs. Experimental (Augmented).

- Inputs:
  - Raw Qase export: `QA-Backend-Data.json`
  - Simplified output: `simplified_tests.json`
  - Augmented output: `augmented_tests.json`
- Scripts:
  - Simplification: `simplify_dataset.py`
  - Augmentation: `augment_dataset.py`

## 2. Goals and Risks
- Primary goal: Remove proprietary noise and create a dataset that teaches intent-to-test-step mapping.
- Secondary goals: Detect and fix dataset biases that would harm model behavior.
- Key risks found:
  - "One-Step" Bias: excessive single-step tests.
  - Method Imbalance: overrepresentation of GET requests.
  - Remaining atomic nature limiting long multi-step workflows.

## 3. Phase 1 — Data Simplification (Baseline)
### 3.1 Objective
Remove administrative metadata and deep nesting so the dataset focuses on semantic test content.

### 3.2 Processing
Actions performed by `simplify_qase_export.py`:
- Flattened suite/folder hierarchy into a linear list of test cases.
- Filtered out administrative metadata (ids, automation status, is_flaky, etc.).
- Retained semantic fields only:
  - Test Name
  - Objective (Description)
  - Preconditions
  - Steps (consolidated into Action -> Expected Result pairs)

### 3.3 Baseline Metrics (The "Before" State)
- Total Tests: 476  
- Total Steps: 488  
- Average Complexity: 1.03 steps/test (extremely atomic)  
- Method Distribution:
  - GET: 50.0% (dominant)  
  - POST: 12.9%  
  - DELETE: 3.7%

Conclusion: The simplified baseline dataset is heavily read-oriented and atomic, risking a model that is strong at fetch checks but weak at stateful, write-driven workflows.

## 4. Phase 2 — Data Analysis \& Visualization
### 4.1 Purpose
Audit the simplified dataset to detect biases and gaps the AI agent would learn.

### 4.2 Key Insights
- One-Step Bias: Skewness > 3.0; over 90% atomic checks (e.g., "Hit endpoint -> Check 200"). AI may struggle with multi-step workflows.
- Method Imbalance: GET requests comprised ~50% of tests; POST/PUT/DELETE underrepresented. AI might avoid or mishandle data-modifying operations.
- High-Quality Objectives: Detailed and verbose description/objective fields benefit embedding quality.

## 5. Phase 2 — Data Augmentation (Experimental)
### 5.1 Hypothesis
Algorithmically expanding POST/PUT coverage with negative scenarios, injecting security cases, and chaining lifecycle tests will reduce GET dominance and increase multi-step workflows without manual test authoring.

### 5.2 Augmentation Strategies (applied by `augment_dataset.py`)
- Negative Test Generation:
  - For every Create/Update (POST/PUT), generate a negative variant using empty or invalid payloads to enforce validation behavior (expect 400).
- Security Fuzzing:
  - Inject common attack patterns (e.g., SQL-like payloads such as `\' OR \'1\'=\'1`) into data fields to teach the agent to verify secure, non-crashing responses.
- Lifecycle Chaining:
  - Link related tests (Create \-> Get \-> Delete) into single multi-step "Full Lifecycle" tests when corresponding cases exist.

### 5.3 Implementation Notes
- Augmentation logic is rule-based and deterministic to keep traceability.
- Negative tests and fuzzing add high volume single-step cases; lifecycle chaining produces multi-step workflows to counteract the one-step bias.

## 6. Experimental Results (The "After" State)
- Output produced: `augmented_tests.json`
- Aggregate metrics:
  - Total Tests: 686 (▲ +44.1%)  
  - Total Steps: 714 (▲ +46.3%)  
  - GET Frequency: 36.4% (▼ -13.6%, more balanced)  
  - Avg. Steps/test: ~1.04 (nearly unchanged)

Note: Average steps remained low because many negative/fuzz tests are single-step; however, lifecycle chaining increased workflow diversity and created stateful scenarios.

## 7. Outcome and Trade-offs
Successes:
- Reduced GET dominance and increased relative presence of POST/PUT/DELETE.
- Increased dataset size and diversity by ~44% using algorithmic augmentation.
- Improved security-awareness and validation coverage via fuzzing and negative tests.

Trade-offs:
- Dataset remains largely atomic on average; achieving frequent long workflows (5+ steps) likely requires manual curation of complex scenarios.
- Large number of generated single-step negative tests kept mean step count low despite improved diversity.

## 8. Conclusion
The combined simplification, analysis, and augmentation pipeline converted a noisy, highly skewed QA export into a balanced, richer dataset for LLM training. The dataset now better supports write-heavy behaviors and security checks while preserving strong objective descriptions for embedding models. For teaching long-term, multi-step state management, manual creation of complex workflows is recommended as the next phase.

## 9. Next Steps
- Manually author or review complex 5+ step workflows for key entities.
- Monitor model behavior when trained on `augmented_tests.json` and iterate augmentation heuristics.
- Add provenance metadata to augmented cases to trace rule origins.
