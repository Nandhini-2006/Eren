# Eren: User-Controllable Context Memory Optimization for Long-Running AI Agents

A modular, zero-external-dependency Python backend implementing core algorithmic foundations for context memory retrieval, selection, compression, lifecycle management (POMDP), and alignment evaluation in long-running AI agents.

## Architecture & Algorithm Suite

The backend contains the following modular algorithm implementations inside `backend/context_memory/`:

### 1. Context-Memory Retrieval
- **Dense Vector Retrieval (`retrieval.py`)**: Local TF-IDF vectorizer and cosine similarity computation for semantic retrieval.
- **Hierarchical Navigable Small World (`hnsw_index.py`)**: Approximate Nearest Neighbor (ANN) vector index with multi-layer skip-list graph navigation.
- **Reciprocal Rank Fusion (`rrf.py`)**: Combines multiple rankings (e.g. lexical keyword & vector semantic) into a single robust fused score.

### 2. Context Selection & Ranking
- **Maximal Marginal Relevance (`mmr.py`)**: Greedy diverse selection balancing relevance against pairwise redundancy ($\lambda$-tradeoff).
- **Lexicographic Multi-Criteria Ranking (`ranking.py`)**: Multi-attribute candidate scoring across relevance, importance, recency, and access frequency.
- **0/1 Knapsack Optimization (`knapsack.py`)**: Context budget constraint optimizer using Dynamic Programming and Greedy approximations for token capacity limits.

### 3. Memory Compression & Deduplication
- **MinHash (`minhash.py`)**: Locality-sensitive k-shingle hashing with universal hash families for Jaccard similarity estimation.
- **LSH Index (`lsh.py`)**: Multi-band hash index enabling sub-linear near-duplicate memory collision candidate search.
- **Delta-Encoding Compression (`delta_encoding.py`)**: Prefix/suffix run-length delta encoding for episodic memory streams.

### 4. Lifecycle Management
- **POMDP Eviction & Retention (`pomdp.py`)**: Partially Observable Markov Decision Process for value-state estimation, Bayesian belief distribution updates, and optimal memory management policies (`keep`, `compress`, `archive`, `delete`).

### 5. Evaluation, Optimization & Alignment
- **Bayesian Optimization (`bayesian_optimization.py`)**: Gaussian Process regression with RBF kernel and Expected Improvement (EI) acquisition function for hyperparameter tuning.
- **Information Loss Metrics (`information_loss.py`)**: KL divergence, token compression ratio, and semantic retention scoring.
- **Memory Alignment (`memory_alignment.py`)**: Measures intent alignment, persona consistency, and constraint adherence.
- **Confidence Calibration (`confidence_validation.py`)**: Expected Calibration Error (ECE) and Brier scoring.
- **Full Benchmark Evaluation Suite (`evaluation.py`)**: Comprehensive end-to-end evaluation harness.

---

## Directory Structure

```
eren/
├── backend/
│   ├── requirements.txt
│   └── context_memory/
│       ├── __init__.py
│       ├── main.py
│       ├── memory_store.py
│       ├── retrieval.py
│       ├── hnsw_index.py
│       ├── rrf.py
│       ├── mmr.py
│       ├── ranking.py
│       ├── knapsack.py
│       ├── minhash.py
│       ├── lsh.py
│       ├── delta_encoding.py
│       ├── pomdp.py
│       ├── bayesian_optimization.py
│       ├── information_loss.py
│       ├── memory_alignment.py
│       ├── confidence_validation.py
│       ├── evaluation.py
│       └── tests/
│           ├── __init__.py
│           ├── test_retrieval.py
│           ├── test_selection.py
│           ├── test_compression.py
│           ├── test_pomdp.py
│           └── test_evaluation.py
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.8+ (Zero external production dependencies; runs purely on Python standard library).
- `pytest` for test execution.

### Running Tests
```bash
cd backend
python -m pytest context_memory/tests/ -v
```

### Running Example Pipeline
```bash
Expected Outcome

A lightweight, modular Python framework that combines relevance-driven retrieval, redundancy-aware selection, token-budget optimization, and user-controlled memory dependence for efficient long-running AI agents.
cd backend
python -m context_memory.main
```
