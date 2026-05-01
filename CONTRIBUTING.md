# Contributing to NeuroPlastic Memory

Thanks for your interest in contributing! This document covers setup, conventions, and how to run tests.

## Development Setup

### Prerequisites

- Python 3.10+
- Docker (for Neo4j and Qdrant)
- A Gemini API key (or any LiteLLM-compatible provider)

### Getting Started

```bash
# 1. Clone and install
git clone https://github.com/SANYAM-PANSARI/Neuroplastic-Memory.git
cd neuroplastic-memory
pip install -e ".[dev]"

# 2. Start databases
docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Configure
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### Running Tests

```bash
# Smoke test (verifies ingestion pipeline)
python -m tests.smoke_test

# Query test (single query against ingested data)
python -m tests.test_query

# Full integration test (multi-file ingest + 4 query types)
python -m tests.test_real_data
```

> **Note:** Tests require running Neo4j and Qdrant instances and a valid API key in `.env`.

## Code Conventions

### Architecture Rules

1. **All LLM calls go through `agents/llm.py`** — never call LiteLLM directly from other modules.
2. **All storage access goes through `storage/`** wrappers — never write raw Cypher, SQL, or Qdrant queries outside the storage layer.
3. **Node IDs are deterministic** — UUIDv5 from content hash. This enables deduplication.
4. **Use MERGE, not CREATE** — All Neo4j writes use MERGE for idempotency.

### Edge Types

| Type | Created By | Purpose |
|------|-----------|---------|
| `PARENT_CHILD` | Ingestion | Summary → chunk (structural) |
| `NEXT_IN_DOC` | Ingestion | Chunk → next chunk (reading order) |
| `SEMANTIC_LENS` | Queries | Learned shortcuts (neuroplasticity) |

### Style

- Type hints on all function signatures
- Docstrings on public functions
- `async/await` throughout — no blocking I/O in the main module

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Ensure all tests pass
5. Submit a pull request with a description of what changed and why
