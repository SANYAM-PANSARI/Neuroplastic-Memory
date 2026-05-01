# Engineering Analysis: Codebase Architecture

A technical breakdown of the NeuroPlastic Memory codebase — module structure, complexity mapping, and dependency analysis.

---

## Module Architecture

```
context_memory/
├── api.py                   # Public API surface (query, ingest, health, close)
├── config.py                # Pydantic Settings — reads from .env
├── types.py                 # Public types: ContextResult, IngestResult, HealthReport
├── agents/
│   └── llm.py               # LiteLLM wrappers (provider-agnostic, retries)
├── ingestion/
│   ├── chunker.py           # Markdown chunking (header-based + token fallback)
│   └── pipeline.py          # Ingest flow: chunk → embed → store → summarize → link
├── pathfinding/
│   ├── traversal.py         # Active Pathfinding: teleport → traverse → verify → cement
│   └── cache.py             # Semantic result cache (in-memory, cosine similarity)
└── storage/
    ├── schemas.py           # Pydantic schemas for all 3 stores
    ├── graph.py             # Neo4j async wrapper (nodes, edges, shortcuts)
    ├── vector.py            # Qdrant async wrapper (embeddings, similarity search)
    └── relational.py        # SQLite async wrapper (text payloads, provenance)
```

---

## Complexity Heatmap

The most complex components, ranked by implementation difficulty:

| Rank | File | Why |
|------|------|-----|
| 🔴 1 | `pathfinding/traversal.py` | Core algorithm — state machine, LLM interaction, backtracking, escape hatch. Most bugs live here. |
| 🔴 2 | `storage/graph.py` | Complex Cypher queries with versioning, quarantine, decay math, parameterized inputs. |
| 🟡 3 | `ingestion/pipeline.py` | Orchestrates chunking → hashing → embedding → 3-way write. Many failure modes. |
| 🟡 4 | `agents/llm.py` | Provider-agnostic LLM calls with retry logic, rate limiting, lazy imports. |
| 🟢 5+ | Everything else | Straightforward wrappers, configs, and API glue. |

---

## Dependency Map

```
context_memory
├── neo4j          (graph database driver)
├── qdrant-client  (vector database)
├── aiosqlite      (async SQLite)
├── pydantic       (schemas, validation)
├── pydantic-settings (env config)
└── litellm        (provider-agnostic LLM calls)
    └── supports: gemini, anthropic, openai, ollama, ...
```

**Total direct dependencies: 6** (intentionally minimal)

---

## Design Principles

1. **Cost gates** — Every query exits at the cheapest possible level (cache → shortcut → broad → traverse)
2. **Idempotent writes** — All Neo4j operations use `MERGE`, not `CREATE`. Safe to retry.
3. **Deterministic IDs** — UUIDv5 from content hash enables deduplication across ingestion runs.
4. **Provider agnostic** — All LLM calls route through LiteLLM. Switch providers by changing `.env`.
5. **Lazy initialization** — Database connections are created on first use with `asyncio.Lock` for concurrency safety.
6. **Verbatim retrieval** — The framework returns source text, never generated text. Anti-hallucination by design.

---

## Key Architecture Decisions

| Decision | Alternative Considered | Why This Approach |
|----------|----------------------|-------------------|
| Tri-storage (Neo4j + Qdrant + SQLite) | Single vector DB | Each DB handles what it's best at — graph structure, vector search, text storage |
| LiteLLM wrapper | Direct SDK calls | Provider lock-in is unacceptable for a pluggable module |
| Unified hop prompt | Separate evaluate + decide calls | Merged to 1 LLM call per hop — 50% cost reduction |
| Summarizer model for navigation | Pathfinder model for everything | Traversal decisions are usually obvious — no need for expensive models |
| Escape hatch reuses query vector | Re-embed on fallback | Saves 1 API call per escape hatch activation |
| Content-hash UUIDs | Random UUIDs | Enables deduplication — re-ingesting same content is a no-op |
