# Security Audit: NeuroPlastic Memory

A threat model identifying 10 vulnerabilities across 4 attack surfaces, with mitigations.

---

## Attack Surface 1: The Public API

### 🔴 CRITICAL — No Authentication or Authorization

**The Problem:** `query()` and `ingest()` accept `agent_id` and `dataset_id` as plain strings with no verification. Any caller can impersonate any agent, query any dataset, or ingest poisoned data.

**Mitigation:** Token-based authentication with ACLs. `agent_id` is derived from a signed JWT, never self-reported. Each agent has an ACL defining which datasets it can access.

### 🔴 CRITICAL — Priority Queue Flooding (DoS)

**The Problem:** A malicious or buggy agent could flood the queue with high-priority requests, starving other agents.

**Mitigation:** Priority derived from agent ACL tier, not caller-specified. Rate limiting per `agent_id` with queue depth limits.

---

## Attack Surface 2: LLM Prompt Injection

### 🔴 CRITICAL — Ingested Data Can Contain Prompt Injections

**The Problem:** Malicious text in chunks could trick the Summarizer, Verifier, or Pathfinder LLM calls.

**Mitigation:**
- **Delimiter hardening** — All user data wrapped in XML-style `<data_begin>` / `<data_end>` tags in every prompt
- **Input sanitization** — Strip known prompt injection patterns before storage
- **Dual verifier** (future) — Two independent models must agree for verification

### 🟡 MEDIUM — Cypher Injection via LLM Output

**The Problem:** LLM-returned node IDs get used in graph queries. A hallucinated malicious string could corrupt the graph.

**Mitigation:** All Cypher queries use parameterized inputs (`$node_id`), never string interpolation. Returned IDs are validated against the current valid neighborhood set.

---

## Attack Surface 3: Graph Integrity

### 🔴 CRITICAL — Graph Poisoning via High-Confidence Edge Injection

**The Problem:** Carefully crafted ingested data could trigger traversals to malicious nodes, cementing poisoned shortcuts.

**Mitigation:**
- **Confidence ceiling** — Edge weight capped at 5.0
- **New edge quarantine** — Fresh edges get `quarantine: true`, forcing re-verification
- **Anomaly detection** — Flag edges with anomalous weight growth

### 🟡 MEDIUM — Glossary Manipulation

**The Problem:** An attacker controlling disproportionate ingested data could bias the Living Glossary.

**Mitigation:** Stratified sampling across `source_uri` groups. Glossary diffs emit telemetry events for audit.

---

## Attack Surface 4: Data Security

### 🟡 MEDIUM — No Encryption at Rest

**Mitigation:** Enable database-level encryption (Neo4j Enterprise, SQLCipher for SQLite, encrypted volumes for Qdrant).

### 🟡 MEDIUM — `source_uri` Leaks Internal Paths

**Mitigation:** Filter through agent ACLs. Optionally return `dataset_id:chunk_id` instead of raw filesystem paths.

### 🟢 LOW — Traversal Log Exposure

**Mitigation:** `traversal_metadata` only included for agents with `debug` ACL permissions.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| 🔴 CRITICAL | 4 | No Auth, DoS, Prompt Injection, Graph Poisoning |
| 🟡 MEDIUM | 5 | MCP auth, Cypher injection, Glossary manipulation, No encryption, Path leaks |
| 🟢 LOW | 1 | Traversal log exposure |

### The #1 Fix

**Implement token-based authentication with ACLs.** This single change resolves 6 of 10 vulnerabilities: agent impersonation, unauthorized access, priority flooding, HITL bypass, path leaks, and traversal log exposure.
