# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root (if present)
- **`docs/AGENT_DECISION_TREE.md`** — routing index for authoritative engineering documents
- **`docs/PHASE_REGISTRY.md`** — phase status authority
- **`docs/adr/`** — architectural decision records (if present)

## File structure

Single-context repo:

```
/
├── docs/
│   ├── AGENT_DECISION_TREE.md
│   ├── PHASE_REGISTRY.md
│   ├── product.md
│   ├── TECH_STACK.md
│   └── adr/
└── src/redthread/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the terms defined in `docs/AGENT_ARCHITECTURE.md`, `docs/PHASE_REGISTRY.md`, and `docs/product.md`.
