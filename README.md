# imprint-memory

Persistent memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Gives Claude long-term memory that survives across conversations.

Built as an [MCP server](https://modelcontextprotocol.io/) — works locally (stdio) or remotely via HTTP with OAuth.

## Features

- **Hybrid search** — FTS5 full-text + vector embeddings + exact-match, fused with [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) ranking and time-decay scoring
- **CJK support** — Chinese/Japanese/Korean text is segmented with jieba for accurate full-text search
- **Memory CRUD** — store, search, update, delete memories with category/source/importance tags
- **Conversation search** — search logged conversations by keyword, filterable by platform
- **Knowledge bank** — drop `.md` files in `bank/`; they're auto-chunked, embedded, and searchable
- **Daily logs** — append-only daily journal
- **Message bus** — shared timeline across all sources
- **Task queue** — submit tasks for Claude Code to execute asynchronously (supports multi-turn sessions)
- **Context compression** — summarize old context lines with a local Ollama model, with truncation fallback

All data in a single SQLite database (WAL mode).

## Quick start

```bash
# Install
pip install git+https://github.com/Qizhan7/imprint-memory.git

# Register with Claude Code
claude mcp add -s user imprint-memory -- imprint-memory
```

Or clone locally:

```bash
git clone https://github.com/Qizhan7/imprint-memory.git
cd imprint-memory && pip install -e .
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a memory (category, source, importance) |
| `memory_search` | Hybrid search across all memories |
| `memory_list` | List recent memories |
| `memory_update` | Update a memory by ID |
| `memory_delete` | Delete a memory by ID |
| `memory_forget` | Delete memories matching a keyword |
| `memory_find_duplicates` | Find semantically similar pairs (dedup audit) |
| `memory_find_stale` | Find low-activity old memories |
| `memory_decay` | Reduce importance of inactive memories (dry-run by default) |
| `memory_reindex` | Rebuild all embeddings (after switching providers) |
| `memory_daily_log` | Append to today's log |
| `conversation_search` | Search conversation history (all platforms) |
| `search_telegram` | Search Telegram + heartbeat conversations |
| `search_channel` | Search any specific channel (discord, slack, etc.) |
| `message_bus_read` / `post` | Read/write the shared message bus |
| `cc_execute` | Submit a task for Claude Code |
| `cc_check` / `cc_tasks` | Check task status, list recent tasks |

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMPRINT_DATA_DIR` | `~/.imprint/` | Base directory for all data |
| `IMPRINT_DB` | `$IMPRINT_DATA_DIR/memory.db` | SQLite database path |
| `TZ_OFFSET` | `0` | Hours offset from UTC (e.g. `12` for NZST) |
| `EMBED_PROVIDER` | `ollama` | `ollama` or `openai` |
| `EMBED_MODEL` | auto | Model name (default: `bge-m3` / `text-embedding-3-small`) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OPENAI_API_KEY` | — | For OpenAI-compatible providers |
| `EMBED_API_BASE` | `https://api.openai.com` | Base URL for OpenAI-compatible API |

### Embedding providers

**Ollama (default)** — free, local:
```bash
ollama pull bge-m3 && ollama serve
```

**OpenAI API** — no local GPU:
```bash
export EMBED_PROVIDER=openai OPENAI_API_KEY=sk-...
```

**Any OpenAI-compatible API** (Voyage AI, Azure, etc.):
```bash
export EMBED_PROVIDER=openai OPENAI_API_KEY=... EMBED_API_BASE=https://... EMBED_MODEL=...
```

No embedding provider? Falls back to FTS5 keyword search only — still works, just less semantic.

After switching providers, run `memory_reindex` to rebuild embeddings.

## HTTP mode

For Claude.ai access through a tunnel:

```bash
pip install imprint-memory[http]
imprint-memory --http   # → http://0.0.0.0:8000/mcp
```

OAuth credentials via `~/.imprint-oauth.json` or env vars (`OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_ACCESS_TOKEN`).

## Data layout

```
~/.imprint/
├── memory.db           # SQLite (memories, vectors, tasks, bus)
├── MEMORY.md           # Auto-generated index
└── memory/
    ├── 2026-04-01.md   # Daily logs
    └── bank/           # Knowledge files (.md)
```

## Standalone vs Full Stack

**This package works on its own** — `pip install` and you get persistent memory in Claude Code. No other dependencies.

If you also want multi-channel messaging (Telegram, etc.), Claude.ai integration, heartbeat automation, a dashboard, and scheduled tasks, see the full system: [claude-imprint](https://github.com/Qizhan7/claude-imprint). It installs imprint-memory as a dependency.

## License

MIT
