# imprint-memory

Persistent memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Gives Claude long-term memory that survives across conversations.

Built as an [MCP server](https://modelcontextprotocol.io/) — works with Claude Code locally (stdio) or remotely via HTTP with OAuth.

## What it does

- **Memory CRUD** — store, search, list, and delete memories
- **Hybrid search** — FTS5 keyword search + bge-m3 vector embeddings + time decay scoring
- **Daily logs** — append-only daily journal files
- **Message bus** — cross-channel message log (for multi-channel setups)
- **Task queue** — submit tasks for Claude Code to execute asynchronously
- **Bank files** — auto-indexes markdown files in a `bank/` directory for semantic search

All data stored in a single SQLite database with WAL mode for concurrent access.

## Quick start

### 1. Install

```bash
pip install git+https://github.com/Qizhan7/imprint-memory.git
```

Or clone and install locally:

```bash
git clone https://github.com/Qizhan7/imprint-memory.git
cd imprint-memory
pip install -e .
```

### 2. Register with Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "imprint-memory": {
      "command": "imprint-memory"
    }
  }
}
```

Or run directly:

```json
{
  "mcpServers": {
    "imprint-memory": {
      "command": "python3",
      "args": ["/path/to/imprint-memory/imprint_memory/server.py"]
    }
  }
}
```

### 3. Use

Claude Code will automatically have access to these tools:

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a memory (with category, source, importance) |
| `memory_search` | Hybrid search across all memories |
| `memory_forget` | Delete memories by keyword |
| `memory_list` | List recent memories |
| `memory_daily_log` | Append to today's log |
| `message_bus_read` | Read cross-channel message history |
| `message_bus_post` | Write to the message bus |
| `cc_execute` | Submit a task for Claude Code |
| `cc_check` | Check task status |
| `cc_tasks` | List recent tasks |

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMPRINT_DATA_DIR` | `~/.imprint/` | Base directory for all data |
| `IMPRINT_DB` | `$IMPRINT_DATA_DIR/memory.db` | SQLite database path |
| `TZ_OFFSET` | `0` | Hours offset from UTC (e.g., `12` for NZST) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint for embeddings |
| `EMBED_MODEL` | `bge-m3` | Embedding model name |
| `MESSAGE_BUS_LIMIT` | `40` | Max messages in the bus (rolling window) |

### Vector search (optional)

Vector search requires [Ollama](https://ollama.com/) running locally with the bge-m3 model:

```bash
ollama pull bge-m3
ollama serve
```

Without Ollama, the system falls back to FTS5 keyword search only — still functional, just less semantic.

### Bank files

Put markdown files in `$IMPRINT_DATA_DIR/memory/bank/` and they'll be automatically chunked, embedded, and included in search results. Useful for storing structured knowledge (preferences, relationships, experiences).

## HTTP mode (remote access)

For use with Claude.ai through a tunnel:

```bash
imprint-memory --http
# or
pip install imprint-memory[http]
imprint-memory --http
```

Serves on `http://0.0.0.0:8000/mcp` with OAuth 2.0 support.

Configure OAuth credentials via `~/.imprint-oauth.json`:

```json
{
  "client_id": "your-client-id",
  "client_secret": "your-client-secret",
  "access_token": "your-access-token"
}
```

Or via environment variables: `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_ACCESS_TOKEN`.

## Data storage

```
~/.imprint/
├── memory.db          # SQLite database (memories, vectors, tasks, bus)
├── MEMORY.md          # Auto-generated memory index
└── memory/
    ├── 2026-03-29.md  # Daily log files
    └── bank/
        ├── experience.md
        └── preferences.md
```

## Part of Claude Imprint

This is the core memory module extracted from [claude-imprint](https://github.com/Qizhan7/claude-imprint), a self-hosted AI agent system. It works standalone or as part of the full imprint stack.

## License

MIT
