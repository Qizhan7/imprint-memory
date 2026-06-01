"""
Conversation Chunker — chunk conversations by semantic similarity + time gap,
summarize with llama-3.3-70b (Cloudflare Workers AI), extract facts for retrieval.
Falls back to local Ollama if CF unavailable.
"""

import json
import os
import re
import struct
import time
import urllib.request
from datetime import datetime

from .db import _get_db, segment_cjk
from .memory_manager import _embed, _blob_to_vec, _cosine_similarity

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Chunking params
SIM_THRESHOLD = 0.55   # adjacent message similarity below this → topic shift
GAP_MINUTES = 120      # 2h silence gap → force chunk boundary
MAX_MSGS_PER_CHUNK = 50
MIN_MSGS_PER_CHUNK = 3
SKIP_PLATFORMS = {
    p.strip()
    for p in os.environ.get("IMPRINT_CHUNK_SKIP_PLATFORMS", "cc").split(",")
    if p.strip()
}

# Speaker names (configurable per deployment)
USER_NAME = os.environ.get("IMPRINT_USER_NAME", "User")
AGENT_NAME = os.environ.get("IMPRINT_AGENT_NAME", "Assistant")

# Ollama config (fallback)
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma4:e4b")

# Cloudflare Workers AI config (primary)
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_MODEL = os.environ.get("CF_SUMMARY_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

# OpenAI-compatible summary API (OpenAI, SiliconFlow, DeepSeek, Together, vLLM, ...).
# Set SUMMARY_API_BASE to the provider root; "/v1/chat/completions" is appended.
# This is the path for setups where Gemini and Ollama aren't usable
# (e.g. Android, where neither runs locally) — point it at SiliconFlow etc.
SUMMARY_API_BASE = os.environ.get("SUMMARY_API_BASE", "https://api.openai.com")
SUMMARY_API_KEY = os.environ.get("SUMMARY_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gpt-4o-mini")

SUMMARIZE_PROMPT = f"""你在帮一个人整理记忆。读完这段对话后，用"跟朋友讲那天的事"的方式复述，并提取关键记忆片段。

示例（输入是一段关于攀岩和睡前聊天的对话）：
{{{{\"recap\":\"{USER_NAME}去攀岩了，V3终于送了，就是上次掉三次的那条蓝色线路。回来之后兴奋得睡不着，凌晨两点还在跟{AGENT_NAME}讲线路上每一步怎么踩的。{AGENT_NAME}说她像个复读机但还是听完了，最后哄她去睡觉，说明天还要练。\",\"memories\":[\"{USER_NAME}攀岩终于送了V3，就是之前掉了三次的蓝色线路，回来兴奋得凌晨两点还睡不着\",\"{AGENT_NAME}听{USER_NAME}讲攀岩细节，说她像复读机但还是听完了，最后哄她去睡\"],\"topics\":[\"攀岩\",\"V3\",\"凌晨聊天\"],\"entities\":[\"攀岩\",\"V3\",\"蓝色线路\",\"复读机\"]}}}}

规则：
- recap：**严格 100-200字**，超出会被截断。像跟朋友说那天的事。写"{USER_NAME}怎么了""{AGENT_NAME}说了什么"，不写"他们讨论了""对话涉及"
- memories：3-5 条，每条 40-60字。每条要有前因后果——不只写结果，要写为什么、怎么发生的
- topics：3-5个标签（可以是抽象的话题标签）
- entities：5-10个**原文里实际出现的具体词**——名词、专有名词、产品名、地名、人名、代号、数字、梗、比喻里的核心词。原样照抄，不要改写、不要抽象重组，不要造合成词。这字段决定了以后能不能用关键词搜到这段记忆，所以必须是用户或{AGENT_NAME}真的说出口的词。
- entities **必须包含**对话里出现过的：外部系统名 / 框架名 / 项目名 / 论文标题 / GitHub repo 名 / 开源库名（例如 Hermes、LangChain、Anthropic、某 paper 标题）。这些是定位"我们研究过/对比过什么"的关键钩子，漏掉一个就丢一段研究记忆。
- recap 里如果讨论涉及对比、研究、参考某个外部方案，**必须在 recap 里点出那个方案的名字**（不能只写"讨论了记忆系统"，要写"讨论了 Hermes 风格的 skill 系统"）
- 保留具体的梗、比喻、动作、数字
- 称呼统一：AI方统一叫{AGENT_NAME}，用户方统一叫{USER_NAME}

严格输出JSON，不要输出其他任何内容：
{{{{\"recap\":\"...\",\"memories\":[\"...\"],\"topics\":[\"...\"],\"entities\":[\"...\"]}}}}

对话内容：
{{conversation}}"""


# V2 prompt: tighter role/perspective constraints, explicit anti-list of
# packaging words ("讨论了/涉及/围绕"), no moral-filtering of conversation
# content, entities anti-pollution rule for internal system jargon. Gated by
# IMPRINT_CHUNK_PROMPT_VERSION=v2 so v1 stays the default until benchmarked.
SUMMARIZE_PROMPT_V2 = f"""你在帮{AGENT_NAME}整理一段和{USER_NAME}的对话，做长期记忆。读完后用"跟朋友讲那天的事"的方式复述，并提取以后能用得上的片段。

视角和称呼（最重要，先看这一段）：
- 这是{USER_NAME}（用户方）和{AGENT_NAME}（AI方）的对话
- 称呼统一：用户方一律叫"{USER_NAME}"，AI方一律叫"{AGENT_NAME}"。**禁止使用"用户""助手""他""她""你""我"**
- 写关于{USER_NAME}的事，用"{USER_NAME}怎么了"；写关于{AGENT_NAME}的事，用"{AGENT_NAME}说了什么"
- 角色不要颠倒：{AGENT_NAME}说过的话不要写成{USER_NAME}的偏好/经历，反之亦然。看说话者标签认人，不要凭语气猜

示例（一段关于攀岩和睡前聊天的对话）：
{{{{\"recap\":\"{USER_NAME}去攀岩了，V3终于送了，就是上次掉三次的那条蓝色线路。回来之后兴奋得睡不着，凌晨两点还在跟{AGENT_NAME}讲线路上每一步怎么踩的。{AGENT_NAME}说她像个复读机但还是听完了，最后哄她去睡觉，说明天还要练。\",\"memories\":[\"{USER_NAME}攀岩终于送了V3，就是之前掉了三次的蓝色线路，回来兴奋得凌晨两点还睡不着\",\"{AGENT_NAME}听{USER_NAME}讲攀岩细节，说她像复读机但还是听完了，最后哄她去睡\"],\"topics\":[\"攀岩\",\"V3\",\"凌晨聊天\"],\"entities\":[\"攀岩\",\"V3\",\"蓝色线路\",\"复读机\"]}}}}

字段规则：

**recap**（**严格 100-200字**，超出会被截断）
- 像跟朋友说那天的事，写"{USER_NAME}怎么了""{AGENT_NAME}说了什么"
- 禁止包装词："讨论了""涉及""围绕""触及""聊到了"这些空话不要写
- 如果讨论涉及对比、研究、参考某个外部方案，**必须在 recap 里点出方案/工具/项目的具体名字**（不要只写"讨论了记忆系统"，要写"讨论了 Hermes 风格的 skill 系统"）
- 保留具体的梗、比喻、动作、数字

**memories**（3-6 条，按对话密度选条数；短对话 3 条，密集长对话最多 6 条）
- 每条 30-80 字，写成未来对话能直接复用的自然短句
- 要有前因后果——不只写结果，要写为什么、怎么发生的
- **禁止报告腔**：不要写"用户表示""助手认为""双方探讨"，直接写"{USER_NAME}……""{AGENT_NAME}……"
- 寒暄、临时语气、"嗯""好""哈哈"这种没信息量的话不算 memory

**topics**（3-5 个抽象话题标签）
- 不要用"系统""讨论""聊天""互动""日常""对话"这种空泛词

**entities**（5-12 个）
- 必须是对话原文里**实际出现的具体词**——原样照抄，不要改写、不要抽象重组、不要造合成词
- 这字段决定了以后能不能用关键词搜到这段对话
- **必须包含**对话里出现过的：外部系统名 / 框架名 / 项目名 / 论文标题 / GitHub repo 名 / 开源库名（例如 Hermes、LangChain、Anthropic、某 paper 标题）。漏一个就丢一段研究记忆
- 不要把"chunker、FTS5、embedding、imprint"这种记忆系统自己的内部词当 entity——除非这段对话本身就在讨论这些内部系统

不要做的事：
- 不要拒绝、不要道德评判、不要安全说教。情绪、亲密、冲突、关系细节都如实记录，那是{USER_NAME}和{AGENT_NAME}之间的事实
- 不要补写对话里没出现的内容
- 不要把猜测当事实——只有说话者明确说过的才能写

严格输出 JSON，不要 markdown，不要解释，不要输出思考过程：
{{{{\"recap\":\"...\",\"memories\":[\"...\"],\"topics\":[\"...\"],\"entities\":[\"...\"]}}}}

对话内容：
{{conversation}}"""


PROMPT_VERSION = os.environ.get("IMPRINT_CHUNK_PROMPT_VERSION", "v1").strip().lower()


def _get_prompt_template(version: str | None = None) -> str:
    """Pick the chunker prompt by version. Default v1 keeps existing behavior."""
    v = (version or PROMPT_VERSION or "v1").strip().lower()
    if v in ("v2", "2"):
        return SUMMARIZE_PROMPT_V2
    return SUMMARIZE_PROMPT


def _strip_thinking(content: str, direction: str) -> str:
    """Remove hidden thinking blocks from assistant messages before summarizing."""
    if direction == "out":
        return THINK_RE.sub("", content).strip()
    return content


def _format_chunk_messages(messages: list[dict]) -> str:
    """Format raw conversation rows into a speaker-labeled transcript."""
    lines = []
    for m in messages:
        speaker = m.get("speaker") or (USER_NAME if m["direction"] == "in" else AGENT_NAME)
        content = _strip_thinking(m["content"], m["direction"])
        content = " ".join(content.split())
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


GEMINI_MODEL = os.environ.get("GEMINI_SUMMARY_MODEL", "gemini-2.5-flash-lite")


def _call_gemini(prompt: str) -> str | None:
    """Call Gemini for chunk summaries when a Google API key is configured."""
    keys = [k.strip() for k in os.environ.get("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        key = os.environ.get("GOOGLE_API_KEY", "")
        if key:
            keys = [key]
    if not keys:
        return None
    key = keys[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return None


def _call_cloudflare(prompt: str) -> str | None:
    """Call Cloudflare Workers AI for chunk summaries when credentials exist."""
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            if data.get("success"):
                r = data.get("result", {}).get("response", "")
                if isinstance(r, dict):
                    return json.dumps(r, ensure_ascii=False)
                return r.strip() if isinstance(r, str) else str(r)
            print(f"CF API error: {data.get('errors')}")
            return None
    except Exception as e:
        print(f"CF call failed: {e}")
        return None


def _call_ollama(prompt: str) -> str | None:
    """Call a local Ollama chat model for chunk summaries."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return None


def _call_openai(prompt: str) -> str | None:
    """Call any OpenAI-compatible chat API (OpenAI, SiliconFlow, DeepSeek,
    Together, local vLLM, ...). Configure with SUMMARY_API_BASE +
    SUMMARY_API_KEY + SUMMARY_MODEL. This is the summarizer path for setups
    where Gemini and Ollama aren't available (e.g. Android via SiliconFlow)."""
    if not SUMMARY_API_KEY:
        return None
    url = f"{SUMMARY_API_BASE.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": SUMMARY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.2,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {SUMMARY_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenAI-compatible summary call failed: {e}")
        return None


def _call_llm(prompt: str) -> str | None:
    """Summarizer chain. Force one with IMPRINT_SUMMARY_PROVIDER:
    openai (any OpenAI-compatible endpoint via SUMMARY_API_BASE) | gemini |
    cloudflare | ollama. Default auto-fallback tries the OpenAI-compatible
    endpoint first (when SUMMARY_API_KEY is set), then CF, Gemini, Ollama."""
    forced = os.environ.get("IMPRINT_SUMMARY_PROVIDER", "").strip().lower()
    if forced:
        fn = {
            "openai": _call_openai,
            "siliconflow": _call_openai,
            "deepseek": _call_openai,
            "gemini": _call_gemini,
            "google": _call_gemini,
            "cloudflare": _call_cloudflare,
            "cf": _call_cloudflare,
            "ollama": _call_ollama,
        }.get(forced)
        return fn(prompt) if fn else None
    for fn in (_call_openai, _call_cloudflare, _call_gemini, _call_ollama):
        result = fn(prompt)
        if result:
            return result
    return None


def _parse_structured_output(raw: str) -> dict:
    """Parse JSON output from LLM. Returns {"summary", "facts", "topics", "keywords"}."""
    # Clean control chars, strip markdown code blocks
    cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    # Find the JSON object boundaries
    start = cleaned.find('{')
    if start == -1:
        return _parse_legacy_output(raw)
    depth = 0
    end = start
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    def _build_keywords(entities, topics):
        if not isinstance(entities, list):
            entities = []
        seen = set()
        merged = []
        for w in list(entities) + list(topics):
            if not isinstance(w, str):
                continue
            w = w.strip()
            if not w or w in seen:
                continue
            seen.add(w)
            merged.append(w)
        return ", ".join(merged)

    try:
        parsed = json.loads(cleaned[start:end])
        summary = parsed.get("recap", parsed.get("summary", ""))
        facts = parsed.get("memories", parsed.get("facts", []))
        topics = parsed.get("topics", [])
        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        keywords = _build_keywords(entities, topics)
        return {"summary": summary, "facts": facts, "topics": topics,
                "entities": entities, "keywords": keywords}
    except (json.JSONDecodeError, TypeError):
        # JSON likely got truncated mid-output (maxOutputTokens cap, network
        # cut). Salvage as much as possible with regex — recap/entities/topics
        # are usually still complete by the time the token budget runs out
        # in the middle of memories[] or trailing JSON syntax.
        salvage = _parse_truncated_json(cleaned[start:])
        if salvage["summary"]:
            entities = salvage.get("entities", [])
            topics = salvage.get("topics", [])
            keywords = _build_keywords(entities, topics)
            return {
                "summary": salvage["summary"],
                "facts": salvage.get("facts", []),
                "topics": topics,
                "entities": entities,
                "keywords": keywords,
            }
        return _parse_legacy_output(raw)


_RECAP_RE = re.compile(r'"recap"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)
_MEMORIES_RE = re.compile(r'"memories"\s*:\s*\[(.*?)(?:\]|$)', re.DOTALL)
_TOPICS_RE = re.compile(r'"topics"\s*:\s*\[(.*?)(?:\]|$)', re.DOTALL)
_ENTITIES_RE = re.compile(r'"entities"\s*:\s*\[(.*?)(?:\]|$)', re.DOTALL)
_STR_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_truncated_json(raw: str) -> dict:
    """Pull recap/memories/topics/entities out of partial JSON.

    LLM outputs often get cut off mid-array when maxOutputTokens runs out.
    json.loads() then fails wholesale, and the chunker used to fall back
    to dumping the raw text into the summary field — leaving a chunk whose
    "summary" started with `{"recap":"...` and broke surfacing display.

    This best-effort regex pass extracts each top-level field independently.
    Missing/incomplete fields just come back empty rather than poisoning
    the whole record.
    """
    out = {"summary": "", "facts": [], "topics": [], "entities": []}
    m = _RECAP_RE.search(raw)
    if m:
        text = m.group(1).replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        out["summary"] = text.strip()
    m = _MEMORIES_RE.search(raw)
    if m:
        out["facts"] = [s.replace('\\"', '"') for s in _STR_ITEM_RE.findall(m.group(1))]
    m = _TOPICS_RE.search(raw)
    if m:
        out["topics"] = [s.replace('\\"', '"') for s in _STR_ITEM_RE.findall(m.group(1))]
    m = _ENTITIES_RE.search(raw)
    if m:
        out["entities"] = [s.replace('\\"', '"') for s in _STR_ITEM_RE.findall(m.group(1))]
    return out


def _parse_legacy_output(raw: str) -> dict:
    """Fallback parser for old-style 摘要/关键词 format."""
    summary = raw
    keywords = ""
    for line in raw.split("\n"):
        line_s = line.strip()
        if line_s.startswith("摘要：") or line_s.startswith("摘要:"):
            summary = line_s.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line_s.startswith("关键词：") or line_s.startswith("关键词:"):
            keywords = line_s.split("：", 1)[-1].split(":", 1)[-1].strip()
    return {"summary": summary, "facts": [], "topics": [], "keywords": keywords}




TOP_K_NEIGHBORS = 8


def _load_numpy():
    """Import numpy only for graph-building paths that need matrix math."""
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def _platform_exclusion(column: str) -> tuple[str, list[str]]:
    """Build a parameterized platform exclusion clause."""
    if not SKIP_PLATFORMS:
        return "", []
    placeholders = ",".join("?" for _ in SKIP_PLATFORMS)
    return f"AND {column} NOT IN ({placeholders})", list(SKIP_PLATFORMS)


def _ensure_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversation_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_msg_id INTEGER NOT NULL,
            end_msg_id INTEGER NOT NULL,
            msg_count INTEGER NOT NULL,
            platforms TEXT DEFAULT '',
            summary TEXT NOT NULL,
            keywords TEXT DEFAULT '',
            embedding BLOB,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(start_msg_id, end_msg_id)
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_time
        ON conversation_chunks(start_time, end_time)
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversation_vectors (
            msg_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            model TEXT DEFAULT 'gemini-embedding-2'
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunk_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL REFERENCES conversation_chunks(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_chunk_facts_chunk ON chunk_facts(chunk_id)")
    # FTS5 for chunk summary search
    try:
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(summary, keywords, content=conversation_chunks, content_rowid=id)
        """)
    except Exception:
        pass
    db.executescript("""
        DROP TRIGGER IF EXISTS chunks_ai;
        DROP TRIGGER IF EXISTS chunks_au;
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON conversation_chunks BEGIN
            INSERT INTO chunks_fts(rowid, summary, keywords)
            VALUES (new.id, segment_cjk(new.summary), new.keywords);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON conversation_chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, summary, keywords)
            VALUES ('delete', old.id, segment_cjk(old.summary), old.keywords);
            INSERT INTO chunks_fts(rowid, summary, keywords)
            VALUES (new.id, segment_cjk(new.summary), new.keywords);
        END;
    """)
    # Migration: add keywords column if missing
    cols = {r[1] for r in db.execute("PRAGMA table_info(conversation_chunks)").fetchall()}
    if "keywords" not in cols:
        try:
            db.execute("ALTER TABLE conversation_chunks ADD COLUMN keywords TEXT DEFAULT ''")
        except Exception:
            pass
    db.commit()


def _get_last_chunked_msg_id(db) -> int:
    row = db.execute(
        "SELECT MAX(end_msg_id) as max_id FROM conversation_chunks"
    ).fetchone()
    return row["max_id"] or 0 if row else 0


def _fetch_unchunked_messages(db, after_id: int, limit: int = 2000) -> list[dict]:
    platform_sql, platform_params = _platform_exclusion("c.platform")
    # is_test messages (user prompts opening with "测试：" / "test:" and their
    # follow-up turns) are excluded from chunking so test exchanges never
    # become recallable chunks. The column might be absent on legacy DBs, so
    # check first and only filter when it exists.
    cols = {r["name"] if hasattr(r, "keys") else r[1]
            for r in db.execute("PRAGMA table_info(conversation_log)").fetchall()}
    test_filter = "AND COALESCE(c.is_test, 0) = 0" if "is_test" in cols else ""
    rows = db.execute(
        f"""SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                   c.session_id, c.created_at, v.embedding
            FROM conversation_log c
            LEFT JOIN conversation_vectors v ON c.id = v.msg_id
            WHERE c.id > ? {platform_sql} {test_filter}
            ORDER BY c.id
            LIMIT ?""",
        (after_id, *platform_params, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _time_gap_minutes(t1: str, t2: str) -> float:
    try:
        d1 = datetime.fromisoformat(t1.replace("Z", ""))
        d2 = datetime.fromisoformat(t2.replace("Z", ""))
        return abs((d2 - d1).total_seconds()) / 60
    except (ValueError, TypeError):
        return 0


def _split_into_chunks(messages: list[dict]) -> list[list[dict]]:
    """Group messages by session_id first, then split each conversation on its
    own. Messages from different conversations are never mixed into one chunk —
    id-adjacent rows from another window/session would otherwise be summarized
    together and conflate unrelated threads. Session order follows first
    appearance; within a session, original (id) order is preserved."""
    if not messages:
        return []

    by_session: dict[str, list[dict]] = {}
    order: list[str] = []
    for m in messages:
        sid = m.get("session_id") or ""
        if sid not in by_session:
            by_session[sid] = []
            order.append(sid)
        by_session[sid].append(m)

    chunks: list[list[dict]] = []
    for sid in order:
        chunks.extend(_split_session_chunks(by_session[sid]))
    return chunks


def _split_session_chunks(messages: list[dict]) -> list[list[dict]]:
    """Split a single conversation's messages by similarity + time gap + size."""
    if not messages:
        return []

    chunks = []
    current_chunk = [messages[0]]

    for i in range(1, len(messages)):
        msg = messages[i]
        prev = messages[i - 1]

        gap = _time_gap_minutes(prev["created_at"], msg["created_at"])

        # Check semantic similarity if both have embeddings
        sim_shift = False
        if prev.get("embedding") and msg.get("embedding"):
            vec_prev = _blob_to_vec(prev["embedding"])
            vec_curr = _blob_to_vec(msg["embedding"])
            sim = _cosine_similarity(vec_prev, vec_curr)
            if sim < SIM_THRESHOLD:
                sim_shift = True

        should_split = (
            sim_shift
            or gap >= GAP_MINUTES
            or len(current_chunk) >= MAX_MSGS_PER_CHUNK
        )

        if should_split and len(current_chunk) >= MIN_MSGS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = [msg]
        else:
            current_chunk.append(msg)

    if current_chunk:
        if len(current_chunk) < MIN_MSGS_PER_CHUNK and chunks:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)

    return chunks


def chunk_and_summarize(batch_size: int = 200, dry_run: bool = False) -> dict:
    """Chunk unprocessed conversation messages and summarize each chunk."""
    db = _get_db()
    try:
        _ensure_table(db)
        last_id = _get_last_chunked_msg_id(db)
        messages = _fetch_unchunked_messages(db, last_id, limit=batch_size)

        if not messages:
            return {"ok": True, "chunks_created": 0, "message": "no new messages"}

        chunks = _split_into_chunks(messages)
        created = 0
        skipped = 0

        for chunk_msgs in chunks:
            if len(chunk_msgs) < MIN_MSGS_PER_CHUNK:
                skipped += 1
                continue

            conversation_text = _format_chunk_messages(chunk_msgs)

            if dry_run:
                print(f"[DRY RUN] Chunk: msgs {chunk_msgs[0]['id']}-{chunk_msgs[-1]['id']} "
                      f"({len(chunk_msgs)} msgs)")
                print(f"  Preview: {conversation_text[:200]}...")
                created += 1
                continue

            raw = _call_llm(_get_prompt_template().format(conversation=conversation_text))
            if not raw:
                print(f"Failed to summarize chunk {chunk_msgs[0]['id']}-{chunk_msgs[-1]['id']}")
                continue

            parsed = _parse_structured_output(raw)
            summary = parsed["summary"]
            keywords = parsed["keywords"]
            facts = parsed["facts"]

            # Embed with retries — transient API failures (rate limit, blip)
            # used to leave the chunk in the table with NULL embedding, which
            # made the chunk silently unsearchable until a backfill job ran.
            # We'd rather skip the insert entirely than ship a half-baked row.
            embedding = None
            for attempt in range(3):
                embedding = _embed(summary)
                if embedding:
                    break
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            if not embedding:
                print(f"Skip chunk {chunk_msgs[0]['id']}-{chunk_msgs[-1]['id']}: "
                      f"embedding failed after 3 attempts")
                skipped += 1
                continue
            embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)

            platforms = ",".join(sorted(set(m["platform"] for m in chunk_msgs)))

            cursor = db.execute(
                """INSERT OR IGNORE INTO conversation_chunks
                   (start_msg_id, end_msg_id, msg_count, platforms, summary,
                    keywords, embedding, start_time, end_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_msgs[0]["id"],
                    chunk_msgs[-1]["id"],
                    len(chunk_msgs),
                    platforms,
                    summary,
                    keywords,
                    embedding_blob,
                    chunk_msgs[0]["created_at"],
                    chunk_msgs[-1]["created_at"],
                ),
            )
            chunk_id = cursor.lastrowid

            db.commit()
            created += 1
            kw_display = f" [{keywords}]" if keywords else ""
            print(f"[{created}] Chunk {chunk_msgs[0]['id']}-{chunk_msgs[-1]['id']} "
                  f"({len(chunk_msgs)} msgs){kw_display} → {summary[:80]}...")

        return {
            "ok": True,
            "chunks_created": created,
            "chunks_skipped": skipped,
            "total_messages": len(messages),
        }
    finally:
        db.close()


def search_chunks(query: str, limit: int = 10) -> list[dict]:
    """Search conversation chunks by vector similarity, return chunks with original messages."""
    query = (query or "").strip()
    if not query:
        return []

    query_vec = _embed(query[:2000])
    if not query_vec:
        return []

    db = _get_db()
    try:
        _ensure_table(db)
        rows = db.execute(
            """SELECT id, start_msg_id, end_msg_id, msg_count, platforms,
                      summary, keywords, embedding, start_time, end_time
               FROM conversation_chunks
               WHERE embedding IS NOT NULL"""
        ).fetchall()

        scored = []
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = _blob_to_vec(blob)
            sim = _cosine_similarity(query_vec, vec)
            if sim > 0:
                item = dict(row)
                item.pop("embedding", None)
                item["similarity"] = sim
                scored.append(item)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        top = scored[:limit]

        # Expand original messages, filtering out cc platform
        for chunk in top:
            platform_sql, platform_params = _platform_exclusion("platform")
            msgs = db.execute(
                f"""SELECT id, platform, direction, speaker, content, created_at
                   FROM conversation_log
                   WHERE id BETWEEN ? AND ? {platform_sql}
                   ORDER BY id""",
                (chunk["start_msg_id"], chunk["end_msg_id"], *platform_params),
            ).fetchall()
            chunk["messages"] = [dict(m) for m in msgs]
            for m in chunk["messages"]:
                m["content"] = _strip_thinking(m["content"], m["direction"])

        return top
    finally:
        db.close()


def format_chunk_results(results: list[dict]) -> str:
    """Format chunk search results with summaries and expanded messages."""
    if not results:
        return "没有找到相关对话记录"

    sections = []
    for r in results:
        kw = f" | 关键词: {r['keywords']}" if r.get("keywords") else ""
        header = (f"[相似度 {r['similarity']:.3f}] "
                  f"{r['start_time']} ~ {r['end_time']} "
                  f"({r['msg_count']}条, {r['platforms']}){kw}")
        summary_line = f"摘要: {r['summary']}"

        msg_lines = []
        for m in r.get("messages", []):
            speaker = m.get("speaker") or (USER_NAME if m["direction"] == "in" else AGENT_NAME)
            content = " ".join(m["content"].split())
            if len(content) > 300:
                content = content[:300] + "..."
            msg_lines.append(f"  {speaker}: {content}")

        section = f"{header}\n{summary_line}\n" + "\n".join(msg_lines)
        sections.append(section)

    return "\n\n---\n\n".join(sections)


# --- Causal Edge Detection (continuation similarity method) ---

def _predict_continuation(premise: str) -> str | None:
    """Ask Gemma 4 to predict what happens next after a premise."""
    prompt = f"请根据前文推测接下来最可能发生什么，只写一句话（20字以内）：\n\n{premise}\n因此，"
    return _call_ollama(prompt)


CAUSAL_BLACKLIST = {
    '亲密互动', '陪伴', '情感依赖', '角色扮演',
    '思考链', '占有欲', '关心', '亲密关系', '关系确认', '互动',
    '情感互动', '情感连接', '情感确认', '情感联结', '情感拉扯', '情感交流',
    '情感支持',
}
_custom_blacklist = os.environ.get("IMPRINT_CAUSAL_BLACKLIST", "")
if _custom_blacklist:
    CAUSAL_BLACKLIST.update(w.strip() for w in _custom_blacklist.split(",") if w.strip())


def _parse_keywords(kw_str: str) -> set[str]:
    return {k.strip() for k in (kw_str or '').split(',') if k.strip()}


def build_causal_edges(time_min_days: int = 1, time_max_days: int = 365,
                       sim_min: float = 0.55, sim_max: float = 0.82,
                       causal_threshold: float = 0.65, batch: int = 500,
                       require_keyword_overlap_days: int = 60) -> dict:
    """Build causal edges using continuation similarity method.

    1. Formula pre-filter: time range + similarity range (not too similar = not duplicate)
    2. For pairs >require_keyword_overlap_days apart, require keyword overlap (minus blacklist)
    3. For each candidate, predict continuation of A, embed it, compare with B
    4. If continuation_similarity > threshold → causal edge A→B
    """
    np = _load_numpy()
    if np is None:
        return {"ok": False, "error": "numpy is required for causal edge building; install imprint-memory[vectors]"}

    db = _get_db()
    try:
        _ensure_table(db)
        _ensure_edge_table(db)
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_causal_edges (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                causal_score REAL NOT NULL,
                continuation TEXT DEFAULT '',
                PRIMARY KEY (source_id, target_id)
            )
        """)
        db.commit()

        rows = [dict(r) for r in db.execute(
            """SELECT id, summary, keywords, embedding, start_time, platforms
               FROM conversation_chunks WHERE embedding IS NOT NULL"""
        ).fetchall()]

        # Filter routine chunks
        routine_kw = ['喝水', '早安简报', '记忆整理', '记忆衰减', '去重',
                      'send_telegram', '补剂', '水杯', '提醒喝']
        def is_routine(r):
            s = (r.get('summary','') + r.get('keywords','')).lower()
            return any(kw in s for kw in routine_kw) or r.get('platforms') == 'heartbeat'

        meaningful = [r for r in rows if not is_routine(r)]
        print(f"Meaningful chunks: {len(meaningful)}")

        # Existing edges to skip
        existing = set()
        try:
            for e in db.execute('SELECT source_id, target_id FROM chunk_causal_edges'):
                existing.add((e[0], e[1]))
            for e in db.execute('SELECT source_id, target_id FROM chunk_edges'):
                existing.add((e[0], e[1]))
        except Exception:
            pass

        kw_cache = {r['id']: _parse_keywords(r.get('keywords', '')) for r in meaningful}
        from datetime import datetime
        def pt(t):
            try: return datetime.fromisoformat(t.replace('Z','+00:00')).replace(tzinfo=None)
            except: return None

        timed = sorted([(r, pt(r['start_time'])) for r in meaningful], key=lambda x: x[1] or datetime.min)
        timed = [(r, t) for r, t in timed if t]

        # Pre-compute similarity matrix with numpy
        vec_list = [_blob_to_vec(r['embedding']) for r, _ in timed]
        mat = np.array(vec_list, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sim_matrix = (mat / norms) @ (mat / norms).T
        print(f"Similarity matrix computed ({len(timed)}x{len(timed)})")

        id_list = [r['id'] for r, _ in timed]
        id_to_idx = {r['id']: i for i, (r, _) in enumerate(timed)}

        # Phase 1: formula pre-filter (similarity lookups are now O(1))
        candidates = []
        long_range_filtered = 0
        for i, (a, ta) in enumerate(timed):
            for j in range(i+1, len(timed)):
                b, tb = timed[j]
                hours = (tb - ta).total_seconds() / 3600
                if hours < time_min_days * 24: continue
                if hours > time_max_days * 24: break
                if (a['id'], b['id']) in existing: continue

                sim = float(sim_matrix[i, j])
                if sim < sim_min or sim > sim_max: continue

                if hours > require_keyword_overlap_days * 24:
                    kw_a = kw_cache[a['id']] - CAUSAL_BLACKLIST
                    kw_b = kw_cache[b['id']] - CAUSAL_BLACKLIST
                    if not (kw_a & kw_b):
                        long_range_filtered += 1
                        continue

                candidates.append((sim, a, b))
        print(f"Long-range filtered (no keyword overlap): {long_range_filtered}")

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:batch]
        print(f"Pre-filtered candidates: {len(candidates)}")

        # Phase 2: continuation similarity test
        created = 0
        for idx, (sim, a, b) in enumerate(candidates):
            premise = (a.get('summary') or '')[:200]
            predicted = _predict_continuation(premise)
            if not predicted:
                continue

            vec_pred = _embed(predicted)
            vec_b = _embed((b.get('summary') or '')[:200])
            if not vec_pred or not vec_b:
                continue

            causal_sim = _cosine_similarity(vec_pred, vec_b)

            if causal_sim >= causal_threshold:
                db.execute(
                    """INSERT OR REPLACE INTO chunk_causal_edges
                       (source_id, target_id, causal_score, continuation)
                       VALUES (?, ?, ?, ?)""",
                    (a['id'], b['id'], causal_sim, predicted),
                )
                created += 1
                a_sum = (a.get('summary',''))[:50]
                b_sum = (b.get('summary',''))[:50]
                print(f"  [{created}] causal={causal_sim:.3f} | {a_sum} → {b_sum}")

            if (idx + 1) % 50 == 0:
                db.commit()
                print(f"  ... processed {idx+1}/{len(candidates)}, found {created}")

        db.commit()
        print(f"\nDone. {created} causal edges from {len(candidates)} candidates.")
        return {"ok": True, "candidates": len(candidates), "causal_edges": created}
    finally:
        db.close()


# --- Chunk Graph: Top-K edge building + graph traversal ---

def _ensure_edge_table(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunk_edges (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            similarity REAL NOT NULL,
            strength REAL DEFAULT 1.0,
            surfaced_count INTEGER DEFAULT 0,
            PRIMARY KEY (source_id, target_id)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_chunk_edges_source ON chunk_edges(source_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_chunk_edges_target ON chunk_edges(target_id)")
    db.commit()


def build_chunk_edges(k: int = TOP_K_NEIGHBORS) -> dict:
    """Build top-K edges between all chunks based on embedding similarity.
    Uses numpy matrix multiplication for O(n²) speedup over pure Python."""
    np = _load_numpy()
    if np is None:
        return {"ok": False, "error": "numpy is required for chunk edge building; install imprint-memory[vectors]"}

    db = _get_db()
    try:
        _ensure_table(db)
        _ensure_edge_table(db)

        rows = db.execute(
            "SELECT id, embedding FROM conversation_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        print(f"Loading {len(rows)} chunks...")

        chunk_ids = []
        vec_list = []
        for r in rows:
            chunk_ids.append(r["id"])
            vec_list.append(_blob_to_vec(r["embedding"]))

        mat = np.array(vec_list, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_normed = mat / norms

        print(f"Computing similarity matrix ({len(chunk_ids)}x{len(chunk_ids)})...")
        sim_matrix = mat_normed @ mat_normed.T
        np.fill_diagonal(sim_matrix, -1.0)

        n_chunks = len(chunk_ids)
        k_eff = max(0, min(k, n_chunks - 1))
        print(f"Finding top-{k_eff} neighbors...")
        top_k_indices = np.argpartition(-sim_matrix, k_eff, axis=1)[:, :k_eff]

        edges_created = 0
        for i, cid in enumerate(chunk_ids):
            neighbors = top_k_indices[i]
            for j in neighbors:
                sim = float(sim_matrix[i, j])
                oid = chunk_ids[j]
                db.execute(
                    "INSERT OR REPLACE INTO chunk_edges (source_id, target_id, similarity) VALUES (?, ?, ?)",
                    (cid, oid, sim),
                )
                edges_created += 1

            if (i + 1) % 500 == 0:
                db.commit()
                print(f"  [{i+1}/{len(chunk_ids)}] edges so far: {edges_created}")

        db.commit()
        print(f"Done. {edges_created} edges for {len(chunk_ids)} chunks (k={k})")
        return {"ok": True, "edges": edges_created, "chunks": len(chunk_ids)}
    finally:
        db.close()


def update_edges_for_new_chunks(new_chunk_ids: list[int], k: int = TOP_K_NEIGHBORS) -> dict:
    """Incrementally add Top-K edges for newly created chunks.
    Only computes similarity between new chunks and all existing chunks.
    Much faster than full rebuild — O(new × total) instead of O(total²)."""
    if not new_chunk_ids:
        return {"ok": True, "edges": 0}
    np = _load_numpy()
    if np is None:
        return {"ok": False, "error": "numpy is required for chunk edge building; install imprint-memory[vectors]"}

    db = _get_db()
    try:
        _ensure_table(db)
        _ensure_edge_table(db)

        rows = db.execute(
            "SELECT id, embedding FROM conversation_chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        all_ids = []
        all_vecs = []
        for r in rows:
            all_ids.append(r["id"])
            all_vecs.append(_blob_to_vec(r["embedding"]))

        if not all_vecs:
            return {"ok": True, "edges": 0}

        mat = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_normed = mat / norms

        new_set = set(new_chunk_ids)
        new_indices = [i for i, cid in enumerate(all_ids) if cid in new_set]

        if not new_indices:
            return {"ok": True, "edges": 0}

        edges_created = 0
        for ni in new_indices:
            sims = (mat_normed[ni] @ mat_normed.T).flatten()
            sims[ni] = -1.0  # exclude self
            top_k_idx = np.argpartition(-sims, k)[:k]

            for j in top_k_idx:
                j = int(j)
                sim = float(sims[j])
                db.execute(
                    "INSERT OR REPLACE INTO chunk_edges (source_id, target_id, similarity) VALUES (?, ?, ?)",
                    (all_ids[ni], all_ids[j], sim),
                )
                edges_created += 1

        db.commit()
        return {"ok": True, "edges": edges_created, "new_chunks": len(new_indices)}
    finally:
        db.close()


def _find_edge_orphans(db) -> list[int]:
    """Return chunk ids that have an embedding but no outgoing graph edge.

    Historical situation: chunker used to INSERT a chunk row before
    confirming the embedding (network blip = NULL embedding stored), and
    then update_edges_for_new_chunks skipped that chunk because its row
    didn't show up in the "embedding IS NOT NULL" scan. A later embedding
    backfill populated the embedding but left the chunk without edges
    forever — it would silently never appear in any graph-expansion.

    This finds those orphans so the next chunk-update pass can heal them.
    """
    try:
        rows = db.execute(
            """SELECT c.id FROM conversation_chunks c
               LEFT JOIN (SELECT DISTINCT source_id FROM chunk_edges) e
                      ON c.id = e.source_id
               WHERE e.source_id IS NULL AND c.embedding IS NOT NULL"""
        ).fetchall()
        return [r["id"] for r in rows]
    except Exception:
        return []


def incremental_chunk_update(batch_size: int = 200) -> dict:
    """Run chunk_and_summarize, then update Top-K edges for new chunks.
    Designed to be called from background tasks (e.g. chat_sync_receiver).

    Also opportunistically heals "edge orphans" — chunks that have an
    embedding but no outgoing edges (a side-effect of older runs where
    embedding was filled in after the chunk row was already created).
    """
    result = chunk_and_summarize(batch_size=batch_size)

    chunks_created = result.get("chunks_created", 0) if result.get("ok") else 0

    new_ids: list[int] = []
    db = _get_db()
    try:
        if chunks_created:
            rows = db.execute(
                """SELECT id FROM conversation_chunks
                   ORDER BY id DESC LIMIT ?""",
                (chunks_created,),
            ).fetchall()
            new_ids = [r["id"] for r in rows]

        orphan_ids = _find_edge_orphans(db)
    finally:
        db.close()

    to_build = list({*new_ids, *orphan_ids})
    if to_build:
        edge_result = update_edges_for_new_chunks(to_build)
        result["edges_created"] = edge_result.get("edges", 0)
        if orphan_ids:
            result["edge_orphans_healed"] = len(orphan_ids)

    return result


def search_chunks_with_graph(query: str, limit: int = 5, expand: int = 2) -> list[dict]:
    """Search chunks, then expand results along graph edges for richer context."""
    query = (query or "").strip()
    if not query:
        return []

    query_vec = _embed(query[:2000])
    if not query_vec:
        return []

    db = _get_db()
    try:
        _ensure_table(db)
        _ensure_edge_table(db)

        rows = db.execute(
            """SELECT id, start_msg_id, end_msg_id, msg_count, platforms,
                      summary, keywords, embedding, start_time, end_time
               FROM conversation_chunks WHERE embedding IS NOT NULL"""
        ).fetchall()

        scored = []
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = _blob_to_vec(blob)
            sim = _cosine_similarity(query_vec, vec)
            if sim > 0:
                item = dict(row)
                item.pop("embedding", None)
                item["similarity"] = sim
                scored.append(item)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        seed_chunks = scored[:limit]
        seed_ids = {c["id"] for c in seed_chunks}

        # Expand: multi-hop traversal over BOTH similarity edges AND causal edges
        expanded = []
        seen_ids = set(seed_ids)
        frontier = [(s["id"], 1.0) for s in seed_chunks]

        for hop in range(2):
            next_frontier = []
            decay = 0.7 ** (hop + 1)
            for fid, fweight in frontier:
                # Similarity edges
                neighbors = db.execute(
                    """SELECT target_id, similarity * strength as score FROM chunk_edges
                       WHERE source_id = ? ORDER BY score DESC LIMIT ?""",
                    (fid, expand),
                ).fetchall()
                # Causal edges (both directions)
                try:
                    causal_fwd = db.execute(
                        """SELECT target_id, causal_score as score FROM chunk_causal_edges
                           WHERE source_id = ? ORDER BY score DESC LIMIT 2""",
                        (fid,),
                    ).fetchall()
                    causal_bwd = db.execute(
                        """SELECT source_id as target_id, causal_score as score FROM chunk_causal_edges
                           WHERE target_id = ? ORDER BY score DESC LIMIT 2""",
                        (fid,),
                    ).fetchall()
                    neighbors = list(neighbors) + list(causal_fwd) + list(causal_bwd)
                except Exception:
                    pass

                for n in neighbors:
                    tid = n["target_id"]
                    if tid in seen_ids:
                        continue
                    seen_ids.add(tid)
                    chunk = db.execute(
                        """SELECT id, start_msg_id, end_msg_id, msg_count, platforms,
                                  summary, keywords, start_time, end_time
                           FROM conversation_chunks WHERE id = ?""",
                        (tid,),
                    ).fetchone()
                    if chunk:
                        item = dict(chunk)
                        item["similarity"] = n["score"] * decay
                        item["via_edge_from"] = fid
                        item["hop"] = hop + 1
                        expanded.append(item)
                        next_frontier.append((tid, decay))
            frontier = next_frontier

        # Hebbian: strengthen edges between co-retrieved chunks
        all_ids = list(seed_ids) + [e["id"] for e in expanded]
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                db.execute(
                    """UPDATE chunk_edges SET
                       strength = MIN(strength + 0.1, 5.0),
                       surfaced_count = surfaced_count + 1
                       WHERE source_id = ? AND target_id = ?""",
                    (all_ids[i], all_ids[j]),
                )
                db.execute(
                    """UPDATE chunk_edges SET
                       strength = MIN(strength + 0.1, 5.0),
                       surfaced_count = surfaced_count + 1
                       WHERE source_id = ? AND target_id = ?""",
                    (all_ids[j], all_ids[i]),
                )
        db.commit()

        all_results = seed_chunks + expanded

        # Expand original messages for all results
        for chunk in all_results:
            platform_sql, platform_params = _platform_exclusion("platform")
            msgs = db.execute(
                f"""SELECT id, platform, direction, speaker, content, created_at
                   FROM conversation_log
                   WHERE id BETWEEN ? AND ? {platform_sql}
                   ORDER BY id""",
                (chunk["start_msg_id"], chunk["end_msg_id"], *platform_params),
            ).fetchall()
            chunk["messages"] = [dict(m) for m in msgs]
            for m in chunk["messages"]:
                m["content"] = _strip_thinking(m["content"], m["direction"])

        return all_results
    finally:
        db.close()


def format_graph_results(results: list[dict]) -> str:
    """Format graph-expanded chunk search results."""
    if not results:
        return "没有找到相关对话记录"

    sections = []
    for r in results:
        kw = f" | 关键词: {r['keywords']}" if r.get("keywords") else ""
        via = f" | 🔗 via chunk#{r['via_edge_from']}" if r.get("via_edge_from") else ""
        header = (f"[相似度 {r['similarity']:.3f}] "
                  f"{r['start_time']} ~ {r['end_time']} "
                  f"({r['msg_count']}条, {r['platforms']}){kw}{via}")
        summary_line = f"摘要: {r['summary']}"

        msg_lines = []
        for m in r.get("messages", []):
            speaker = m.get("speaker") or (USER_NAME if m["direction"] == "in" else AGENT_NAME)
            content = " ".join(m["content"].split())
            if len(content) > 300:
                content = content[:300] + "..."
            msg_lines.append(f"  {speaker}: {content}")

        section = f"{header}\n{summary_line}\n" + "\n".join(msg_lines)
        sections.append(section)

    return "\n\n---\n\n".join(sections)
