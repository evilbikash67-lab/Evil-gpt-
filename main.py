"""
Evil GPT - Telegram AI Platform
================================
Single-file production-ready implementation.

Stack: Python 3.12, aiogram 3.x, FastAPI, SQLite (aiosqlite), OpenAI-compatible
client, Hugging Face Inference Client, Tavily Search, Pillow, aiohttp, dotenv.

Run modes:
  - Webhook (recommended for Render): set USE_WEBHOOK=true and WEBHOOK_BASE_URL
  - Polling fallback: set USE_WEBHOOK=false (or leave WEBHOOK_BASE_URL empty)

Two bots run concurrently:
  - User bot   (BOT_TOKEN)      -> natural-language AI assistant
  - Admin bot  (ADMIN_BOT_TOKEN)-> management dashboard via chat commands
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sqlite3
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    Update,
    BufferedInputFile,
)
from aiogram.exceptions import TelegramAPIError

from fastapi import FastAPI, Request, Response
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name: str) -> set[int]:
    raw = _env(name)
    out = set()
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            out.add(int(part))
    return out


class Config:
    BOT_TOKEN = _env("BOT_TOKEN")
    ADMIN_BOT_TOKEN = _env("ADMIN_BOT_TOKEN")
    ADMIN_IDS = _env_int_list("ADMIN_IDS")

    # Text chat now runs on the Hugging Face router by default (same token as
    # vision/image). If you ever want to point chat at a different
    # OpenAI-compatible provider instead, set CHAT_PROVIDER=openai and fill in
    # OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL below.
    CHAT_PROVIDER = _env("CHAT_PROVIDER", "huggingface")  # "huggingface" | "openai"

    OPENAI_API_KEY = _env("OPENAI_API_KEY")
    OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")

    HF_TOKEN = _env("HF_TOKEN")
    HF_ROUTER_BASE_URL = _env("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")

    # Chat models tried in order via the HF router, first success wins.
    # Override with HF_CHAT_MODELS as a comma-separated list.
    HF_CHAT_MODELS = [
        m.strip()
        for m in _env(
            "HF_CHAT_MODELS",
            "Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct,"
            "Qwen/Qwen2.5-7B-Instruct",
        ).split(",")
        if m.strip()
    ]

    # Vision models tried in order via the HF router (OpenAI-compatible), first
    # success wins. Override with HF_VISION_MODELS as a comma-separated list.
    HF_VISION_MODELS = [
        m.strip()
        for m in _env(
            "HF_VISION_MODELS",
            "Qwen/Qwen2.5-VL-72B-Instruct:featherless-ai,Qwen/Qwen2-VL-7B-Instruct",
        ).split(",")
        if m.strip()
    ]

    # Text-to-image models tried in order via huggingface_hub InferenceClient,
    # first success wins. Override with HF_IMAGE_MODELS as a comma-separated list.
    HF_IMAGE_MODELS = [
        m.strip()
        for m in _env(
            "HF_IMAGE_MODELS",
            "black-forest-labs/FLUX.1-schnell,black-forest-labs/FLUX.1-dev",
        ).split(",")
        if m.strip()
    ]
    HF_IMAGE_PROVIDER = _env("HF_IMAGE_PROVIDER", "fal-ai")

    HF_BG_REMOVE_MODEL = _env("HF_BG_REMOVE_MODEL", "briaai/RMBG-1.4")

    TAVILY_API_KEY = _env("TAVILY_API_KEY")

    DATABASE_PATH = _env("DATABASE_PATH", "evilgpt.db")

    USE_WEBHOOK = _env_bool("USE_WEBHOOK", False)
    WEBHOOK_BASE_URL = _env("WEBHOOK_BASE_URL")
    WEBHOOK_SECRET = _env("WEBHOOK_SECRET", "evilgpt-secret")
    PORT = int(_env("PORT", "10000"))

    MAX_HISTORY_MESSAGES = int(_env("MAX_HISTORY_MESSAGES", "20"))
    RATE_LIMIT_PER_MINUTE = int(_env("RATE_LIMIT_PER_MINUTE", "20"))
    REQUEST_TIMEOUT = int(_env("REQUEST_TIMEOUT", "60"))
    MAX_RETRIES = int(_env("MAX_RETRIES", "3"))

    DEFAULT_SYSTEM_PROMPT = _env(
        "DEFAULT_SYSTEM_PROMPT",
        "You are Evil GPT, a helpful, sharp, no-nonsense AI assistant inside "
        "Telegram. Be concise, clear, and useful. Format answers cleanly for "
        "chat (short paragraphs, bullet points where useful).",
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("evilgpt")
logging.getLogger("aiogram").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

FRIENDLY_ERROR = (
    "Sorry, something went wrong on my side while handling that. "
    "Please try again in a moment."
)


def log_exception(context: str, exc: BaseException) -> None:
    log.error("Error in %s: %s\n%s", context, exc, traceback.format_exc())


# ---------------------------------------------------------------------------
# Database layer (sqlite3 used through a thread executor for async safety)
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    def _init_sync(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                created_at TEXT,
                last_seen TEXT,
                banned INTEGER DEFAULT 0,
                muted INTEGER DEFAULT 0,
                premium INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event TEXT,
                detail TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS stats (
                day TEXT PRIMARY KEY,
                messages INTEGER DEFAULT 0,
                images INTEGER DEFAULT 0,
                searches INTEGER DEFAULT 0,
                new_users INTEGER DEFAULT 0
            );
            """
        )
        conn.commit()
        conn.close()

    async def init(self):
        await self._run(self._init_sync)
        # seed default settings
        if await self.get_setting("maintenance_mode") is None:
            await self.set_setting("maintenance_mode", "0")
        if await self.get_setting("system_prompt") is None:
            await self.set_setting("system_prompt", Config.DEFAULT_SYSTEM_PROMPT)

    # ---- users -----------------------------------------------------------

    def _get_or_create_user_sync(self, user_id: int, username: str, first_name: str):
        conn = self._connect()
        cur = conn.cursor()
        row = cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        is_new = False
        if row is None:
            cur.execute(
                "INSERT INTO users (user_id, username, first_name, created_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, first_name, now, now),
            )
            is_new = True
        else:
            cur.execute(
                "UPDATE users SET username=?, first_name=?, last_seen=? WHERE user_id=?",
                (username, first_name, now, user_id),
            )
        conn.commit()
        row = cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        conn.close()
        return dict(row), is_new

    async def get_or_create_user(self, user_id: int, username: str, first_name: str):
        return await self._run(self._get_or_create_user_sync, user_id, username, first_name)

    def _bump_message_count_sync(self, user_id: int):
        conn = self._connect()
        conn.execute(
            "UPDATE users SET message_count = message_count + 1 WHERE user_id=?",
            (user_id,),
        )
        conn.commit()
        conn.close()

    async def bump_message_count(self, user_id: int):
        await self._run(self._bump_message_count_sync, user_id)

    def _set_flag_sync(self, user_id: int, field_name: str, value: int):
        conn = self._connect()
        conn.execute(f"UPDATE users SET {field_name}=? WHERE user_id=?", (value, user_id))
        conn.commit()
        conn.close()

    async def set_banned(self, user_id: int, banned: bool):
        await self._run(self._set_flag_sync, user_id, "banned", int(banned))

    async def set_muted(self, user_id: int, muted: bool):
        await self._run(self._set_flag_sync, user_id, "muted", int(muted))

    async def set_premium(self, user_id: int, premium: bool):
        await self._run(self._set_flag_sync, user_id, "premium", int(premium))

    def _get_user_sync(self, user_id: int):
        conn = self._connect()
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    async def get_user(self, user_id: int):
        return await self._run(self._get_user_sync, user_id)

    def _all_user_ids_sync(self):
        conn = self._connect()
        rows = conn.execute("SELECT user_id FROM users WHERE banned=0").fetchall()
        conn.close()
        return [r["user_id"] for r in rows]

    async def all_user_ids(self):
        return await self._run(self._all_user_ids_sync)

    # ---- history -----------------------------------------------------------

    def _add_message_sync(self, user_id: int, role: str, content: str):
        conn = self._connect()
        conn.execute(
            "INSERT INTO history (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    async def add_message(self, user_id: int, role: str, content: str):
        await self._run(self._add_message_sync, user_id, role, content)

    def _get_history_sync(self, user_id: int, limit: int):
        conn = self._connect()
        rows = conn.execute(
            "SELECT role, content FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    async def get_history(self, user_id: int, limit: Optional[int] = None):
        return await self._run(self._get_history_sync, user_id, limit or Config.MAX_HISTORY_MESSAGES)

    def _clear_history_sync(self, user_id: int):
        conn = self._connect()
        conn.execute("DELETE FROM history WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()

    async def clear_history(self, user_id: int):
        await self._run(self._clear_history_sync, user_id)

    # ---- settings ----------------------------------------------------------

    def _get_setting_sync(self, key: str):
        conn = self._connect()
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row["value"] if row else None

    async def get_setting(self, key: str) -> Optional[str]:
        return await self._run(self._get_setting_sync, key)

    def _set_setting_sync(self, key: str, value: str):
        conn = self._connect()
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()
        conn.close()

    async def set_setting(self, key: str, value: str):
        await self._run(self._set_setting_sync, key, value)

    # ---- logs & stats --------------------------------------------------------

    def _add_log_sync(self, user_id: int, event: str, detail: str):
        conn = self._connect()
        conn.execute(
            "INSERT INTO logs (user_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
            (user_id, event, detail, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    async def add_log(self, user_id: int, event: str, detail: str = ""):
        await self._run(self._add_log_sync, user_id, event, detail)

    def _bump_stat_sync(self, field_name: str, is_new_user: bool):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = self._connect()
        conn.execute(
            "INSERT INTO stats (day) VALUES (?) ON CONFLICT(day) DO NOTHING", (day,)
        )
        conn.execute(f"UPDATE stats SET {field_name} = {field_name} + 1 WHERE day=?", (day,))
        if is_new_user:
            conn.execute("UPDATE stats SET new_users = new_users + 1 WHERE day=?", (day,))
        conn.commit()
        conn.close()

    async def bump_stat(self, field_name: str, is_new_user: bool = False):
        await self._run(self._bump_stat_sync, field_name, is_new_user)

    def _get_overview_sync(self):
        conn = self._connect()
        total_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
        banned = conn.execute("SELECT COUNT(*) c FROM users WHERE banned=1").fetchone()["c"]
        premium = conn.execute("SELECT COUNT(*) c FROM users WHERE premium=1").fetchone()["c"]
        total_messages = conn.execute("SELECT COUNT(*) c FROM history").fetchone()["c"]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_row = conn.execute("SELECT * FROM stats WHERE day=?", (today,)).fetchone()
        conn.close()
        return {
            "total_users": total_users,
            "banned": banned,
            "premium": premium,
            "total_messages": total_messages,
            "today": dict(today_row) if today_row else {},
        }

    async def get_overview(self):
        return await self._run(self._get_overview_sync)


db = Database(Config.DATABASE_PATH)

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[int, list[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        window = self._hits.setdefault(user_id, [])
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


rate_limiter = RateLimiter(Config.RATE_LIMIT_PER_MINUTE)

# ---------------------------------------------------------------------------
# In-memory response cache (for identical repeated prompts, short TTL)
# ---------------------------------------------------------------------------

class TTLCache:
    def __init__(self, ttl: int = 120, max_items: int = 500):
        self.ttl = ttl
        self.max_items = max_items
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str):
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if time.monotonic() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        if len(self._store) >= self.max_items:
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest, None)
        self._store[key] = (time.monotonic(), value)


search_cache = TTLCache(ttl=300)  # 5 minutes, per spec

# ---------------------------------------------------------------------------
# HTTP helper with retries
# ---------------------------------------------------------------------------

async def http_post_json(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = Config.REQUEST_TIMEOUT,
    retries: int = Config.MAX_RETRIES,
) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            async with session.post(
                url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("HTTP attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            await asyncio.sleep(min(2 ** attempt, 8))
    assert last_exc is not None
    raise last_exc


async def http_get_bytes(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    timeout: int = Config.REQUEST_TIMEOUT,
    retries: int = Config.MAX_RETRIES,
) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                return await resp.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("HTTP GET attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            await asyncio.sleep(min(2 ** attempt, 8))
    assert last_exc is not None
    raise last_exc


async def http_post_bytes(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = Config.REQUEST_TIMEOUT,
    retries: int = Config.MAX_RETRIES,
) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            async with session.post(
                url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                return await resp.read()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("HTTP POST(bytes) attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            await asyncio.sleep(min(2 ** attempt, 8))
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# AI provider clients
# ---------------------------------------------------------------------------

class ChatClient:
    """Chat completions client — defaults to the Hugging Face router (uses the
    same HF_TOKEN as vision/image), with model fallback. Set
    CHAT_PROVIDER=openai to use a separate OpenAI-compatible provider instead.
    """

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def complete(self, messages: list[dict], model: Optional[str] = None) -> str:
        if Config.CHAT_PROVIDER == "openai":
            return await self._complete_openai(messages, model)
        return await self._complete_hf(messages, model)

    async def _complete_openai(self, messages: list[dict], model: Optional[str] = None) -> str:
        url = f"{Config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or Config.OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7,
        }
        data = await http_post_json(self.session, url, headers, payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat response shape: {data}") from exc

    async def _complete_hf(self, messages: list[dict], model: Optional[str] = None) -> str:
        url = f"{Config.HF_ROUTER_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}",
            "Content-Type": "application/json",
        }
        model_list = [model] if model else Config.HF_CHAT_MODELS

        last_exc: Optional[Exception] = None
        for candidate in model_list:
            payload = {
                "model": candidate,
                "messages": messages,
                "temperature": 0.7,
            }
            try:
                data = await http_post_json(self.session, url, headers, payload, retries=1)
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Chat model %s failed, trying next: %s", candidate, exc)
                continue

        assert last_exc is not None
        raise RuntimeError(f"All chat models failed: {last_exc}")

    async def vision_complete(self, prompt: str, image_bytes: bytes) -> str:
        """Vision completion via the Hugging Face router (OpenAI-compatible).

        Tries each model in Config.HF_VISION_MODELS in order, falling back to
        the next on failure (rate limit, model cold, unavailable, etc).
        """
        b64 = base64.b64encode(image_bytes).decode()
        url = f"{Config.HF_ROUTER_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}",
            "Content-Type": "application/json",
        }
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

        last_exc: Optional[Exception] = None
        for model in Config.HF_VISION_MODELS:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_content,
                        ],
                    }
                ],
                "temperature": 0.4,
            }
            try:
                data = await http_post_json(self.session, url, headers, payload, retries=1)
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Vision model %s failed, trying next: %s", model, exc)
                continue

        assert last_exc is not None
        raise RuntimeError(f"All vision models failed: {last_exc}")


class ImageClient:
    """Hugging Face Inference for image generation and background removal."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        # Sync InferenceClient from huggingface_hub — text_to_image has no
        # first-class async client, so calls are offloaded via asyncio.to_thread.
        self._hf_client = InferenceClient(
            provider=Config.HF_IMAGE_PROVIDER,
            api_key=Config.HF_TOKEN,
        )

    def _text_to_image_sync(self, prompt: str, model: str):
        return self._hf_client.text_to_image(prompt, model=model)

    async def generate(self, prompt: str) -> bytes:
        """Generate an image, trying each model in Config.HF_IMAGE_MODELS in
        order and falling back to the next on failure."""
        last_exc: Optional[Exception] = None
        for model in Config.HF_IMAGE_MODELS:
            try:
                pil_image = await asyncio.to_thread(self._text_to_image_sync, prompt, model)
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                return buf.getvalue()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Image model %s failed, trying next: %s", model, exc)
                continue
        assert last_exc is not None
        raise RuntimeError(f"All image models failed: {last_exc}")

    async def remove_background(self, image_bytes: bytes) -> bytes:
        url = f"https://api-inference.huggingface.co/models/{Config.HF_BG_REMOVE_MODEL}"
        headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}",
            "Content-Type": "application/octet-stream",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                async with self.session.post(
                    url, headers=headers, data=image_bytes,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
                    return await resp.read()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("BG removal attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(min(2 ** attempt, 8))
        assert last_exc is not None
        raise last_exc


class SearchClient:
    """Tavily realtime search.

    - 5 minute result cache (see search_cache below)
    - advanced search depth, top 5 results
    - requests answer, raw_content, images, and published dates
    - logs query, response time, cache hit/miss, result count, and errors
    - NEVER raises: on any failure returns None so the caller can gracefully
      fall back to answering from the LLM's own knowledge
    """

    MAX_QUERY_LEN = 400

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @staticmethod
    def _sanitize_query(query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"\s+", " ", q)
        # strip control characters
        q = "".join(ch for ch in q if ch.isprintable())
        return q[: SearchClient.MAX_QUERY_LEN]

    async def search(self, query: str) -> Optional[dict]:
        clean_query = self._sanitize_query(query)
        if not clean_query:
            return None

        cache_key = clean_query.lower()
        cached = search_cache.get(cache_key)
        if cached is not None:
            log.info("search_cache HIT query=%r", clean_query)
            return cached

        log.info("search_cache MISS query=%r", clean_query)

        if not Config.TAVILY_API_KEY:
            log.warning("Tavily search skipped: TAVILY_API_KEY is not configured.")
            return None

        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {Config.TAVILY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": clean_query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": True,
            "include_raw_content": True,
            "include_images": True,
        }

        start = time.monotonic()
        try:
            data = await http_post_json(self.session, url, headers, payload, retries=Config.MAX_RETRIES)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            results = data.get("results") or []
            log.info(
                "tavily_search OK query=%r elapsed_ms=%s results=%s",
                clean_query, elapsed_ms, len(results),
            )
            search_cache.set(cache_key, data)
            return data
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "tavily_search FAILED query=%r elapsed_ms=%s error=%s",
                clean_query, elapsed_ms, exc,
            )
            return None


# ---------------------------------------------------------------------------
# Intent router (keyword / heuristic based - no command dependency)
# ---------------------------------------------------------------------------

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|sup|hola|namaste)\b[\s!.]*$", re.I)

SEARCH_TRIGGERS = (
    # news / current events
    "latest", "news", "today", "current", "recent", "right now", "this week",
    "update on", "happening", "breaking",
    # sports / live scores
    "score", "scores", "match", "ipl", "world cup", "tournament", "fixtures",
    "who won", "live score",
    # elections / government
    "election", "elections", "vote count", "poll results", "government announcement",
    "policy update",
    # weather / traffic
    "weather", "forecast", "temperature", "traffic", "road conditions",
    # finance / markets
    "stock price", "stock", "share price", "market trend", "crypto", "bitcoin",
    "ethereum", "currency rate", "exchange rate", "gold price", "nasdaq",
    "sensex", "nifty",
    # products / places
    "product review", "best laptop", "best phone", "top ai tools", "newest",
    "release date", "restaurants near", "hotels near", "nearby", "near me",
    # companies / people / tech
    "company information", "who is the current", "what is happening",
    "ai news", "technology update", "tech news", "announcement", "announced",
)

IMAGE_GEN_TRIGGERS = (
    "create image", "generate image", "generate a wallpaper", "make an image",
    "draw", "wallpaper", "logo", "poster", "anime art", "generate wallpaper",
    "create a logo", "create poster", "create wallpaper", "picture of", "image of",
)

IMAGE_EDIT_TRIGGERS = (
    "remove background", "remove the background", "change clothes", "improve quality",
    "replace object", "enhance image", "enhance this", "restore old photo",
    "restore this photo", "make cinematic", "upscale",
)

TRANSLATE_TRIGGERS = ("translate",)

SUMMARIZE_TRIGGERS = ("summarize", "summarise", "tl;dr", "tldr", "sum up")

CODE_TRIGGERS = (
    "write python", "write code", "write a function", "debug this", "fix this code",
    "code for", "script for", "regex for", "sql query", "write a program",
)

MATH_TRIGGERS = ("solve", "calculate", "integral", "derivative", "equation", "math problem")

OCR_TRIGGERS = ("read this", "ocr", "extract text", "what does this say", "read the document")


def classify_intent(text: str, has_image: bool) -> str:
    t = (text or "").strip().lower()

    if has_image:
        if any(k in t for k in IMAGE_EDIT_TRIGGERS):
            return "image_edit"
        if any(k in t for k in OCR_TRIGGERS):
            return "ocr"
        return "vision"

    if not t:
        return "chat"

    if GREETING_RE.match(t):
        return "greeting"

    if any(k in t for k in IMAGE_GEN_TRIGGERS):
        return "image_gen"

    if any(k in t for k in TRANSLATE_TRIGGERS):
        return "translate"

    if any(k in t for k in SUMMARIZE_TRIGGERS):
        return "summarize"

    if any(k in t for k in CODE_TRIGGERS):
        return "code"

    if any(k in t for k in MATH_TRIGGERS):
        return "math"

    if any(k in t for k in SEARCH_TRIGGERS):
        return "search"

    if "?" in t and len(t.split()) < 12 and any(
        w in t for w in ("who", "what", "when", "where", "which", "how many")
    ):
        return "search"

    return "chat"


# ---------------------------------------------------------------------------
# Shared aiohttp session holder
# ---------------------------------------------------------------------------

@dataclass
class Services:
    session: aiohttp.ClientSession
    chat: ChatClient
    image: ImageClient
    search: SearchClient


services: Optional[Services] = None


async def build_services() -> Services:
    session = aiohttp.ClientSession()
    return Services(
        session=session,
        chat=ChatClient(session),
        image=ImageClient(session),
        search=SearchClient(session),
    )


# ---------------------------------------------------------------------------
# User bot
# ---------------------------------------------------------------------------

user_router = Router(name="user")


def is_admin(user_id: int) -> bool:
    return user_id in Config.ADMIN_IDS


async def guarded(message: Message) -> Optional[dict]:
    """Common guard: maintenance mode, ban/mute checks, user upsert, rate limit."""
    if message.from_user is None:
        return None

    user, is_new = await db.get_or_create_user(
        message.from_user.id,
        message.from_user.username or "",
        message.from_user.first_name or "",
    )
    if is_new:
        await db.bump_stat("messages", is_new_user=True)
        await db.add_log(message.from_user.id, "new_user")

    maintenance = await db.get_setting("maintenance_mode")
    if maintenance == "1" and not is_admin(message.from_user.id):
        await message.answer(
            "Evil GPT is temporarily down for maintenance. Please check back soon."
        )
        return None

    if user["banned"]:
        return None  # silently ignore banned users

    if user["muted"]:
        await message.answer("Your account is currently muted.")
        return None

    if not rate_limiter.allow(message.from_user.id):
        await message.answer("You're sending messages a bit fast — please slow down.")
        return None

    return user


@user_router.message(CommandStart())
async def cmd_start(message: Message):
    await guarded(message)
    await message.answer("Welcome to Evil GPT.\n\nJust send a message, image, or request.")


@user_router.message(Command("help"))
async def cmd_help(message: Message):
    user = await guarded(message)
    if user is None:
        return
    await message.answer(
        "Just talk to me naturally — ask a question, request an image, send a "
        "photo, or ask me to translate, summarize, or code something.\n\n"
        "Commands: /clear (clear memory), /newchat (start fresh), /settings, /ping"
    )


@user_router.message(Command("ping"))
async def cmd_ping(message: Message):
    user = await guarded(message)
    if user is None:
        return
    await message.answer("pong")


@user_router.message(Command("settings"))
async def cmd_settings(message: Message):
    user = await guarded(message)
    if user is None:
        return
    plan = "Premium" if user["premium"] else "Free"
    await message.answer(f"Plan: {plan}\nMessages sent: {user['message_count']}")


@user_router.message(Command("clear"))
async def cmd_clear(message: Message):
    user = await guarded(message)
    if user is None:
        return
    await db.clear_history(message.from_user.id)
    await message.answer("Memory cleared.")


@user_router.message(Command("newchat"))
async def cmd_newchat(message: Message):
    user = await guarded(message)
    if user is None:
        return
    await db.clear_history(message.from_user.id)
    await message.answer("Started a new conversation.")


def _extract_prompt_after_trigger(text: str, triggers: tuple[str, ...]) -> str:
    low = text.lower()
    for trig in triggers:
        idx = low.find(trig)
        if idx != -1:
            rest = text[idx + len(trig):].strip(" :,-")
            if rest:
                return rest
    return text


def _domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # noqa: BLE001
        return "Web"


def format_sources_block(results: list[dict]) -> str:
    """Formats up to 5 Tavily results as a trusted-sources block:
    🌐 Website Name / 📄 Page Title / 🔗 URL
    """
    if not results:
        return ""
    blocks = []
    for r in results[:5]:
        url = r.get("url") or ""
        if not url:
            continue
        title = (r.get("title") or "Source").strip()
        domain = _domain_from_url(url) or "Web"
        blocks.append(f"🌐 {domain}\n📄 {title}\n🔗 {url}")
    if not blocks:
        return ""
    return "Sources:\n\n" + "\n\n".join(blocks)


async def summarize_search_results(user_id: int, text: str, data: dict) -> str:
    """Combines conversation memory + the user's question + Tavily results into
    a single prompt so the LLM produces a natural, synthesized answer instead
    of dumping raw search results."""
    results = data.get("results") or []
    tavily_answer = (data.get("answer") or "").strip()

    context_chunks = []
    if tavily_answer:
        context_chunks.append(f"Quick answer: {tavily_answer}")
    for r in results[:5]:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or r.get("raw_content") or "").strip()
        url = r.get("url") or ""
        published = r.get("published_date") or r.get("published") or ""
        chunk = f"Source: {title}\nURL: {url}"
        if published:
            chunk += f"\nPublished: {published}"
        if content:
            chunk += f"\nContent: {content[:600]}"
        context_chunks.append(chunk)

    search_context = "\n\n".join(context_chunks) if context_chunks else "No results found."

    history = await db.get_history(user_id)
    system_prompt = await db.get_setting("system_prompt") or Config.DEFAULT_SYSTEM_PROMPT
    system_prompt += (
        "\n\nYou have been given live web search results below the user's message. "
        "Use them, along with the conversation so far, to answer naturally and "
        "conversationally in your own words. Never dump raw search results or "
        "copy them verbatim — synthesize a clear, direct answer."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"{text}\n\n[Live web search results]\n{search_context}",
        }
    )
    return await services.chat.complete(messages)


async def answer_with_search(user_id: int, text: str) -> str:
    """Full search-augmented answer flow with graceful, silent fallback to
    plain LLM knowledge if Tavily is unavailable or fails for any reason."""
    data = await services.search.search(text)

    if not data:
        # Tavily unavailable/failed — fall back to normal AI knowledge, no
        # error shown to the user.
        reply = await handle_chat(user_id, text)
        return reply

    results = data.get("results") or []
    try:
        summary = await summarize_search_results(user_id, text, data)
    except Exception as exc:  # noqa: BLE001
        log_exception("summarize_search_results", exc)
        reply = await handle_chat(user_id, text)
        return reply

    sources_block = format_sources_block(results)
    reply = f"{summary}\n\n{sources_block}" if sources_block else summary

    await db.add_message(user_id, "user", text)
    await db.add_message(user_id, "assistant", reply)
    return reply


async def handle_chat(user_id: int, text: str) -> str:
    history = await db.get_history(user_id)
    system_prompt = await db.get_setting("system_prompt") or Config.DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": text})
    reply = await services.chat.complete(messages)
    await db.add_message(user_id, "user", text)
    await db.add_message(user_id, "assistant", reply)
    return reply


@user_router.message(F.photo)
async def handle_photo(message: Message):
    user = await guarded(message)
    if user is None:
        return
    user_id = message.from_user.id
    caption = message.caption or "Explain this image."
    intent = classify_intent(caption, has_image=True)

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await message.bot.download_file(file.file_path, destination=buf)
        image_bytes = buf.getvalue()

        if intent == "image_edit":
            await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
            edited = await services.image.remove_background(image_bytes)
            out_img = Image.open(io.BytesIO(edited))
            out_buf = io.BytesIO()
            fmt = "PNG" if out_img.mode == "RGBA" else "JPEG"
            out_img.save(out_buf, format=fmt)
            out_buf.seek(0)
            await message.answer_photo(
                BufferedInputFile(out_buf.read(), filename=f"edited.{fmt.lower()}")
            )
            await db.bump_stat("images")
            await db.add_log(user_id, "image_edit")
        else:
            prompt = caption
            if intent == "ocr":
                prompt = "Extract and transcribe all readable text from this image accurately."
            reply = await services.chat.vision_complete(prompt, image_bytes)
            await message.answer(reply)
            await db.add_message(user_id, "user", f"[image] {caption}")
            await db.add_message(user_id, "assistant", reply)
            await db.add_log(user_id, "vision")

        await db.bump_message_count(user_id)
        await db.bump_stat("messages")
    except Exception as exc:  # noqa: BLE001
        log_exception("handle_photo", exc)
        await message.answer(FRIENDLY_ERROR)


@user_router.message(F.text)
async def handle_text(message: Message):
    user = await guarded(message)
    if user is None:
        return

    user_id = message.from_user.id
    text = message.text or ""
    intent = classify_intent(text, has_image=False)

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        if intent == "greeting":
            reply = "Hey! What can I help you with?"
            await message.answer(reply)

        elif intent == "search":
            await db.bump_stat("searches")
            searching_msg = None
            try:
                searching_msg = await message.answer("🔍 Searching the web...")
            except Exception:  # noqa: BLE001
                searching_msg = None

            reply = await answer_with_search(user_id, text)

            if searching_msg is not None:
                try:
                    await searching_msg.edit_text(reply, disable_web_page_preview=True)
                except Exception:  # noqa: BLE001
                    await message.answer(reply, disable_web_page_preview=True)
            else:
                await message.answer(reply, disable_web_page_preview=True)

        elif intent == "image_gen":
            await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
            prompt = _extract_prompt_after_trigger(text, IMAGE_GEN_TRIGGERS)
            image_bytes = await services.image.generate(prompt)
            await message.answer_photo(
                BufferedInputFile(image_bytes, filename="evilgpt.png")
            )
            await db.bump_stat("images")
            await db.add_log(user_id, "image_gen", prompt)

        elif intent == "translate":
            prompt = (
                "Translate the following text. If no target language is specified, "
                "translate to English. Return only the translation.\n\n" + text
            )
            reply = await services.chat.complete(
                [{"role": "system", "content": Config.DEFAULT_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}]
            )
            await message.answer(reply)

        elif intent == "summarize":
            prompt = "Summarize the following clearly and concisely:\n\n" + text
            reply = await services.chat.complete(
                [{"role": "system", "content": Config.DEFAULT_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}]
            )
            await message.answer(reply)

        elif intent == "code":
            reply = await handle_chat(user_id, text)
            await message.answer(f"```\n{reply}\n```" if "```" not in reply else reply,
                                  parse_mode=ParseMode.MARKDOWN)

        elif intent == "math":
            reply = await handle_chat(user_id, text)
            await message.answer(reply)

        else:
            reply = await handle_chat(user_id, text)
            await message.answer(reply)

        await db.bump_message_count(user_id)
        await db.bump_stat("messages")

    except Exception as exc:  # noqa: BLE001
        log_exception(f"handle_text[{intent}]", exc)
        await message.answer(FRIENDLY_ERROR)


# ---------------------------------------------------------------------------
# Admin bot
# ---------------------------------------------------------------------------

admin_router = Router(name="admin")


def admin_only(message: Message) -> bool:
    return bool(message.from_user) and is_admin(message.from_user.id)


@admin_router.message(CommandStart())
async def admin_start(message: Message):
    if not admin_only(message):
        await message.answer("Unauthorized.")
        return
    await message.answer(
        "Evil GPT — Admin Panel\n\n"
        "/stats — usage statistics\n"
        "/broadcast <text> — message all users\n"
        "/ban <user_id>\n"
        "/unban <user_id>\n"
        "/mute <user_id>\n"
        "/unmute <user_id>\n"
        "/premium <user_id> <on|off>\n"
        "/setprompt <text> — update system prompt\n"
        "/maintenance <on|off>"
    )


@admin_router.message(Command("stats"))
async def admin_stats(message: Message):
    if not admin_only(message):
        return
    overview = await db.get_overview()
    today = overview.get("today") or {}
    await message.answer(
        "Statistics\n"
        f"Total users: {overview['total_users']}\n"
        f"Banned: {overview['banned']}\n"
        f"Premium: {overview['premium']}\n"
        f"Total messages: {overview['total_messages']}\n\n"
        "Today\n"
        f"Messages: {today.get('messages', 0)}\n"
        f"Images: {today.get('images', 0)}\n"
        f"Searches: {today.get('searches', 0)}\n"
        f"New users: {today.get('new_users', 0)}"
    )


def _parse_target_id(args: str) -> Optional[int]:
    args = args.strip()
    return int(args) if args.isdigit() else None


@admin_router.message(Command("ban"))
async def admin_ban(message: Message, command: Command):
    if not admin_only(message):
        return
    target = _parse_target_id(command.args or "")
    if target is None:
        await message.answer("Usage: /ban <user_id>")
        return
    await db.set_banned(target, True)
    await message.answer(f"User {target} banned.")


@admin_router.message(Command("unban"))
async def admin_unban(message: Message, command: Command):
    if not admin_only(message):
        return
    target = _parse_target_id(command.args or "")
    if target is None:
        await message.answer("Usage: /unban <user_id>")
        return
    await db.set_banned(target, False)
    await message.answer(f"User {target} unbanned.")


@admin_router.message(Command("mute"))
async def admin_mute(message: Message, command: Command):
    if not admin_only(message):
        return
    target = _parse_target_id(command.args or "")
    if target is None:
        await message.answer("Usage: /mute <user_id>")
        return
    await db.set_muted(target, True)
    await message.answer(f"User {target} muted.")


@admin_router.message(Command("unmute"))
async def admin_unmute(message: Message, command: Command):
    if not admin_only(message):
        return
    target = _parse_target_id(command.args or "")
    if target is None:
        await message.answer("Usage: /unmute <user_id>")
        return
    await db.set_muted(target, False)
    await message.answer(f"User {target} unmuted.")


@admin_router.message(Command("premium"))
async def admin_premium(message: Message, command: Command):
    if not admin_only(message):
        return
    parts = (command.args or "").split()
    if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in ("on", "off"):
        await message.answer("Usage: /premium <user_id> <on|off>")
        return
    await db.set_premium(int(parts[0]), parts[1] == "on")
    await message.answer(f"Premium {'enabled' if parts[1] == 'on' else 'disabled'} for {parts[0]}.")


@admin_router.message(Command("setprompt"))
async def admin_setprompt(message: Message, command: Command):
    if not admin_only(message):
        return
    new_prompt = (command.args or "").strip()
    if not new_prompt:
        await message.answer("Usage: /setprompt <text>")
        return
    await db.set_setting("system_prompt", new_prompt)
    await message.answer("System prompt updated.")


@admin_router.message(Command("maintenance"))
async def admin_maintenance(message: Message, command: Command):
    if not admin_only(message):
        return
    arg = (command.args or "").strip().lower()
    if arg not in ("on", "off"):
        await message.answer("Usage: /maintenance <on|off>")
        return
    await db.set_setting("maintenance_mode", "1" if arg == "on" else "0")
    await message.answer(f"Maintenance mode {'enabled' if arg == 'on' else 'disabled'}.")


@admin_router.message(Command("broadcast"))
async def admin_broadcast(message: Message, command: Command):
    if not admin_only(message):
        return
    text = (command.args or "").strip()
    if not text:
        await message.answer("Usage: /broadcast <text>")
        return
    user_ids = await db.all_user_ids()
    sent, failed = 0, 0
    await message.answer(f"Broadcasting to {len(user_ids)} users...")
    for uid in user_ids:
        try:
            await user_bot.send_message(uid, text)
            sent += 1
        except TelegramAPIError as exc:
            failed += 1
            log.warning("Broadcast failed for %s: %s", uid, exc)
        await asyncio.sleep(0.05)  # gentle rate limiting
    await message.answer(f"Broadcast done. Sent: {sent}, failed: {failed}.")


# ---------------------------------------------------------------------------
# Bot / dispatcher instances
# ---------------------------------------------------------------------------

user_bot: Optional[Bot] = None
admin_bot: Optional[Bot] = None
user_dp: Optional[Dispatcher] = None
admin_dp: Optional[Dispatcher] = None


def build_bots():
    global user_bot, admin_bot, user_dp, admin_dp

    user_bot = Bot(
        token=Config.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    user_dp = Dispatcher()
    user_dp.include_router(user_router)

    if Config.ADMIN_BOT_TOKEN:
        admin_bot = Bot(
            token=Config.ADMIN_BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        admin_dp = Dispatcher()
        admin_dp.include_router(admin_router)


# ---------------------------------------------------------------------------
# FastAPI app (webhook + health)
# ---------------------------------------------------------------------------

polling_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global services

    await db.init()
    build_bots()
    services = await build_services()

    if Config.USE_WEBHOOK and Config.WEBHOOK_BASE_URL:
        base = Config.WEBHOOK_BASE_URL.rstrip("/")
        try:
            await user_bot.set_webhook(
                f"{base}/webhook/user/{Config.WEBHOOK_SECRET}",
                drop_pending_updates=True,
            )
            if admin_bot:
                await admin_bot.set_webhook(
                    f"{base}/webhook/admin/{Config.WEBHOOK_SECRET}",
                    drop_pending_updates=True,
                )
            log.info("Webhooks configured.")
        except Exception as exc:  # noqa: BLE001
            log_exception("set_webhook", exc)
            log.warning("Falling back to polling due to webhook setup failure.")
            polling_tasks.append(asyncio.create_task(user_dp.start_polling(user_bot)))
            if admin_bot:
                polling_tasks.append(asyncio.create_task(admin_dp.start_polling(admin_bot)))
    else:
        log.info("Starting in polling mode.")
        try:
            await user_bot.delete_webhook(drop_pending_updates=True)
            if admin_bot:
                await admin_bot.delete_webhook(drop_pending_updates=True)
        except Exception as exc:  # noqa: BLE001
            log_exception("delete_webhook", exc)
        polling_tasks.append(asyncio.create_task(user_dp.start_polling(user_bot)))
        if admin_bot:
            polling_tasks.append(asyncio.create_task(admin_dp.start_polling(admin_bot)))

    yield

    for task in polling_tasks:
        task.cancel()
    if services:
        await services.session.close()
    await user_bot.session.close()
    if admin_bot:
        await admin_bot.session.close()


app = FastAPI(title="Evil GPT", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/")
async def root():
    return {"service": "Evil GPT", "status": "running"}


@app.post("/webhook/user/{secret}")
async def webhook_user(secret: str, request: Request):
    if secret != Config.WEBHOOK_SECRET:
        return Response(status_code=403)
    try:
        data = await request.json()
        update = Update.model_validate(data, context={"bot": user_bot})
        await user_dp.feed_update(user_bot, update)
    except Exception as exc:  # noqa: BLE001
        log_exception("webhook_user", exc)
    return Response(status_code=200)


@app.post("/webhook/admin/{secret}")
async def webhook_admin(secret: str, request: Request):
    if secret != Config.WEBHOOK_SECRET or admin_bot is None:
        return Response(status_code=403)
    try:
        data = await request.json()
        update = Update.model_validate(data, context={"bot": admin_bot})
        await admin_dp.feed_update(admin_bot, update)
    except Exception as exc:  # noqa: BLE001
        log_exception("webhook_admin", exc)
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not Config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is required. Set it in your environment or .env file.")
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT, log_level="info")
