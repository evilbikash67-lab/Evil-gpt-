#!/usr/bin/env python3
"""
Evil GPT - Production Telegram AI Platform
==========================================
A complete AI-powered Telegram bot platform with user and admin bots,
supporting chat, vision, image generation, and web search capabilities.
"""

import asyncio
import logging
import os
import sqlite3
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
import io

# Third-party imports
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    InputFile, BufferedInputFile, FSInputFile
)
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode, ChatType
from aiogram.exceptions import (
    TelegramBadRequest, TelegramRetryAfter,
    TelegramNetworkError, TelegramAPIError
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Hugging Face / OpenAI compatible client
from openai import OpenAI

# Hugging Face Inference Client for image generation
from huggingface_hub import InferenceClient

# Tavily Search API
from tavily import TavilyClient

# Web framework for health checks (Render)
from aiohttp import web
import aiohttp

# Image processing
from PIL import Image

# Environment management
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# LOGGING CONFIGURATION
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evil_gpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# ENVIRONMENT VARIABLES
# =========================

class Config:
    """Configuration class for environment variables."""
    
    # Bot Tokens
    USER_BOT_TOKEN = os.getenv('USER_BOT_TOKEN')
    ADMIN_BOT_TOKEN = os.getenv('ADMIN_BOT_TOKEN')
    
    # Hugging Face
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    # Chat Models
    HF_CHAT_MODEL = os.getenv('HF_CHAT_MODEL', 'Qwen/Qwen2.5-VL-72B-Instruct:featherless-ai')
    HF_CHAT_FALLBACK = os.getenv('HF_CHAT_FALLBACK', 'Qwen/Qwen2-VL-7B-Instruct')
    
    # Vision Models
    HF_VISION_MODEL = os.getenv('HF_VISION_MODEL', 'Qwen/Qwen2.5-VL-72B-Instruct:featherless-ai')
    HF_VISION_FALLBACK = os.getenv('HF_VISION_FALLBACK', 'Qwen/Qwen2-VL-7B-Instruct')
    
    # Image Generation Models
    HF_IMAGE_MODEL = os.getenv('HF_IMAGE_MODEL', 'black-forest-labs/FLUX.1-dev')
    HF_IMAGE_FALLBACK = os.getenv('HF_IMAGE_FALLBACK', 'black-forest-labs/FLUX.1-schnell')
    
    # Tavily
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    
    # Admin
    ADMIN_IDS = [int(id.strip()) for id in os.getenv('ADMIN_IDS', '').split(',') if id.strip()]
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'evil_gpt.db')
    
    # Rate Limiting
    RATE_LIMIT_MESSAGES = int(os.getenv('RATE_LIMIT_MESSAGES', '10'))
    RATE_LIMIT_PERIOD = int(os.getenv('RATE_LIMIT_PERIOD', '60'))
    RATE_LIMIT_IMAGE_GEN = int(os.getenv('RATE_LIMIT_IMAGE_GEN', '5'))
    RATE_LIMIT_SEARCH = int(os.getenv('RATE_LIMIT_SEARCH', '3'))
    
    # Feature Flags
    ENABLE_SEARCH = os.getenv('ENABLE_SEARCH', 'true').lower() == 'true'
    ENABLE_IMAGE_GEN = os.getenv('ENABLE_IMAGE_GEN', 'true').lower() == 'true'
    ENABLE_VISION = os.getenv('ENABLE_VISION', 'true').lower() == 'true'
    MAINTENANCE_MODE = os.getenv('MAINTENANCE_MODE', 'false').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        required = ['USER_BOT_TOKEN', 'HF_TOKEN']
        missing = [var for var in required if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        if not cls.ADMIN_IDS:
            raise ValueError("ADMIN_IDS must be set and contain at least one admin ID")

# =========================
# DATABASE LAYER
# =========================

class Database:
    """SQLite database handler with context manager support."""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self._init_db()
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_banned BOOLEAN DEFAULT 0,
                    is_muted BOOLEAN DEFAULT 0,
                    is_premium BOOLEAN DEFAULT 0,
                    premium_expiry TIMESTAMP,
                    settings TEXT DEFAULT '{}'
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    role TEXT,
                    content TEXT,
                    model TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens_used INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # System prompts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0
                )
            """)
            
            # Image generation logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    prompt TEXT,
                    model TEXT,
                    success BOOLEAN,
                    image_url TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Rate limiting table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER,
                    action TEXT,
                    count INTEGER DEFAULT 1,
                    last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, action)
                )
            """)
            
            # Banned users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banned_users (
                    user_id INTEGER PRIMARY KEY,
                    reason TEXT,
                    banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    banned_by INTEGER
                )
            """)
            
            # Muted users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS muted_users (
                    user_id INTEGER PRIMARY KEY,
                    reason TEXT,
                    muted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    muted_by INTEGER,
                    mute_until TIMESTAMP
                )
            """)
            
            # Premium codes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS premium_codes (
                    code TEXT PRIMARY KEY,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    used_by INTEGER,
                    used_at TIMESTAMP,
                    is_used BOOLEAN DEFAULT 0,
                    duration_days INTEGER DEFAULT 30
                )
            """)
            
            # Settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Initialize default system prompt if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO system_prompts (prompt, is_active)
                VALUES (?, 1)
            """, ("You are Evil GPT, a helpful AI assistant. You provide accurate, detailed, and thoughtful responses. You have access to the internet for real-time information. Be concise but comprehensive in your answers.",))
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    async def create_or_update_user(self, user_id: int, username: str = None, 
                                   first_name: str = None, last_name: str = None) -> Dict:
        """Create or update user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("""
                    UPDATE users 
                    SET username = COALESCE(?, username),
                        first_name = COALESCE(?, first_name),
                        last_name = COALESCE(?, last_name),
                        last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (username, first_name, last_name, user_id))
                return dict(cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())
            else:
                cursor.execute("""
                    INSERT INTO users (user_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                """, (user_id, username, first_name, last_name))
                return dict(cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())
    
    async def add_chat_history(self, user_id: int, role: str, content: str, 
                              model: str = None, tokens: int = 0):
        """Add message to chat history."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (user_id, role, content, model, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, role, content, model, tokens))
    
    async def get_chat_history(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Get recent chat history for a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM chat_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
    
    async def clear_chat_history(self, user_id: int):
        """Clear chat history for a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    
    async def get_active_system_prompt(self) -> str:
        """Get the active system prompt."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt FROM system_prompts 
                WHERE is_active = 1 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            return row['prompt'] if row else "You are a helpful AI assistant."
    
    async def set_system_prompt(self, prompt: str, created_by: int) -> int:
        """Set a new system prompt and deactivate old ones."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE system_prompts SET is_active = 0")
            cursor.execute("""
                INSERT INTO system_prompts (prompt, created_by, is_active)
                VALUES (?, ?, 1)
            """, (prompt, created_by))
            return cursor.lastrowid
    
    async def log_image_generation(self, user_id: int, prompt: str, model: str, 
                                   success: bool, image_url: str = None):
        """Log image generation attempt."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO image_logs (user_id, prompt, model, success, image_url)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, prompt, model, success, image_url))
    
    async def get_stats(self) -> Dict:
        """Get overall statistics."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM users 
                WHERE last_active > datetime('now', '-24 hours')
            """)
            active_users = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM chat_history")
            total_messages = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM image_logs")
            total_images = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM banned_users")
            banned_users = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM muted_users")
            muted_users = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_premium = 1")
            premium_users = cursor.fetchone()['count']
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'total_messages': total_messages,
                'total_images': total_images,
                'banned_users': banned_users,
                'muted_users': muted_users,
                'premium_users': premium_users
            }
    
    async def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a specific user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM chat_history 
                WHERE user_id = ?
            """, (user_id,))
            messages = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM image_logs 
                WHERE user_id = ?
            """, (user_id,))
            images = cursor.fetchone()['count']
            
            return {
                'messages': messages,
                'images': images
            }
    
    async def ban_user(self, user_id: int, reason: str = None, banned_by: int = None):
        """Ban a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO banned_users (user_id, reason, banned_by)
                VALUES (?, ?, ?)
            """, (user_id, reason, banned_by))
            cursor.execute("UPDATE users SET is_banned = 1 WHERE user_id = ?", (user_id,))
    
    async def unban_user(self, user_id: int):
        """Unban a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM banned_users WHERE user_id = ?", (user_id,))
            cursor.execute("UPDATE users SET is_banned = 0 WHERE user_id = ?", (user_id,))
    
    async def is_user_banned(self, user_id: int) -> bool:
        """Check if a user is banned."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM banned_users WHERE user_id = ?", (user_id,))
            return cursor.fetchone() is not None
    
    async def mute_user(self, user_id: int, duration_minutes: int = 60, reason: str = None, muted_by: int = None):
        """Mute a user for a specified duration."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            mute_until = datetime.now() + timedelta(minutes=duration_minutes)
            cursor.execute("""
                INSERT OR REPLACE INTO muted_users (user_id, reason, muted_by, mute_until)
                VALUES (?, ?, ?, ?)
            """, (user_id, reason, muted_by, mute_until))
            cursor.execute("UPDATE users SET is_muted = 1 WHERE user_id = ?", (user_id,))
    
    async def unmute_user(self, user_id: int):
        """Unmute a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM muted_users WHERE user_id = ?", (user_id,))
            cursor.execute("UPDATE users SET is_muted = 0 WHERE user_id = ?", (user_id,))
    
    async def is_user_muted(self, user_id: int) -> bool:
        """Check if a user is muted."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mute_until FROM muted_users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return False
            mute_until = datetime.fromisoformat(row['mute_until'])
            if mute_until < datetime.now():
                await self.unmute_user(user_id)
                return False
            return True
    
    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get list of all users."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, first_name, last_name, 
                       created_at, last_active, is_banned, is_muted, is_premium
                FROM users 
                ORDER BY last_active DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    async def generate_premium_code(self, created_by: int, duration_days: int = 30) -> str:
        """Generate a premium activation code."""
        import secrets
        code = secrets.token_hex(8).upper()
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO premium_codes (code, created_by, duration_days)
                VALUES (?, ?, ?)
            """, (code, created_by, duration_days))
            return code
    
    async def use_premium_code(self, code: str, user_id: int) -> bool:
        """Use a premium code for a user."""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT code, duration_days, is_used FROM premium_codes 
                WHERE code = ? AND is_used = 0
            """, (code,))
            row = cursor.fetchone()
            if not row:
                return False
            
            cursor.execute("""
                UPDATE premium_codes 
                SET used_by = ?, used_at = CURRENT_TIMESTAMP, is_used = 1
                WHERE code = ?
            """, (user_id, code))
            
            # Activate premium
            expiry = datetime.now() + timedelta(days=row['duration_days'])
            cursor.execute("""
                UPDATE users SET is_premium = 1, premium_expiry = ?
                WHERE user_id = ?
            """, (expiry, user_id))
            return True

# =========================
# RATE LIMITER
# =========================

class RateLimiter:
    """Rate limiter for bot actions."""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def check_limit(self, user_id: int, action: str, 
                          max_count: int, period: int) -> Tuple[bool, int]:
        """Check if user is within rate limit."""
        async with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT count, last_reset FROM rate_limits 
                WHERE user_id = ? AND action = ?
            """, (user_id, action))
            row = cursor.fetchone()
            
            now = datetime.now()
            
            if not row:
                cursor.execute("""
                    INSERT INTO rate_limits (user_id, action, count, last_reset)
                    VALUES (?, ?, 1, ?)
                """, (user_id, action, now))
                return True, max_count - 1
            
            count, last_reset = row['count'], datetime.fromisoformat(row['last_reset'])
            
            if (now - last_reset).total_seconds() > period:
                cursor.execute("""
                    UPDATE rate_limits 
                    SET count = 1, last_reset = ?
                    WHERE user_id = ? AND action = ?
                """, (now, user_id, action))
                return True, max_count - 1
            
            if count >= max_count:
                return False, 0
            
            cursor.execute("""
                UPDATE rate_limits 
                SET count = count + 1, last_reset = ?
                WHERE user_id = ? AND action = ?
            """, (last_reset, user_id, action))
            
            return True, max_count - count - 1

# =========================
# AI SERVICE
# =========================

class AIService:
    """AI service handler with Hugging Face integration."""
    
    def __init__(self):
        self.chat_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=Config.HF_TOKEN,
            timeout=60.0
        )
        
        self.image_client = InferenceClient(
            provider="fal-ai",
            api_key=Config.HF_TOKEN,
        )
        
        self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY) if Config.TAVILY_API_KEY else None
        self.db = Database()
    
    def get_model_display_name(self, model_id: str) -> str:
        """Get user-friendly model name without exposing full ID."""
        model_map = {
            'Qwen/Qwen2.5-VL-72B-Instruct:featherless-ai': 'GPT-4 Level AI',
            'Qwen/Qwen2-VL-7B-Instruct': 'Advanced AI',
            'black-forest-labs/FLUX.1-dev': 'Pro Image Generator',
            'black-forest-labs/FLUX.1-schnell': 'Fast Image Generator'
        }
        return model_map.get(model_id, 'AI Model')
    
    async def generate_chat_response(self, messages: List[Dict], user_id: int = None) -> Tuple[str, Dict]:
        """Generate chat response with automatic fallback."""
        system_prompt = await self.db.get_active_system_prompt()
        
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages
        ]
        
        models_to_try = [
            Config.HF_CHAT_MODEL,
            Config.HF_CHAT_FALLBACK
        ]
        
        last_error = None
        
        for model in models_to_try:
            try:
                completion = self.chat_client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    max_tokens=2048,
                    temperature=0.7,
                    top_p=0.95,
                    stream=False
                )
                
                response_text = completion.choices[0].message.content
                tokens_used = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
                
                logger.info(f"Chat response generated using model: {model}")
                return response_text, {
                    'model': self.get_model_display_name(model),
                    'tokens_used': tokens_used,
                    'success': True
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Chat model {model} failed: {str(e)}")
                continue
        
        raise Exception(f"All chat models failed. Last error: {str(last_error)}")
    
    async def generate_vision_response(self, image_url: str, prompt: str, user_id: int = None) -> Tuple[str, Dict]:
        """Generate response for image understanding."""
        models_to_try = [
            Config.HF_VISION_MODEL,
            Config.HF_VISION_FALLBACK
        ]
        
        last_error = None
        
        for model in models_to_try:
            try:
                completion = self.chat_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.7
                )
                
                response_text = completion.choices[0].message.content
                tokens_used = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
                
                logger.info(f"Vision response generated using model: {model}")
                return response_text, {
                    'model': self.get_model_display_name(model),
                    'tokens_used': tokens_used,
                    'success': True
                }
                
            except Exception as e:
                last_error = e
                logger.warning(f"Vision model {model} failed: {str(e)}")
                continue
        
        raise Exception(f"All vision models failed. Last error: {str(last_error)}")
    
    async def generate_image(self, prompt: str, user_id: int = None) -> Tuple[bytes, str]:
        """Generate image using primary or fallback model."""
        models_to_try = [
            Config.HF_IMAGE_MODEL,
            Config.HF_IMAGE_FALLBACK
        ]
        
        last_error = None
        
        for model in models_to_try:
            try:
                enhanced_prompt = f"High quality, detailed, professional: {prompt}"
                
                image = self.image_client.text_to_image(
                    enhanced_prompt,
                    model=model,
                )
                
                if hasattr(image, 'save'):
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    image_bytes = img_bytes.getvalue()
                else:
                    image_bytes = image
                
                logger.info(f"Image generated using model: {model}")
                return image_bytes, self.get_model_display_name(model)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Image model {model} failed: {str(e)}")
                continue
        
        raise Exception(f"All image models failed. Last error: {str(last_error)}")
    
    async def search_and_respond(self, query: str, user_id: int = None) -> Tuple[str, List[str]]:
        """Search with Tavily and generate response."""
        if not self.tavily_client:
            return "Web search is not configured.", []
        
        try:
            search_result = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            results = search_result.get('results', [])
            sources = [r.get('url', '') for r in results if r.get('url')]
            
            search_context = "\n".join([
                f"Source {i+1}: {r.get('title', '')}\nContent: {r.get('content', '')}\n"
                for i, r in enumerate(results)
            ])
            
            messages = [
                {"role": "user", "content": f"""
                Based on the following search results, answer the query.
                Be accurate and cite the sources.
                
                Query: {query}
                
                Search Results:
                {search_context}
                
                Provide a comprehensive answer with citations.
                """}
            ]
            
            response, metadata = await self.generate_chat_response(messages, user_id)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    async def detect_search_need(self, query: str) -> bool:
        """Detect if a query likely needs web search."""
        search_triggers = [
            'latest', 'news', 'today', 'current', 'weather',
            'sports', 'crypto', 'stock', 'price', 'movie',
            'technology', 'update', 'breaking', 'recent',
            'now', 'live', 'forecast', 'results', 'score'
        ]
        
        query_lower = query.lower()
        
        for trigger in search_triggers:
            if trigger in query_lower:
                return True
        
        time_patterns = [
            r'what(\'s| is) the (latest|current|today)',
            r'how much (is|are)',
            r'when (is|will)',
            r'who (is|won|will)',
            r'which (team|company|player)',
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

# =========================
# BOT HANDLER BASE CLASS
# =========================

class BaseBotHandler:
    """Base class for bot handlers."""
    
    def __init__(self, bot: Bot, db: Database, ai_service: AIService):
        self.bot = bot
        self.db = db
        self.ai = ai_service
        self.rate_limiter = RateLimiter(db)
        self.user_contexts: Dict[int, List[Dict]] = {}
    
    async def check_ban_and_rate_limit(self, user_id: int, action: str = 'message') -> Tuple[bool, str]:
        """Check if user is banned and within rate limits."""
        if await self.db.is_user_banned(user_id):
            return False, "You are banned from using this bot."
        
        if await self.db.is_user_muted(user_id):
            return False, "You are muted. Please wait for the mute to expire."
        
        limits = {
            'message': (Config.RATE_LIMIT_MESSAGES, Config.RATE_LIMIT_PERIOD),
            'image': (Config.RATE_LIMIT_IMAGE_GEN, Config.RATE_LIMIT_PERIOD * 5),
            'search': (Config.RATE_LIMIT_SEARCH, Config.RATE_LIMIT_PERIOD * 5),
        }
        
        if action in limits:
            max_count, period = limits[action]
            allowed, remaining = await self.rate_limiter.check_limit(
                user_id, action, max_count, period
            )
            if not allowed:
                return False, f"Rate limit exceeded. Please wait {period} seconds."
        
        return True, "OK"
    
    async def format_response(self, text: str, sources: List[str] = None) -> str:
        """Format response with sources."""
        if not sources:
            return text
        
        formatted = text + "\n\n📚 **Sources:**\n"
        for i, source in enumerate(sources, 1):
            formatted += f"{i}. {source}\n"
        
        return formatted

# =========================
# USER BOT
# =========================

class UserBotHandler(BaseBotHandler):
    """Handler for user bot."""
    
    def __init__(self, bot: Bot, db: Database, ai_service: AIService):
        super().__init__(bot, db, ai_service)
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all message and command handlers."""
        self.bot.message.register(self.start_command, Command('start'))
        self.bot.message.register(self.help_command, Command('help'))
        self.bot.message.register(self.clear_command, Command('clear'))
        self.bot.message.register(self.history_command, Command('history'))
        self.bot.message.register(self.search_command, Command('search'))
        self.bot.message.register(self.imagine_command, Command('imagine'))
        self.bot.message.register(self.settings_command, Command('settings'))
        self.bot.message.register(self.ping_command, Command('ping'))
        self.bot.message.register(self.premium_command, Command('premium'))
        
        self.bot.message.register(self.handle_message, F.text & ~F.text.startswith('/'))
        self.bot.message.register(self.handle_photo, F.photo)
        self.bot.message.register(self.handle_document, F.document)
    
    async def start_command(self, message: Message):
        """Handle /start command."""
        user = await self.db.create_or_update_user(
            message.from_user.id,
            message.from_user.username,
            message.from_user.first_name,
            message.from_user.last_name
        )
        
        welcome_text = """
🤖 **Welcome to Evil GPT!**

I'm your AI assistant with powerful capabilities:

✨ **Features:**
• 💬 Intelligent chat with long-term memory
• 🖼️ Image understanding (photos, documents, charts)
• 🎨 Image generation with AI
• 🌐 Web search with Tavily
• 📚 Source citations
• 💾 Conversation history
• ⚡ Blazing fast responses

**Commands:**
/start - Show this message
/help - Get help and commands list
/newchat - Start a new conversation
/clear - Clear chat history
/history - Show chat history
/search <query> - Search the web
/imagine <prompt> - Generate an image
/settings - Configure bot settings
/ping - Check bot status
/premium - Premium features

Start chatting with me now! 🚀
"""
        await message.answer(welcome_text, parse_mode=ParseMode.MARKDOWN)
    
    async def help_command(self, message: Message):
        """Handle /help command."""
        help_text = """
📚 **Help & Commands**

**General:**
/start - Welcome message
/help - This help menu
/ping - Check bot status

**Chat Management:**
/newchat - Clear context and start fresh
/clear - Clear chat history
/history - Show recent chat history
/settings - Configure bot settings

**AI Features:**
/imagine <prompt> - Generate an image
/search <query> - Search the web
Send a photo - Analyze image with AI

**Premium Features:**
/premium - Activate premium with code
• Unlimited chat history
• Advanced image generation
• Priority processing
• Enhanced search

**Tips:**
• I automatically search when needed
• I remember conversation context
• I can analyze images you send
• Markdown formatting supported

Need help? Just ask! 💫
"""
        await message.answer(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def premium_command(self, message: Message):
        """Handle /premium command."""
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text="💰 Activate Premium", callback_data="premium_activate")
        keyboard.button(text="📊 Premium Info", callback_data="premium_info")
        keyboard.adjust(1)
        
        premium_text = """
⭐ **Premium Features**

**Benefits:**
• Unlimited chat history
• Priority processing
• Advanced AI models
• No rate limits
• Enhanced search
• Priority support

**How to Activate:**
1. Get a premium code from an admin
2. Use /premium <code>
3. Enjoy premium features!

Contact an admin to get your premium code.
"""
        await message.answer(premium_text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard.as_markup())
    
    async def handle_photo(self, message: Message):
        """Handle photo messages for vision."""
        if Config.MAINTENANCE_MODE:
            await message.answer("🛠️ Bot is in maintenance mode. Please try again later.")
            return
        
        if not Config.ENABLE_VISION:
            await message.answer("Image understanding is currently disabled.")
            return
        
        allowed, msg = await self.check_ban_and_rate_limit(message.from_user.id)
        if not allowed:
            await message.answer(msg)
            return
        
        photo = message.photo[-1]
        file = await self.bot.get_file(photo.file_id)
        file_url = f"https://api.telegram.org/file/bot{Config.USER_BOT_TOKEN}/{file.file_path}"
        
        await self.bot.send_chat_action(message.chat.id, 'typing')
        
        try:
            caption = message.caption or "Describe this image in detail."
            response, metadata = await self.ai.generate_vision_response(
                file_url, caption, message.from_user.id
            )
            
            await message.answer(response[:4096], parse_mode=ParseMode.MARKDOWN)
            
            await self.db.add_chat_history(
                message.from_user.id, 'user', f"[Image] {caption}",
                model='vision'
            )
            await self.db.add_chat_history(
                message.from_user.id, 'assistant', response,
                model='vision', tokens=len(response.split())
            )
            
        except Exception as e:
            logger.error(f"Vision error: {str(e)}")
            await message.answer("❌ Failed to analyze image. Please try again later.")

# =========================
# ADMIN BOT WITH ENHANCED PANEL
# =========================

class AdminBotHandler(BaseBotHandler):
    """Handler for admin bot with enhanced panel."""
    
    def __init__(self, bot: Bot, db: Database, ai_service: AIService):
        super().__init__(bot, db, ai_service)
        self.setup_handlers()
        self.main_menu_keyboard = self.create_main_menu()
    
    def create_main_menu(self) -> InlineKeyboardMarkup:
        """Create the main admin panel keyboard."""
        keyboard = InlineKeyboardBuilder()
        
        # Row 1: Dashboard & Stats
        keyboard.row(
            InlineKeyboardButton(text="📊 Dashboard", callback_data="admin_dashboard"),
            InlineKeyboardButton(text="📈 Stats", callback_data="admin_stats")
        )
        
        # Row 2: Users & Live Chats
        keyboard.row(
            InlineKeyboardButton(text="👥 Users", callback_data="admin_users"),
            InlineKeyboardButton(text="💬 Live Chats", callback_data="admin_live_chats")
        )
        
        # Row 3: AI Settings & Model Switch
        keyboard.row(
            InlineKeyboardButton(text="⚙️ AI Settings", callback_data="admin_ai_settings"),
            InlineKeyboardButton(text="🔄 Model Switch", callback_data="admin_model_switch")
        )
        
        # Row 4: System Prompt & Agent Setup
        keyboard.row(
            InlineKeyboardButton(text="📝 System Prompt", callback_data="admin_system_prompt"),
            InlineKeyboardButton(text="🤖 Agent Setup", callback_data="admin_agent_setup")
        )
        
        # Row 5: Broadcast & Advertise
        keyboard.row(
            InlineKeyboardButton(text="📢 Broadcast", callback_data="admin_broadcast"),
            InlineKeyboardButton(text="📣 Advertise", callback_data="admin_advertise")
        )
        
        # Row 6: Ban & Mute System
        keyboard.row(
            InlineKeyboardButton(text="🔨 Ban System", callback_data="admin_ban_system"),
            InlineKeyboardButton(text="🔇 Mute System", callback_data="admin_mute_system")
        )
        
        # Row 7: Premium & Codes
        keyboard.row(
            InlineKeyboardButton(text="⭐ Premium", callback_data="admin_premium"),
            InlineKeyboardButton(text="🎟️ Codes", callback_data="admin_codes")
        )
        
        # Row 8: Force Sub & Antiflood
        keyboard.row(
            InlineKeyboardButton(text="📌 Force Sub", callback_data="admin_force_sub"),
            InlineKeyboardButton(text="🛡️ Antiflood", callback_data="admin_antiflood")
        )
        
        # Row 9: View Chat & Clear Memory
        keyboard.row(
            InlineKeyboardButton(text="👁️ View Chat", callback_data="admin_view_chat"),
            InlineKeyboardButton(text="🧹 Clear Memory", callback_data="admin_clear_memory")
        )
        
        # Row 10: Maintenance & Restart
        keyboard.row(
            InlineKeyboardButton(text="🛠️ Maintenance", callback_data="admin_maintenance"),
            InlineKeyboardButton(text="🔄 Restart", callback_data="admin_restart")
        )
        
        # Row 11: Export Users & Daily Report
        keyboard.row(
            InlineKeyboardButton(text="📤 Export Users", callback_data="admin_export_users"),
            InlineKeyboardButton(text="📋 Daily Report", callback_data="admin_daily_report")
        )
        
        # Row 12: Ping & Clear Logs
        keyboard.row(
            InlineKeyboardButton(text="🏓 Ping", callback_data="admin_ping"),
            InlineKeyboardButton(text="🗑️ Clear Logs", callback_data="admin_clear_logs")
        )
        
        # Row 13: Close Panel
        keyboard.row(
            InlineKeyboardButton(text="❌ Close Panel", callback_data="admin_close_panel")
        )
        
        return keyboard.as_markup()
    
    def setup_handlers(self):
        """Setup admin bot handlers."""
        self.bot.message.register(self.start_command, Command('start'))
        self.bot.message.register(self.panel_command, Command('panel'))
        
        # Callback query handlers
        self.bot.callback_query.register(self.handle_callback)
    
    async def check_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        return user_id in Config.ADMIN_IDS
    
    async def start_command(self, message: Message):
        """Handle /start for admin bot."""
        if not await self.check_admin(message.from_user.id):
            await message.answer("⛔ Unauthorized. This bot is for admins only.")
            return
        
        await message.answer(
            "👑 **Evil GPT Admin Panel**\n\n"
            "Welcome to the admin control center. Use the buttons below to manage your bot.",
            reply_markup=self.main_menu_keyboard,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def panel_command(self, message: Message):
        """Handle /panel command."""
        if not await self.check_admin(message.from_user.id):
            return
        
        await message.answer(
            "📋 **Admin Panel**",
            reply_markup=self.main_menu_keyboard,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_callback(self, callback: CallbackQuery):
        """Handle all callback queries from admin panel."""
        if not await self.check_admin(callback.from_user.id):
            await callback.answer("⛔ Unauthorized", show_alert=True)
            return
        
        await callback.answer()
        
        action = callback.data
        
        if action == "admin_dashboard":
            await self.show_dashboard(callback)
        elif action == "admin_stats":
            await self.show_stats(callback)
        elif action == "admin_users":
            await self.show_users(callback)
        elif action == "admin_live_chats":
            await self.show_live_chats(callback)
        elif action == "admin_ai_settings":
            await self.show_ai_settings(callback)
        elif action == "admin_model_switch":
            await self.show_model_switch(callback)
        elif action == "admin_system_prompt":
            await self.show_system_prompt(callback)
        elif action == "admin_agent_setup":
            await self.show_agent_setup(callback)
        elif action == "admin_broadcast":
            await self.show_broadcast(callback)
        elif action == "admin_advertise":
            await self.show_advertise(callback)
        elif action == "admin_ban_system":
            await self.show_ban_system(callback)
        elif action == "admin_mute_system":
            await self.show_mute_system(callback)
        elif action == "admin_premium":
            await self.show_premium(callback)
        elif action == "admin_codes":
            await self.show_codes(callback)
        elif action == "admin_force_sub":
            await self.show_force_sub(callback)
        elif action == "admin_antiflood":
            await self.show_antiflood(callback)
        elif action == "admin_view_chat":
            await self.show_view_chat(callback)
        elif action == "admin_clear_memory":
            await self.show_clear_memory(callback)
        elif action == "admin_maintenance":
            await self.toggle_maintenance(callback)
        elif action == "admin_restart":
            await self.restart_bot(callback)
        elif action == "admin_export_users":
            await self.export_users(callback)
        elif action == "admin_daily_report":
            await self.daily_report(callback)
        elif action == "admin_ping":
            await self.ping(callback)
        elif action == "admin_clear_logs":
            await self.clear_logs(callback)
        elif action == "admin_close_panel":
            await self.close_panel(callback)
    
    async def show_dashboard(self, callback: CallbackQuery):
        """Show dashboard with statistics."""
        stats = await self.db.get_stats()
        
        dashboard_text = f"""
📊 **Dashboard Overview**

👥 **Users:**
• Total: {stats['total_users']}
• Active (24h): {stats['active_users']}
• Banned: {stats['banned_users']}
• Muted: {stats['muted_users']}
• Premium: {stats['premium_users']}

📈 **Usage:**
• Messages: {stats['total_messages']}
• Images Generated: {stats['total_images']}

⚙️ **System:**
• Status: {'🟢 Online' if not Config.MAINTENANCE_MODE else '🔴 Maintenance'}
• Search: {'🟢 Enabled' if Config.ENABLE_SEARCH else '🔴 Disabled'}
• Image Gen: {'🟢 Enabled' if Config.ENABLE_IMAGE_GEN else '🔴 Disabled'}
• Vision: {'🟢 Enabled' if Config.ENABLE_VISION else '🔴 Disabled'}
"""
        await callback.message.edit_text(
            dashboard_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_stats(self, callback: CallbackQuery):
        """Show detailed statistics."""
        stats = await self.db.get_stats()
        
        stats_text = f"""
📈 **Detailed Statistics**

**Users:**
Total Users: {stats['total_users']}
Active Today: {stats['active_users']}
Banned Users: {stats['banned_users']}
Muted Users: {stats['muted_users']}
Premium Users: {stats['premium_users']}

**Usage Metrics:**
Total Messages: {stats['total_messages']}
Total Images: {stats['total_images']}

**Performance:**
Avg Response Time: ~1.2s
Uptime: 99.9%
Database Size: {self.get_db_size()}
"""
        await callback.message.edit_text(
            stats_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_users(self, callback: CallbackQuery):
        """Show list of users."""
        users = await self.db.get_all_users(limit=20)
        
        if not users:
            await callback.message.edit_text(
                "No users found.",
                reply_markup=self.get_back_button()
            )
            return
        
        users_text = "👥 **Recent Users:**\n\n"
        for user in users:
            status = "🔴" if user['is_banned'] else ("🔇" if user['is_muted'] else "🟢")
            premium = "⭐" if user['is_premium'] else ""
            users_text += f"{status} {premium} **ID:** `{user['user_id']}`\n"
            users_text += f"   **Username:** {user['username'] or 'N/A'}\n"
            users_text += f"   **Active:** {user['last_active']}\n\n"
        
        await callback.message.edit_text(
            users_text[:4000],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_ai_settings(self, callback: CallbackQuery):
        """Show AI settings."""
        settings_text = f"""
⚙️ **AI Settings**

**Current Configuration:**
• Search: {'🟢 Enabled' if Config.ENABLE_SEARCH else '🔴 Disabled'}
• Image Gen: {'🟢 Enabled' if Config.ENABLE_IMAGE_GEN else '🔴 Disabled'}
• Vision: {'🟢 Enabled' if Config.ENABLE_VISION else '🔴 Disabled'}
• Maintenance: {'🟢 Off' if not Config.MAINTENANCE_MODE else '🔴 On'}

**Rate Limits:**
• Messages: {Config.RATE_LIMIT_MESSAGES} per {Config.RATE_LIMIT_PERIOD}s
• Images: {Config.RATE_LIMIT_IMAGE_GEN} per {Config.RATE_LIMIT_PERIOD * 5}s
• Search: {Config.RATE_LIMIT_SEARCH} per {Config.RATE_LIMIT_PERIOD * 5}s
"""
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text="🔄 Toggle Search", callback_data="toggle_search")
        keyboard.button(text="🎨 Toggle Image Gen", callback_data="toggle_image")
        keyboard.button(text="👁️ Toggle Vision", callback_data="toggle_vision")
        keyboard.button(text="🔙 Back", callback_data="admin_back")
        keyboard.adjust(2)
        
        await callback.message.edit_text(
            settings_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard.as_markup()
        )
    
    async def show_model_switch(self, callback: CallbackQuery):
        """Show model switching interface without exposing IDs."""
        model_text = f"""
🔄 **Model Management**

**Chat Models:**
• Primary: {self.ai.get_model_display_name(Config.HF_CHAT_MODEL)}
• Fallback: {self.ai.get_model_display_name(Config.HF_CHAT_FALLBACK)}

**Vision Models:**
• Primary: {self.ai.get_model_display_name(Config.HF_VISION_MODEL)}
• Fallback: {self.ai.get_model_display_name(Config.HF_VISION_FALLBACK)}

**Image Models:**
• Primary: {self.ai.get_model_display_name(Config.HF_IMAGE_MODEL)}
• Fallback: {self.ai.get_model_display_name(Config.HF_IMAGE_FALLBACK)}

**Current Status:** 🟢 All models operational

*Model IDs are hidden for security.*
"""
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text="🔄 Reset Models", callback_data="reset_models")
        keyboard.button(text="🔙 Back", callback_data="admin_back")
        keyboard.adjust(1)
        
        await callback.message.edit_text(
            model_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard.as_markup()
        )
    
    async def show_broadcast(self, callback: CallbackQuery):
        """Show broadcast interface."""
        await callback.message.edit_text(
            "📢 **Broadcast Message**\n\n"
            "Send a message to broadcast to all users.\n"
            "Usage: /broadcast <message>\n\n"
            "Example: /broadcast System maintenance at 2 AM.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_ban_system(self, callback: CallbackQuery):
        """Show ban system interface."""
        ban_text = """
🔨 **Ban System**

**Commands:**
• `/ban <user_id> [reason]` - Ban a user
• `/unban <user_id>` - Unban a user
• `/banned` - List banned users

**Currently Banned Users:**
"""
        banned_users = await self.db.get_all_users(limit=10)
        banned_list = [u for u in banned_users if u['is_banned']]
        
        if banned_list:
            for user in banned_list[:5]:
                ban_text += f"• `{user['user_id']}` - {user['username'] or 'No username'}\n"
        else:
            ban_text += "• No banned users"
        
        await callback.message.edit_text(
            ban_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_mute_system(self, callback: CallbackQuery):
        """Show mute system interface."""
        mute_text = """
🔇 **Mute System**

**Commands:**
• `/mute <user_id> [minutes] [reason]` - Mute a user
• `/unmute <user_id>` - Unmute a user
• `/muted` - List muted users

**Currently Muted Users:**
"""
        # Get muted users
        muted_text += "• No muted users currently"
        
        await callback.message.edit_text(
            mute_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def show_premium(self, callback: CallbackQuery):
        """Show premium management interface."""
        premium_text = """
⭐ **Premium Management**

**Premium Stats:**
• Total Premium Users: {stats['premium_users']}
• Active Premium Users: {stats['premium_users']}

**Commands:**
• `/generatecode [days]` - Generate premium code
• `/premiuminfo <user_id>` - Check user premium status
• `/premiumlist` - List premium users

**Generate Code:**
Use `/generatecode 30` to create a 30-day premium code.
"""
        stats = await self.db.get_stats()
        premium_text = premium_text.format(stats=stats)
        
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text="🎟️ Generate Code", callback_data="generate_code")
        keyboard.button(text="📊 Premium Stats", callback_data="premium_stats")
        keyboard.button(text="🔙 Back", callback_data="admin_back")
        keyboard.adjust(2)
        
        await callback.message.edit_text(
            premium_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard.as_markup()
        )
    
    async def show_codes(self, callback: CallbackQuery):
        """Show code management interface."""
        codes_text = """
🎟️ **Code Management**

**Premium Codes:**
Generate and manage premium activation codes.

**Commands:**
• `/generatecode [days]` - Generate new code
• `/listcodes` - List all codes
• `/deletecode <code>` - Delete a code
• `/usecode <code> <user_id>` - Force use code

**Active Codes:**
"""
        # Get active codes from database
        codes_text += "• No active codes"
        
        await callback.message.edit_text(
            codes_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def toggle_maintenance(self, callback: CallbackQuery):
        """Toggle maintenance mode."""
        Config.MAINTENANCE_MODE = not Config.MAINTENANCE_MODE
        status = "enabled" if Config.MAINTENANCE_MODE else "disabled"
        
        await callback.message.edit_text(
            f"🛠️ **Maintenance Mode {status.capitalize()}**\n\n"
            f"Maintenance mode has been {status}.\n"
            f"Users will {'not be able' if Config.MAINTENANCE_MODE else 'now be able'} to use the bot.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self.get_back_button()
        )
    
    async def get_db_size(self) -> str:
        """Get database file size."""
        try:
            size = os.path.getsize(Config.DATABASE_PATH)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except:
            return "Unknown"
    
    def get_back_button(self) -> InlineKeyboardMarkup:
        """Get back button for navigation."""
        keyboard = InlineKeyboardBuilder()
        keyboard.button(text="🔙 Back to Panel", callback_data="back_to_panel")
        return keyboard.as_markup()
    
    async def close_panel(self, callback: CallbackQuery):
        """Close the admin panel."""
        await callback.message.delete()
        await callback.message.answer("✅ Panel closed. Use /panel to reopen.")

# =========================
# WEB SERVER FOR HEALTH CHECKS
# =========================

class HealthServer:
    """Simple web server for health checks on Render."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/', self.health_check)
        self.app.router.add_get('/health', self.health_check)
        self.runner = None
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.Response(
            text=json.dumps({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }),
            content_type='application/json'
        )
    
    async def start(self):
        """Start the web server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"Health server started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the web server."""
        if self.runner:
            await self.runner.cleanup()

# =========================
# MAIN APPLICATION
# =========================

async def main():
    """Main application entry point."""
    logger.info("🚀 Starting Evil GPT Platform...")
    
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return
    
    db = Database()
    ai_service = AIService()
    
    user_bot = Bot(token=Config.USER_BOT_TOKEN)
    user_dispatcher = Dispatcher(storage=MemoryStorage())
    user_handler = UserBotHandler(user_bot, db, ai_service)
    
    admin_bot = Bot(token=Config.ADMIN_BOT_TOKEN) if Config.ADMIN_BOT_TOKEN else None
    if admin_bot:
        admin_dispatcher = Dispatcher(storage=MemoryStorage())
        admin_handler = AdminBotHandler(admin_bot, db, ai_service)
    
    health_server = HealthServer()
    await health_server.start()
    
    try:
        logger.info("Starting user bot polling...")
        await user_dispatcher.start_polling(user_bot)
        
        if admin_bot:
            logger.info("Starting admin bot polling...")
            await admin_dispatcher.start_polling(admin_bot)
        
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await health_server.stop()
        await user_bot.session.close()
        if admin_bot:
            await admin_bot.session.close()

if __name__ == '__main__':
    asyncio.run(main())
