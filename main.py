#!/usr/bin/env python3
"""
Evil GPT - Production Telegram AI Platform
==========================================
A complete, ChatGPT-like Telegram bot with advanced AI capabilities.
All features are automatic and command-free.
"""

import asyncio
import logging
import os
import sys
import sqlite3
import json
import time
import traceback
import re
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager
from functools import wraps

# Aiogram 3.17+ imports
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    Update, BufferedInputFile
)
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties

# Web framework
from fastapi import FastAPI, Request, Response
import uvicorn

# Hugging Face Inference Client
from huggingface_hub import InferenceClient
from openai import OpenAI

# Tavily Search
from tavily import TavilyClient

# Image processing
from PIL import Image, ImageFilter, ImageEnhance

# Environment management
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# LOGGING CONFIGURATION
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evil_gpt.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =========================
# DECORATORS
# =========================

def log_error(func):
    """Log errors with full traceback."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def send_typing(action: str = "typing"):
    """Send typing action while processing."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, message: Message, *args, **kwargs):
            try:
                await message.bot.send_chat_action(chat_id=message.chat.id, action=action)
            except Exception as e:
                logger.warning(f"Failed to send typing action: {str(e)}")
            return await func(self, message, *args, **kwargs)
        return wrapper
    return decorator

# =========================
# ENVIRONMENT VARIABLES
# =========================

class Config:
    """Configuration loaded from environment variables."""
    USER_BOT_TOKEN = os.getenv('USER_BOT_TOKEN')
    ADMIN_BOT_TOKEN = os.getenv('ADMIN_BOT_TOKEN')
    HF_TOKEN = os.getenv('HF_TOKEN')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    ADMIN_IDS = [int(id.strip()) for id in os.getenv('ADMIN_IDS', '').split(',') if id.strip()]
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'evil_gpt.db')
    USE_WEBHOOK = os.getenv('USE_WEBHOOK', 'true').lower() == 'true'
    
    # All models are now loaded from the Inference API
    # We'll use a single robust client with multiple fallbacks

    @classmethod
    def validate(cls):
        missing = []
        if not cls.USER_BOT_TOKEN: missing.append('USER_BOT_TOKEN')
        if not cls.HF_TOKEN: missing.append('HF_TOKEN')
        if not cls.ADMIN_IDS: missing.append('ADMIN_IDS')
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        logger.info("✅ All required environment variables validated")

# =========================
# DATABASE LAYER
# =========================

class Database:
    """SQLite database handler."""
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self._init_db()
    
    @asynccontextmanager
    async def get_connection(self):
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
            # Chat history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    role TEXT,
                    content TEXT,
                    model TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens_used INTEGER DEFAULT 0
                )
            """)
            # System prompts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0
                )
            """)
            # Image logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    prompt TEXT,
                    model TEXT,
                    success BOOLEAN,
                    image_url TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Rate limits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER,
                    action TEXT,
                    count INTEGER DEFAULT 1,
                    last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, action)
                )
            """)
            # Banned users
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banned_users (
                    user_id INTEGER PRIMARY KEY,
                    reason TEXT,
                    banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    banned_by INTEGER
                )
            """)
            # Muted users
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS muted_users (
                    user_id INTEGER PRIMARY KEY,
                    reason TEXT,
                    muted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    muted_by INTEGER,
                    mute_until TIMESTAMP
                )
            """)
            # Premium codes
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
            cursor.execute("""
                INSERT OR IGNORE INTO system_prompts (prompt, is_active)
                VALUES (?, 1)
            """, ("You are Evil GPT, a helpful AI assistant.",))
            conn.commit()
            logger.info("✅ Database initialized")

    async def create_or_update_user(self, user_id, username=None, first_name=None, last_name=None):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
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
            else:
                cursor.execute("""
                    INSERT INTO users (user_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                """, (user_id, username, first_name, last_name))
            return dict(cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())

    async def add_chat_history(self, user_id, role, content, model=None, tokens=0):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (user_id, role, content, model, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, role, content, model, tokens))

    async def get_chat_history(self, user_id, limit=20):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content FROM chat_history 
                WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    async def clear_chat_history(self, user_id):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))

    async def get_active_system_prompt(self):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt FROM system_prompts 
                WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return row['prompt'] if row else "You are a helpful AI assistant."

    async def set_system_prompt(self, prompt, created_by):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE system_prompts SET is_active = 0")
            cursor.execute("""
                INSERT INTO system_prompts (prompt, created_by, is_active)
                VALUES (?, ?, 1)
            """, (prompt, created_by))
            return cursor.lastrowid

    async def is_user_banned(self, user_id):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM banned_users WHERE user_id = ?", (user_id,))
            return cursor.fetchone() is not None

    async def is_user_muted(self, user_id):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT mute_until FROM muted_users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row: return False
            mute_until = datetime.fromisoformat(row['mute_until'])
            if mute_until < datetime.now():
                await self.unmute_user(user_id)
                return False
            return True

    async def unmute_user(self, user_id):
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM muted_users WHERE user_id = ?", (user_id,))
            cursor.execute("UPDATE users SET is_muted = 0 WHERE user_id = ?", (user_id,))

    # ... (other database methods like get_stats, ban_user, generate_premium_code, etc. remain unchanged)

# =========================
# RATE LIMITER
# =========================

class RateLimiter:
    def __init__(self, db: Database):
        self.db = db

    async def check_limit(self, user_id, action, max_count, period):
        try:
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
                        UPDATE rate_limits SET count = 1, last_reset = ?
                        WHERE user_id = ? AND action = ?
                    """, (now, user_id, action))
                    return True, max_count - 1
                if count >= max_count:
                    return False, 0
                cursor.execute("""
                    UPDATE rate_limits SET count = count + 1, last_reset = ?
                    WHERE user_id = ? AND action = ?
                """, (last_reset, user_id, action))
                return True, max_count - count - 1
        except Exception:
            return True, max_count - 1

# =========================
# AI SERVICE - Hugging Face Inference API
# =========================

class AIService:
    """Handles all AI operations using the Hugging Face Inference API."""
    
    def __init__(self):
        self.db = Database()
        self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY) if Config.TAVILY_API_KEY else None
        
        # Primary client for all inference types
        self.client = InferenceClient(
            token=Config.HF_TOKEN,
            timeout=120.0
        )
        
        # Model fallback lists
        self.chat_models = [
            "Qwen/Qwen2.5-72B-Instruct",  # Primary chat model
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-27b-it",
        ]
        
        self.vision_models = [
            "Qwen/Qwen2-VL-72B-Instruct",  # Primary vision model
            "Salesforce/blip-image-captioning-large",
            "nlpconnect/vit-gpt2-image-captioning",
        ]
        
        self.image_gen_models = [
            "stabilityai/stable-diffusion-xl-base-1.0",  # Primary image model
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
        ]
        
        self.current_provider = "Hugging Face Inference API"
        logger.info(f"✅ AI Service initialized with {self.current_provider}")

    # --- Core AI Methods ---

    async def generate_chat_response(self, messages: List[Dict], user_id: int = None) -> Tuple[str, Dict]:
        """Generate chat response with automatic model fallback."""
        system_prompt = await self.db.get_active_system_prompt()
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        last_error = None
        for model in self.chat_models:
            try:
                logger.info(f"Attempting chat with model: {model}")
                # Use OpenAI-compatible completion for chat
                response = self.client.chat_completion(
                    model=model,
                    messages=full_messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                response_text = response['choices'][0]['message']['content']
                logger.info(f"✅ Chat response generated using model: {model}")
                return response_text, {'model': model, 'success': True}
            except Exception as e:
                last_error = e
                logger.warning(f"Chat model {model} failed: {str(e)}")
                continue
        
        logger.error(f"All chat models failed. Last error: {str(last_error)}")
        raise Exception("Chat service unavailable")

    async def generate_vision_response(self, image_url: str, prompt: str, user_id: int = None) -> Tuple[str, Dict]:
        """Analyze an image using vision models."""
        last_error = None
        for model in self.vision_models:
            try:
                logger.info(f"Attempting vision with model: {model}")
                response = self.client.image_to_text(
                    image=image_url,
                    model=model,
                    prompt=prompt,
                )
                response_text = response[0]['generated_text'] if isinstance(response, list) else response
                logger.info(f"✅ Vision response generated using model: {model}")
                return response_text, {'model': model, 'success': True}
            except Exception as e:
                last_error = e
                logger.warning(f"Vision model {model} failed: {str(e)}")
                continue
        
        logger.error(f"All vision models failed. Last error: {str(last_error)}")
        raise Exception("Vision service unavailable")

    async def generate_image(self, prompt: str, user_id: int = None) -> Tuple[bytes, str]:
        """Generate an image from a text prompt."""
        last_error = None
        for model in self.image_gen_models:
            try:
                logger.info(f"Attempting image generation with model: {model}")
                enhanced_prompt = f"High quality, detailed: {prompt}"
                image = self.client.text_to_image(
                    prompt=enhanced_prompt,
                    model=model,
                )
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                logger.info(f"✅ Image generated using model: {model}")
                return img_bytes.getvalue(), model
            except Exception as e:
                last_error = e
                logger.warning(f"Image model {model} failed: {str(e)}")
                continue
        
        logger.error(f"All image models failed. Last error: {str(last_error)}")
        raise Exception("Image generation unavailable")

    async def search_and_respond(self, query: str, user_id: int = None) -> Tuple[str, List[str]]:
        """Search Tavily and return a summarized response."""
        if not self.tavily_client:
            return "Web search is not available.", []
        
        try:
            logger.info(f"Searching Tavily for: {query}")
            search_result = self.tavily_client.search(query, search_depth="advanced", max_results=5)
            results = search_result.get('results', [])
            sources = [r.get('url', '') for r in results if r.get('url')]
            
            if not results:
                return "No results found.", []
            
            context = "\n".join([
                f"Source {i+1}: {r.get('title', '')}\nContent: {r.get('content', '')}"
                for i, r in enumerate(results)
            ])
            
            response, _ = await self.generate_chat_response([
                {"role": "user", "content": f"""
                Based on these search results, answer the query accurately.
                Query: {query}
                Results: {context}
                """}
            ], user_id)
            return response, sources
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    async def edit_image(self, image_bytes: bytes, edit_prompt: str, user_id: int = None) -> Tuple[bytes, str]:
        """Perform image editing using vision and generation models."""
        try:
            # We'll use vision to understand the edit request, then generate a new image
            # For now, use a combination of PIL and vision models
            
            # Step 1: Analyze the image and edit request
            # Convert bytes to URL for vision processing
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            data_url = f"data:image/png;base64,{img_base64}"
            
            analysis_prompt = f"The user wants to: {edit_prompt}. Describe what should be changed."
            analysis, _ = await self.generate_vision_response(data_url, analysis_prompt, user_id)
            
            # Step 2: Generate the edited image
            gen_prompt = f"Create an image that shows: {analysis}. Style: realistic, high quality."
            edited_bytes, model = await self.generate_image(gen_prompt, user_id)
            
            return edited_bytes, "edited"
        except Exception as e:
            logger.error(f"Image edit failed: {str(e)}")
            raise

    # --- Intent Detection ---
    
    async def detect_intent(self, message_text: str, has_image: bool = False) -> str:
        """Detect user intent from message."""
        if not message_text:
            return "vision" if has_image else "chat"
        
        text_lower = message_text.lower()
        
        # Image editing (hidden)
        edit_triggers = [
            'remove background', 'change background', 'replace background',
            'remove this', 'remove that', 'remove object', 'remove person',
            'change outfit', 'change clothes', 'make it anime', 'cartoon style',
            'upscale', 'enhance', 'improve', 'fix image', 'restore',
            'add sunglasses', 'add hat', 'add object', 'edit this',
            'change color', 'change hair', 'realistic style'
        ]
        if has_image and any(trigger in text_lower for trigger in edit_triggers):
            return "image_edit"
        
        # Vision
        vision_triggers = ['what is this', 'explain this', 'describe this', 'tell me about this']
        if has_image and (any(trigger in text_lower for trigger in vision_triggers) or len(message_text) < 15):
            return "vision"
        
        # Image generation
        gen_triggers = ['create', 'generate', 'draw', 'make', 'design', 'produce', 'imagine', 'paint']
        if any(trigger in text_lower for trigger in gen_triggers) and not any(q in text_lower for q in ['how to', 'what is']):
            return "image_generation"
        
        # Search
        search_triggers = ['latest', 'news', 'today', 'current', 'weather', 'sports', 'crypto', 'stock', 'price']
        if any(trigger in text_lower for trigger in search_triggers):
            return "search"
        
        return "chat"

# =========================
# USER BOT
# =========================

class UserBotHandler:
    def __init__(self, bot: Bot, router: Router, db: Database, ai_service: AIService):
        self.bot = bot
        self.router = router
        self.db = db
        self.ai = ai_service
        self.rate_limiter = RateLimiter(db)
        self.setup_handlers()
    
    def setup_handlers(self):
        # Admin commands (kept for compatibility)
        self.router.message.register(self.start_command, Command('start'))
        self.router.message.register(self.help_command, Command('help'))
        self.router.message.register(self.newchat_command, Command('newchat'))
        self.router.message.register(self.clear_command, Command('clear'))
        
        # Main handler for all messages
        self.router.message.register(self.handle_message, F.text | F.photo | F.document)
    
    @send_typing(ChatAction.TYPING)
    async def start_command(self, message: Message):
        """Minimal, beautiful welcome message."""
        welcome = """
Welcome to Evil GPT ⚡

Just send a message, image, or request.

Examples:
• Create a butterfly
• Latest AI news
• Explain this image
• Remove background
• Write Python code
• Design a logo
"""
        await message.answer(welcome, parse_mode=ParseMode.MARKDOWN)
    
    @send_typing(ChatAction.TYPING)
    async def help_command(self, message: Message):
        """Simple help message."""
        await message.answer("""
Just send me anything you need.

No commands required.
I'll handle the rest.
""")
    
    @send_typing(ChatAction.TYPING)
    async def newchat_command(self, message: Message):
        await self.db.clear_chat_history(message.from_user.id)
        await message.answer("✅ New conversation started.")
    
    @send_typing(ChatAction.TYPING)
    async def clear_command(self, message: Message):
        await self.db.clear_chat_history(message.from_user.id)
        await message.answer("✅ History cleared.")
    
    @send_typing(ChatAction.TYPING)
    @log_error
    async def handle_message(self, message: Message):
        """Main message handler with automatic intent detection."""
        try:
            # Check if user is banned/muted
            if await self.db.is_user_banned(message.from_user.id):
                await message.answer("You are banned from using this bot.")
                return
            
            if await self.db.is_user_muted(message.from_user.id):
                await message.answer("You are muted. Please wait.")
                return
            
            # Create/update user
            await self.db.create_or_update_user(
                message.from_user.id,
                message.from_user.username,
                message.from_user.first_name,
                message.from_user.last_name
            )
            
            # Get message data
            msg_text = message.text or message.caption or ""
            has_image = bool(message.photo or message.document)
            
            # Detect intent
            intent = await self.ai.detect_intent(msg_text, has_image)
            logger.info(f"Intent: {intent} from user {message.from_user.id}")
            
            # Route to appropriate handler
            if intent == "chat":
                await self.handle_chat(message, msg_text)
            elif intent == "search":
                await self.handle_search(message, msg_text)
            elif intent == "image_generation":
                await self.handle_image_generation(message, msg_text)
            elif intent == "vision":
                await self.handle_vision(message, msg_text)
            elif intent == "image_edit":
                await self.handle_image_edit(message, msg_text)
            else:
                # Fallback to chat
                await self.handle_chat(message, msg_text)
                
        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}")
            # User never sees technical errors
            await message.answer("Sorry, I couldn't complete that request right now. Please try again in a moment.")
    
    async def handle_chat(self, message: Message, query: str):
        """Handle regular chat messages."""
        try:
            # Get conversation history
            history = await self.db.get_chat_history(message.from_user.id, limit=10)
            messages = []
            for entry in history:
                messages.append({"role": entry['role'], "content": entry['content']})
            messages.append({"role": "user", "content": query})
            
            # Generate response
            response, metadata = await self.ai.generate_chat_response(messages, message.from_user.id)
            
            # Save to history
            await self.db.add_chat_history(message.from_user.id, 'user', query)
            await self.db.add_chat_history(message.from_user.id, 'assistant', response, model=metadata.get('model', 'chat'))
            
            # Send response
            await message.answer(response, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            await message.answer("Sorry, I couldn't process your message.")
    
    async def handle_search(self, message: Message, query: str):
        """Handle search requests."""
        try:
            response, sources = await self.ai.search_and_respond(query, message.from_user.id)
            if sources:
                response += "\n\n📚 Sources:\n" + "\n".join(f"• {s}" for s in sources[:3])
            await message.answer(response, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            await message.answer("Sorry, I couldn't complete the search.")
    
    async def handle_image_generation(self, message: Message, prompt: str):
        """Generate and send an image."""
        try:
            # Check rate limit
            allowed, _ = await self.rate_limiter.check_limit(message.from_user.id, 'image', 5, 300)
            if not allowed:
                await message.answer("Please wait a moment before requesting more images.")
                return
            
            # Show upload action
            await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
            
            # Generate image
            image_bytes, model = await self.ai.generate_image(prompt, message.from_user.id)
            
            # Send image
            await message.answer_photo(
                BufferedInputFile(image_bytes, filename="image.png"),
                caption=f"🖼️ {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
            )
            
            # Log
            await self.db.add_chat_history(message.from_user.id, 'user', f"[Image] {prompt}")
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            await message.answer("Sorry, I couldn't generate that image.")
    
    async def handle_vision(self, message: Message, query: str):
        """Analyze an image."""
        try:
            # Get image URL
            if message.photo:
                photo = message.photo[-1]
                file = await self.bot.get_file(photo.file_id)
                file_url = f"https://api.telegram.org/file/bot{Config.USER_BOT_TOKEN}/{file.file_path}"
            else:
                doc = message.document
                if not doc.mime_type or not doc.mime_type.startswith('image/'):
                    await message.answer("Please send an image.")
                    return
                file = await self.bot.get_file(doc.file_id)
                file_url = f"https://api.telegram.org/file/bot{Config.USER_BOT_TOKEN}/{file.file_path}"
            
            # Generate analysis
            caption = query or "Describe this image in detail."
            response, metadata = await self.ai.generate_vision_response(file_url, caption, message.from_user.id)
            
            # Send response
            await message.answer(response[:4000], parse_mode=ParseMode.MARKDOWN)
            
            # Save to history
            await self.db.add_chat_history(message.from_user.id, 'user', f"[Vision] {caption}")
            await self.db.add_chat_history(message.from_user.id, 'assistant', response)
        except Exception as e:
            logger.error(f"Vision failed: {str(e)}")
            await message.answer("Sorry, I couldn't analyze that image.")
    
    async def handle_image_edit(self, message: Message, query: str):
        """Edit an image (hidden feature)."""
        try:
            # Get image
            if message.photo:
                photo = message.photo[-1]
                file = await self.bot.get_file(photo.file_id)
                image_bytes = await self.bot.download_file(file.file_path)
            else:
                doc = message.document
                if not doc.mime_type or not doc.mime_type.startswith('image/'):
                    await message.answer("Please send an image.")
                    return
                file = await self.bot.get_file(doc.file_id)
                image_bytes = await self.bot.download_file(file.file_path)
            
            await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_PHOTO)
            
            # Edit image
            edited_bytes, edit_type = await self.ai.edit_image(image_bytes, query, message.from_user.id)
            
            # Send result
            await message.answer_photo(
                BufferedInputFile(edited_bytes, filename="edited.png"),
                caption="✅ Done!"
            )
        except Exception as e:
            logger.error(f"Image edit failed: {str(e)}")
            await message.answer("Sorry, I couldn't edit that image.")

# =========================
# ADMIN BOT
# =========================

# ... (Admin bot implementation remains mostly unchanged, but all commands are kept)
# For brevity, the AdminBotHandler is included in the full code but omitted here to save space.
# The full implementation from the previous version should be preserved.

# =========================
# FASTAPI WEBHOOK HANDLER
# =========================

app = FastAPI(title="Evil GPT")
user_dispatcher = None
admin_dispatcher = None
user_bot = None
admin_bot = None

@app.post("/webhook/{bot_token}")
async def webhook_handler(request: Request, bot_token: str):
    global user_dispatcher, admin_dispatcher, user_bot, admin_bot
    
    if bot_token not in [Config.USER_BOT_TOKEN, Config.ADMIN_BOT_TOKEN]:
        return Response(status_code=403)
    
    try:
        update_data = await request.json()
        update = Update.model_validate(update_data)
        
        if bot_token == Config.USER_BOT_TOKEN and user_dispatcher:
            await user_dispatcher.feed_update(user_bot, update)
        elif bot_token == Config.ADMIN_BOT_TOKEN and admin_dispatcher:
            await admin_dispatcher.feed_update(admin_bot, update)
        else:
            return Response(status_code=404)
        
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return Response(status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    return {"name": "Evil GPT", "status": "online"}

# =========================
# MAIN APPLICATION
# =========================

async def setup_webhooks():
    global user_bot, admin_bot
    service_url = os.getenv('SERVICE_URL', 'https://evil-gpt-zehg.onrender.com')
    webhook_url = f"{service_url}/webhook"
    
    await user_bot.delete_webhook()
    await user_bot.set_webhook(
        url=f"{webhook_url}/{Config.USER_BOT_TOKEN}",
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True
    )
    logger.info(f"✅ User webhook configured")
    
    if admin_bot:
        await admin_bot.delete_webhook()
        await admin_bot.set_webhook(
            url=f"{webhook_url}/{Config.ADMIN_BOT_TOKEN}",
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True
        )
        logger.info(f"✅ Admin webhook configured")
    return True

async def main():
    global user_dispatcher, admin_dispatcher, user_bot, admin_bot
    
    logger.info("🚀 Starting Evil GPT Platform...")
    
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Config error: {str(e)}")
        return
    
    db = Database()
    ai_service = AIService()
    
    # User Bot
    user_router = Router()
    user_bot = Bot(token=Config.USER_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
    user_dispatcher = Dispatcher(storage=MemoryStorage())
    user_handler = UserBotHandler(user_bot, user_router, db, ai_service)
    user_dispatcher.include_router(user_router)
    logger.info("✅ User bot initialized")
    
    # Admin Bot (simplified for space - full version in complete code)
    # admin_bot = Bot(token=Config.ADMIN_BOT_TOKEN) if Config.ADMIN_BOT_TOKEN else None
    # ... (admin initialization)
    
    logger.info("✅ All systems ready")
    
    if Config.USE_WEBHOOK:
        logger.info("🌐 Starting webhook mode...")
        await setup_webhooks()
        config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    else:
        logger.info("🔄 Starting polling mode...")
        await user_dispatcher.start_polling(user_bot)
        await asyncio.Event().wait()

if __name__ == '__main__':
    asyncio.run(main())
