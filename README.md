# Evil GPT - Production Telegram AI Platform

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![Aiogram Version](https://img.shields.io/badge/aiogram-3.5.0-green.svg)](https://docs.aiogram.dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Render](https://img.shields.io/badge/deploy-Render-purple.svg)](https://render.com)

## 🚀 Overview

**Evil GPT** is a comprehensive, production-ready Telegram AI platform featuring two bots: a User Bot and an Admin Bot. It leverages Hugging Face models for chat, vision, and image generation, plus Tavily Search for real-time web information. The platform is designed for scalability, security, and ease of deployment.

### 🌟 Key Features

- **💬 Intelligent Chat**: Powered by Qwen models with automatic fallback
- **🖼️ Image Understanding**: Analyze photos, documents, screenshots, and charts
- **🎨 Image Generation**: Create stunning images with FLUX models
- **🌐 Web Search**: Automatic or explicit search with Tavily
- **📚 Source Citations**: All search results include sources
- **💾 Conversation Memory**: Long-term context retention
- **🔒 Premium System**: Monetization with premium codes
- **👑 Admin Panel**: Complete bot management dashboard
- **⚡ Rate Limiting**: Prevent abuse and ensure fair usage
- **🛡️ Security**: No API keys or model IDs exposed

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Bot Commands](#-bot-commands)
- [Admin Panel](#-admin-panel)
- [Deployment](#-deployment)
- [Database Schema](#-database-schema)
- [API Integration](#-api-integration)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### User Bot Features

| Feature | Description |
|---------|-------------|
| **Chat** | AI-powered conversations with context memory |
| **Vision** | Analyze images, photos, documents, and charts |
| **Image Generation** | Create AI-generated images from text prompts |
| **Web Search** | Real-time search with Tavily API |
| **Source Citations** | Automatic source attribution for search results |
| **Conversation History** | Store and retrieve chat history |
| **Markdown Support** | Full markdown with syntax highlighting |
| **Premium Features** | Unlimited chat, priority processing, advanced models |
| **Rate Limiting** | Fair usage protection for all users |

### Admin Bot Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Real-time statistics and system status |
| **User Management** | Ban, mute, and manage users |
| **Premium System** | Generate and manage premium codes |
| **Broadcast** | Send messages to all users |
| **AI Settings** | Configure AI models and features |
| **System Prompt** | Update global AI personality |
| **Maintenance Mode** | Take bot offline for updates |
| **Logs Management** | View and clear system logs |
| **Export Users** | Export user data for analysis |
| **Daily Reports** | Automated statistics reports |

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12 | Core programming language |
| **Aiogram** | 3.5.0 | Telegram Bot API framework |
| **SQLite** | - | Lightweight database |
| **Hugging Face** | - | AI model inference |
| **Tavily** | - | Web search API |
| **Render** | - | Cloud deployment |

### AI Models

| Model | Purpose | Provider |
|-------|---------|----------|
| Qwen/Qwen2.5-VL-72B-Instruct | Chat & Vision (Primary) | Hugging Face |
| Qwen/Qwen2-VL-7B-Instruct | Chat & Vision (Fallback) | Hugging Face |
| black-forest-labs/FLUX.1-dev | Image Generation (Primary) | Hugging Face |
| black-forest-labs/FLUX.1-schnell | Image Generation (Fallback) | Hugging Face |

---

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Telegram Bot Tokens (User + Admin)
- Hugging Face API Token
- Tavily Search API Key (optional)

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/evil-gpt.git
cd evil-gpt
