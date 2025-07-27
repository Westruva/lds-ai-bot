#!/bin/bash

echo "🧠 Rebuilding FAISS index..."
python app/lds_ai.py

echo "💬 Starting WhatsApp bot..."
python app/whatsapp_bot.py
