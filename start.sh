#!/bin/bash

ls -al /app
echo "🧠 Launching LDS AI..."
python app/lds_ai.py

echo "💬 Starting WhatsApp Bot..."
python app/whatsapp_bot.py
