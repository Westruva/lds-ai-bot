#!/bin/bash

echo "🚀 Running LDS AI PDF indexer..."
python app/lds_ai.py || { echo "❌ lds_ai.py failed"; exit 1; }

echo "💬 Starting WhatsApp Bot..."
python app/whatsapp_bot.py || { echo "❌ whatsapp_bot.py failed"; exit 1; }
