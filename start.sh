#!/bin/bash

echo "🚀 Running LDS AI PDF indexer..."
python lds_ai.py

echo "💬 Starting WhatsApp Bot..."
python app/whatsapp_bot.py
