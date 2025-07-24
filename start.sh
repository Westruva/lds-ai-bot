#!/bin/bash

echo "📖 Running LDS AI indexer..."
python app/lds_ai.py

echo "💬 Starting WhatsApp bot..."
python app/whatsapp_bot.py
