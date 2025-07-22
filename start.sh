echo "🚀 Running LDS AI PDF indexer..."
python lds_ai.py

# Wait until required files are created (example: output.json and data.db)
while [ ! -f "output.json" ] || [ ! -f "data.db" ]; do
    echo "⏳ Waiting for LDS AI to finish generating files..."
    sleep 2
done

echo "✅ Files ready. Starting WhatsApp Bot..."
python whatsapp_bot.py
