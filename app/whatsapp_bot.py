import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

# Set up environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

index_dir = "/faiss_index"

# Check if the index file exists
if not Path(index_dir).joinpath("index.faiss").exists():
    raise FileNotFoundError(f"❌ FAISS index not found at {index_dir}/index.faiss. Run lds_ai.py first to create it.")

vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
# Load vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = vectorstore.as_retriever()

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat model
chat_model = ChatOpenAI(temperature=0)

# Create chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory
)

# Flask setup
app = Flask(__name__)

# Unified webhook handler
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    print(f"[WhatsApp Incoming]: {incoming_msg}")

    if incoming_msg:
        result = qa_chain.invoke({"question": incoming_msg})
        reply = result["answer"]
    else:
        reply = "Please send a message to begin."

    print(f"[LDS AI Reply]: {reply}")
    twilio_response = MessagingResponse()
    twilio_response.message(reply)
    return str(twilio_response)

# Default POST route (matches Twilio default URL `/`)
@app.route("/", methods=["POST"])
def root_webhook():
    return whatsapp_webhook()

# Optional: Custom POST route `/webhook`
@app.route("/webhook", methods=["POST"])
def webhook():
    return whatsapp_webhook()

if __name__ == "__main__":
    app.run(debug=True)
