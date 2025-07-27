import os
import pickle
import time
import subprocess
from datetime import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# 🗂️ CONFIG
load_dotenv() 
pdf_folder = "data"
index_dir = "faiss_index"
processed_file_log = "processed_files.pkl"
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Set up embeddings

embeddings = OpenAIEmbeddings()

# 📘 Load PDFs
def load_pdfs(folder):
    docs = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            print(f"📘 Reading: {filename}")
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            filenames.append(filename)
    return docs, filenames

# 🧩 Split documents
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# 🧠 Embed in safe batches
def embed_in_batches(docs, batch_size=100):
    total = len(docs)
    for i in range(0, total, batch_size):
        yield docs[i:i+batch_size]


# 🧠 Build or load vector store
def build_or_load_vector_store(new_chunks):
    vectorstore = None

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        print(f"📁 Created missing directory: {index_dir}")

    try:
        if os.path.exists(f"{index_dir}/index.faiss"):
            print("📂 FAISS index file found. Loading existing index...")
            vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            print("✅ Successfully loaded FAISS index.")
        else:
            print("⚠️ No existing FAISS index found.")
    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}")

    if new_chunks:
        print(f"🧾 New Chunks to Embed: {len(new_chunks)}")

        for i, batch in enumerate(embed_in_batches(new_chunks)):
            if not batch:
                continue

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
                print(f"🔧 Created new FAISS index with batch {i+1}")
            else:
                vectorstore.add_documents(batch)
                print(f"➕ Added batch {i+1} to existing index")

            time.sleep(1)

        vectorstore.save_local(index_dir)
        print(f"✅ FAISS index saved to '{index_dir}'")
        try:
            print("📦 Saved index files:", os.listdir(index_dir))
        except Exception as e:
            print(f"⚠️ Could not list index_dir contents: {e}")
    else:
        print("⚠️ No new chunks — skipping FAISS index save.")

    return vectorstore


# 💾 Load or initialize processed files log
if os.path.exists(processed_file_log):
    with open(processed_file_log, "rb") as f:
        processed_files = pickle.load(f)
else:
    processed_files = set()

# 🚀 Start pipeline
print(f"🚀 Starting LDS AI with dynamic PDF loading... [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

docs, filenames = load_pdfs(pdf_folder)
new_files = [f for f in filenames if f not in processed_files]

if new_files:
    print("🔎 Found new PDFs to process:", new_files)
    new_docs = [doc for doc in docs if doc.metadata['source'].split('/')[-1] in new_files]
    new_chunks = split_docs(new_docs)
    processed_files.update(new_files)
else:
    new_chunks = []
    print("✅ No new PDFs to embed.")

# 🧠 Build or update vectorstore
vectorstore = build_or_load_vector_store(new_chunks)

# 💾 Save processed file log
with open(processed_file_log, "wb") as f:
    pickle.dump(processed_files, f)

# 🧠 Ready for chatbot use
print("🧠 LDS AI is ready.")    


#WHATSAPP APP
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
