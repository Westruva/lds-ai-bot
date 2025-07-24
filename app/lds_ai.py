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

    # Try to load existing index
    if Path(index_dir).exists():
        print("📂 Loading existing FAISS index...")
        try:
            vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Could not load FAISS index: {e}")

    # If new chunks exist, embed and add them
    if new_chunks:
        print("✨ Embedding new chunks in batches...")
        for i, batch in enumerate(embed_in_batches(new_chunks)):
            if not batch:
                continue  # Skip empty batch

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            time.sleep(1)

        # Save if index was created
        if vectorstore:
            vectorstore.save_local(index_dir)
            print("✅ FAISS index saved.")
    else:
        print("✅ No new PDFs to embed.")

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
