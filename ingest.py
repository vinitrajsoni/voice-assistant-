from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

pdf_dir = "data/"
all_docs = []

# Load all PDFs
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

# Create embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Save FAISS vector store
db = FAISS.from_documents(chunks, embedding)
db.save_local("embeddings/faiss_index")

print("✅ Vector store created and saved.")
