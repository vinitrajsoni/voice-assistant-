import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ MUST BE FIRST Streamlit COMMAND
st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

# --- API Key Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyAOnicR6JOOniCWBu-RwSRX1dz5_rVEW58"  # Replace with your key

# --- Load FAISS Vector Store
@st.cache_resource(show_spinner="Loading FAISS index...")
def load_qa_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("embeddings/faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

qa_chain = load_qa_chain()

# --- Streamlit UI
st.title("🧠 Gemini RAG Chatbot")
st.markdown("Ask anything based on your custom documents!")

# --- Chat Input
query = st.text_input("💬 Enter your question:")

# --- Answer
if query:
    with st.spinner("Generating response..."):
        result = qa_chain.invoke(query)
        st.success("📜 Answer:")
        st.write(result)
