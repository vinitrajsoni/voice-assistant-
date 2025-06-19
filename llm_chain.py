from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from config import GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

LANG_CODE_NAME = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu"
}

def load_qa_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("embeddings/faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt_template = lambda query, lang: f"""
    Answer the following question in {lang} (max 80 words):
    {query}
    """

    def invoke(query, lang_code):
        lang = LANG_CODE_NAME.get(lang_code, "English")
        prompt = prompt_template(query, lang)
        return {"result": llm.invoke(prompt)}

    chain = type("CustomQA", (), {"invoke": staticmethod(invoke)})
    return chain