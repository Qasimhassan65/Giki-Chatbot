# app.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle

# -----------------------------
# Config
# -----------------------------
FAISS_INDEX_FILE = "faiss_index.pkl"
API_KEY = os.environ.get("GIKI_OPENROUTER_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))

# -----------------------------
# Load precomputed FAISS
# -----------------------------
with open(FAISS_INDEX_FILE, "rb") as f:
    faiss_index = pickle.load(f)

# Prompt template
prompt_template = """You are a helpful assistant for GIKI.
Answer questions based on official documents.

Context:
{context}

Question: {question}

Answer:"""
custom_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# LLM and QA chain
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={"HTTP-Referer": "https://huggingface.co", "X-Title": "GIKI-RAG-bot"},
    temperature=0.1
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# -----------------------------
# FastAPI app
# -----------------------------
api = FastAPI(title="GIKI-RAG Chatbot API")

class Question(BaseModel):
    question: str

@api.post("/chat")
def chat_endpoint(q: Question):
    response = qa_chain({"query": q.question})
    answer = response["result"]
    sources = set(f"📄 {doc.metadata['source']}" for doc in response["source_documents"])
    source_text = "\n\nSources:\n" + "\n".join(sources) if sources else ""
    return {"answer": f"{answer}\n\n{source_text}"}

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=PORT)
