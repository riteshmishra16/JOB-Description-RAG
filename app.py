from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Job Description RAG Assistant")

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chat_template = ChatPromptTemplate([
    (
        "system",
        "You are a helpful job information expert. "
        
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }| chat_template| model | StrOutputParser()
)

print("RAG pipeline loaded successfully!")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Job Description RAG API is working"}


@app.post("/ask")
def ask_question(data: Question):
    try:
        response = rag_chain.invoke(data.question)
        return {"answer": response}

    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
