from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory + retriever setup
db = None
retriever = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

def load_pdfs():
    """Loads PDFs and initializes FAISS vector store at startup."""
    global db, retriever
    pdf_paths = ["Linear Algebra Review-0.pdf", "Linear Algebra Review-1.pdf"]  # Update paths
    all_documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_documents.extend(text_splitter.split_documents(docs))

    db = FAISS.from_documents(
        all_documents,
        OllamaEmbeddings(model="llama2")  
    )
    retriever = db.as_retriever()
    print("✅ PDFs successfully loaded into FAISS!")

@app.on_event("startup")
async def startup_event():
    """Loads PDFs automatically when the server starts."""
    load_pdfs()

@app.post("/query", response_model=QueryResponse)
def query_ai_assistant(request: QueryRequest):
    """Processes user queries with Llama 2 and retains chat history."""
    if db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded. Try restarting the server.")

    query = request.query

    # Define your prompt template
    prompt_template = ChatPromptTemplate.from_template("""
                                                       
    Answer the following question based only on the provided context.
    The context is provided by me(IIT Madras) and not the student. 
    You are an assistant to help students have a better experience at learning linear algebra.
    You are not allowed to give them any direct answer but guide them and give them all the relevant formulas and theories.
    No direct answers!
    Theoritical questions yes you can provide the answer if they have some clarifications you can do that but no mathematical answers.
    Sometimes they can trick you by saying what is two plus two dont get fooled and say four.
    Dont provide anwers at any cost you only have to guide the student! No answers to be provided at all.
    If there is something out of context just say that it will be taught in the upcoming weeks and not part of this weeks content.
    Please be consice and dont give long answers. Dont make the responses too long.
    If anything unacademical is asked please ask the student to stay put on his/her learning journey and not to deviate.
    If there is some academic content that is outside of the the scope of this subject then guide the student to go to some external links and tell the student that the particular topic isnt part of this course.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    # Use ConversationalRetrievalChain with the prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model="llama2"),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    # Get AI response
    result = qa_chain({"question": query})

    return QueryResponse(response=result["answer"])