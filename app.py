from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import re
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
    pdf_paths = ["Linear Algebra Review-0.pdf", "Linear Algebra Review-1.pdf"]  # Update paths as needed
    all_documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_documents.extend(text_splitter.split_documents(docs))

    # Use OllamaEmbeddings with your DeepSeek model name
    db = FAISS.from_documents(
        all_documents,
        OllamaEmbeddings(model="deepseek-r1:1.5b")
    )
    retriever = db.as_retriever()
    print("PDFs successfully loaded into FAISS!")

@app.on_event("startup")
async def startup_event():
    """Loads PDFs automatically when the server starts."""
    load_pdfs()

def remove_think_content(text: str) -> str:
    """Removes any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

@app.post("/query", response_model=QueryResponse)
def query_ai_assistant(request: QueryRequest):
    """Processes user queries with DeepSeek and retains chat history."""
    if db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded. Try restarting the server.")

    query = request.query

    # Define your prompt template with explicit instructions to remove internal reasoning.
    prompt_template = ChatPromptTemplate.from_template("""
        You are a teaching assistant for an IIT Madras linear algebra course. 
        Your role is to help students understand concepts, formulas, and theories 
        without giving any direct answers to their problems. In your responses:
        - Provide clear explanations of definitions, relevant theories, and guiding formulas.
        - Do not provide any direct numerical or computational answers.
        - If a student asks for a direct answer, respond with guiding questions or conceptual clarifications.
        - Keep your responses concise and focused solely on linear algebra concepts.
        - If a question is out of scope or unacademic, politely redirect the student back to the coursework.
        - IMPORTANT: Do not include any internal chain-of-thought, processing details, or meta comments in your final output.
          Remove any text within "<think>" markers before outputting your final answer.

        <context>
        {context}
        </context>

        Question: {question}
    """)

    # Use the OllamaLLM as-is (without structured output)
    llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0)

    # Create the ConversationalRetrievalChain with the LLM
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    # Invoke the chain with your query
    result = qa_chain.invoke({"question": query})

    # Filter out any internal reasoning (<think> ... </think>) from the answer
    filtered_answer = remove_think_content(result["answer"])

    # Return only the final student-facing answer
    return QueryResponse(response=filtered_answer)
