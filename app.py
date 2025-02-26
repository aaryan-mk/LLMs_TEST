from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
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

# Global vector store and retriever
db = None
retriever = None

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

class FilteredConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        # Filter out any <think> tags from both inputs and outputs before storing them
        filtered_inputs = {
            key: remove_think_content(value) if isinstance(value, str) else value
            for key, value in inputs.items()
        }
        filtered_outputs = {
            key: remove_think_content(value) if isinstance(value, str) else value
            for key, value in outputs.items()
        }
        super().save_context(filtered_inputs, filtered_outputs)

@app.post("/query", response_model=QueryResponse)
def query_ai_assistant(request: QueryRequest):
    """Processes user queries with DeepSeek using a system prompt and filtered output."""
    if db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded. Try restarting the server.")

    query = request.query

    # Build a chat prompt using a system message plus a human message
    system_message = SystemMessagePromptTemplate.from_template("""
You are a teaching assistant for an IIT Madras linear algebra course.
Your role is to help students understand concepts, formulas, and theories without giving any direct answers to their problems.
If any non-academic question is asked, ask the student to focus on the course and do not help them with non-academic topics.
Provide clear explanations of definitions, relevant theories, and guiding formulas.
Do not provide any direct numerical or computational answers.
If a student asks for a direct answer, respond with guiding questions or conceptual clarifications.
Keep your responses concise and focused solely on linear algebra concepts.
If a question is out of scope or unacademic, politely redirect the student back to the coursework.
IMPORTANT: Do not include any internal chain-of-thought, processing details, or meta comments in your final output.
Remove any text within "<think>" markers before outputting your final answer.
I only want you to give the answers and nothing more—no think statements.
""")

    human_message = HumanMessagePromptTemplate.from_template("""
<context>
{context}
</context>

Question: {question}
""")

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Use the OllamaLLM with your DeepSeek model.
    llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0)

    # Create the ConversationalRetrievalChain using our custom filtered memory.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    # Invoke the chain with the user's query.
    result = qa_chain.invoke({"question": query})

    # Filter out internal reasoning from the final answer before returning.
    filtered_answer = remove_think_content(result["answer"])

    return QueryResponse(response=filtered_answer)