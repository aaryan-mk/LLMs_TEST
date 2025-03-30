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
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
import re
import os
import time
import pytesseract
from pdf2image import convert_from_path
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
from contextlib import asynccontextmanager



INDEX_PATH = "faiss_index"
PDSA_INDEX_PATH = "faiss_index_pdsa"
RECOMMENDER_INDEX_PATH = "faiss_index_recommender"
TRANSCRIPT_INDEX_PATH = "faiss_index_transcript"

PDF_PATHS = [
    "week1_part1.pdf",
    "week1_part2.pdf",
    "week1_rev.pdf",
    "week2.pdf",
    "week2_rev.pdf",
    "week3.pdf",
    "week3_rev.pdf",
    "week4_part1.pdf",
    "week4_part2.pdf",
    "week4_rev.pdf"
]
PDSA_PDF_PATHS = [
    "pdsa1.pdf",
    "pdsa2.pdf",
    "pdsa3.pdf",
    "pdsa4.pdf"
]
RECOMMENDER_PDF_PATHS = [
    "chinky.pdf"
]

MLT_TRANSCRIPT_PATH = [
    "mlt1.pdf",
    "mlt2.pdf",
    "mlt3.pdf",
    "mlt4.pdf"
]

db = None
pdsa_db = None
recommender_db = None
transcript_db = None
retriever = None
pdsa_retriever = None
recommender_retriever = None
transcript_retriever = None

class QueryRequest(BaseModel):
    query: str
    history: list = []

class PDSAQueryRequest(BaseModel):
    query: str
    history: list = []

class RecommenderQueryRequest(BaseModel):
    query: str
    history: list = []

class TranscriptQueryRequest(BaseModel):
    query: str
    history: list = []

class QueryResponse(BaseModel):
    response: str

def extract_text_with_ocr(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=r"/opt/homebrew/bin")
    documents = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        metadata = {"source": f"{os.path.basename(pdf_path)}_page_{i+1}"}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def remove_think_content(text: str) -> str:
    """Removes any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def load_index():
    global db, retriever
    print("Checking if indexing already exists for FAISS")
    if os.path.exists(INDEX_PATH):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        print("FAISS index loaded successfully!")
        return 
    
    print("Index file not found proceeding to create embeddings...")
    documents = []
    for path in PDF_PATHS:
        if path.lower().endswith(".pdf"):
            loaded_docs = extract_text_with_ocr(path)
        else:
            loader = TextLoader(path)
            loaded_docs = loader.load()
        for d in loaded_docs:
            d.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents.extend(splitter.split_documents(loaded_docs))

    print("Computing embeddings for all documents (this may take a while)...")
    start_time = time.time()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    db.save_local(INDEX_PATH)
    end_time = time.time()
    print(f"Embeddings computed in {end_time - start_time:.2f} seconds")

def load_pdsa_index():
    global pdsa_db, pdsa_retriever
    if os.path.exists(PDSA_INDEX_PATH):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pdsa_db = FAISS.load_local(PDSA_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        pdsa_retriever = pdsa_db.as_retriever()
        return

    print("Index file not found proceeding to create embeddings for pdsa...")
    all_docs = []
    for path in PDSA_PDF_PATHS:
        if path.lower().endswith(".pdf"):
            loader = extract_text_with_ocr(path)
        for d in loader:
            d.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_docs.extend(splitter.split_documents(loader))
        
    print("Computing embeddings(pdsa) for all documents (this may take a while)...")
    start_time = time.time()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pdsa_db = FAISS.from_documents(all_docs, embeddings)
    pdsa_retriever = pdsa_db.as_retriever()
    pdsa_db.save_local(PDSA_INDEX_PATH)
    end_time = time.time()
    print(f"Embeddings computed in {end_time - start_time:.2f} seconds")

def load_recommender_index():
    global recommender_db, recommender_retriever
    if os.path.exists(RECOMMENDER_INDEX_PATH):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        recommender_db = FAISS.load_local(RECOMMENDER_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        recommender_retriever = recommender_db.as_retriever()
        return

    print("Index file not found proceeding to create embeddings for recommender...")
    all_docs = []
    for path in RECOMMENDER_PDF_PATHS:
        if path.lower().endswith(".pdf"):
            loader = extract_text_with_ocr(path)
        for d in loader:
            d.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_docs.extend(splitter.split_documents(loader))
        
    print("Computing embeddings(recommender) for all documents (this may take a while)...")
    start_time = time.time()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    recommender_db = FAISS.from_documents(all_docs, embeddings)
    recommender_retriever = recommender_db.as_retriever()
    recommender_db.save_local(RECOMMENDER_INDEX_PATH)
    end_time = time.time()
    print(f"Embeddings computed in {end_time - start_time:.2f} seconds")

def load_transcript_index():
    global transcript_db, transcript_retriever
    if os.path.exists(TRANSCRIPT_INDEX_PATH):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        transcript_db = FAISS.load_local(TRANSCRIPT_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        transcript_retriever = transcript_db.as_retriever()
        return

    print("Index file not found proceeding to create embeddings for transcript...")
    all_docs = []
    for path in MLT_TRANSCRIPT_PATH:
        if path.lower().endswith(".pdf"):
            loader = extract_text_with_ocr(path)
        for d in loader:
            d.metadata["source"] = os.path.basename(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_docs.extend(splitter.split_documents(loader))
        
    print("Computing embeddings(transcript) for all documents (this may take a while)...")
    start_time = time.time()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    transcript_db = FAISS.from_documents(all_docs, embeddings)
    transcript_retriever = transcript_db.as_retriever()
    transcript_db.save_local(TRANSCRIPT_INDEX_PATH)
    end_time = time.time()
    print(f"Embeddings computed in {end_time - start_time:.2f} seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads PDFs automatically when the server starts."""
    print("Starting up the app and loading document embeddings...")
    load_index()
    load_pdsa_index()
    load_recommender_index()
    load_transcript_index()
    yield

app = FastAPI(lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilteredConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
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
    print("Entered the academic helper endpoint : query")
    if db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded. Try restarting the server.")

    query = request.query
    history = request.history

    print("Debugging History received:", history)

    # Build a chat prompt using a system message plus a human message
    system_message = SystemMessagePromptTemplate.from_template("""
        Do not use your global knowledge base to answer the question. Use only the knowledge base that is given to you.
        You are a teaching assistant for an IIT Madras Machine Learning Techniques course.
        Your role is to help students understand concepts, formulas, and theories without giving any direct answers to their problems.
        If any non-academic question is asked, ask the student to focus on the course and do not help them with non-academic topics.
        Provide clear explanations of definitions, relevant theories, and guiding formulas.
        Do not provide any direct numerical or computational answers.
        If a student asks for a direct answer, respond with guiding questions or conceptual clarifications.
        Keep your responses concise and focused solely on linear algebra concepts.
        If a question is out of scope(out of given context in retriever) or unacademic, politely redirect the student back to the coursework.
        If a question is specifically asked out of context that is given to you do not use your global knowledge base to answer the question.
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

    # Use the OllamaLLM with your Llama3.1 model.
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    # Create the ConversationalRetrievalChain using our custom filtered memory.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    # Combine the user query and history (context) into a single string to pass as the input key
    # Here, we pass both `context` (history) and `query` (user's current question) together in one string
    combined_input = " ".join([item["content"] for item in history] + [query])

    # Invoke the chain with the combined context and question (no need for 'context' and 'question' as separate keys)
    result = qa_chain.invoke({"question": combined_input})

    # Filter out internal reasoning from the final answer before returning.
    filtered_answer = remove_think_content(result["answer"])

    # Append the user's query and bot's response to history
    history.append({"role": "User", "content": query})
    history.append({"role": "Bot", "content": filtered_answer})

    print("Academic helper responded successfully!")
    return QueryResponse(response=filtered_answer)

@app.post("/query-pdsa", response_model=QueryResponse)
def query_pdsa_assistant(request: PDSAQueryRequest):
    print("Entered the PDSA assistant endpoint")

    text_index_path = "faiss_index_pdsa"  # Path to the FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    try:
        print("Dynamically loading the embeddings for the text docx...")
        pdsa_db = FAISS.load_local(text_index_path, embeddings, allow_dangerous_deserialization=True)  # Load FAISS index
        pdsa_retriever = pdsa_db.as_retriever()  # Use the retriever for course search
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    print("Dynamic loading successfull!")

    if pdsa_db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDSA PDFs not loaded. Try restarting the server.")

    history = request.history
    query = request.query

    system_message = SystemMessagePromptTemplate.from_template("""
        Do not use your global knowledge base to answer the question. Use only the knowledge base that is given to you.
        You are a teaching assistant for an IIT Programming and Data Structures and Algorithms in Python(PDSA) course.
        Your role is to help students understand concepts, formulas, and theories without giving any direct answers to their problems.
        If any non-academic question is asked, ask the student to focus on the course and do not help them with non-academic topics.
        Provide clear explanations of definitions, relevant theories, and guiding formulas.
        Do not provide any direct numerical or computational answers.
        If a student asks for a direct answer, respond with guiding questions or conceptual clarifications.
        If a question is out of scope(out of given context in retriever) or unacademic, politely redirect the student back to the coursework.
        If a question is specifically asked out of context that is given to you do not use your global knowledge base to answer the question.
        IMPORTANT: Do not include any internal chain-of-thought, processing details, or meta comments in your final output.
        Remove any text within "<think>" markers before outputting your final answer.
        I only want you to give the answers and nothing more—no think statements.
        If a coding question is asked, do the same thing as above.             
        There can be some assignments where trivial questions are asked and the same can be queried to you for help , be liberal about sticking to course content and help the student.Nevertheless never give direct answers.
    """)

    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Question: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=pdsa_retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    combined_input = " ".join([item["content"] for item in history] + [query])

    result = qa_chain.invoke({"question": combined_input})

    filtered_answer = remove_think_content(result["answer"])

    history.append({"role": "User", "content": query})
    history.append({"role": "Bot", "content": filtered_answer})

    print("PDSA assistant responded successfully!")

    return QueryResponse(response=filtered_answer)

@app.post("/generate-notes/{week}", response_model=QueryResponse)
def generate_notes(week: str):
    print("Entered notes summary bot...")
    if db is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded.")

    matched = [f for f in PDF_PATHS if week.lower() in f.lower()]
    if not matched:
        raise HTTPException(status_code=404, detail=f"No PDFs found for '{week}'")

    filtered_retriever = db.as_retriever(
        search_kwargs={"metadata_filter": {"source": {"$in": matched}}}
    )

    system_message = SystemMessagePromptTemplate.from_template("""
        You are an expert summarizer for academic material.
        Your task is to generate a detailed and well-organized summary of the provided content.
        Include definitions, key concepts, and clear explanations where applicable.
        Format your answer using markdown with sections, headings, bullet points, and emphasis as needed.
    """)
    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Using the above context, please provide a detailed summary of the contents for: {question}
    """)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=filtered_retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    result = qa_chain.invoke({"question": f"Provide a detailed summary of the contents of {week}."})
    filtered_answer = remove_think_content(result["answer"])
    print("Summarizer bot responded successfully!")
    return QueryResponse(response=filtered_answer)


@app.post("/generate-notes-pdsa/{week}", response_model=QueryResponse)
def generate_notes_pdsa(week: str):
    print("Entered PDSA notes summary bot...")
    if pdsa_db is None:
        raise HTTPException(status_code=500, detail="PDSA PDFs not loaded.")
    
    matched = [f for f in PDSA_PDF_PATHS if week.lower() in f.lower()]
    if not matched:
        raise HTTPException(status_code=404, detail=f"No PDFs found for '{week}'")

    filtered_retriever = pdsa_db.as_retriever(
        search_kwargs={"metadata_filter": {"source": {"$in": matched}}}
    )

    system_message = SystemMessagePromptTemplate.from_template("""
        You are an assistant tasked with generating a detailed summary of Python and DSA course content.
        Include syntax, example snippets, complexity notes, and key algorithm insights.
    """)

    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Summarize the above for: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=filtered_retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    result = qa_chain.invoke({"question": f"Provide a detailed summary of the contents of {week}"})
    filtered_answer = remove_think_content(result["answer"])
    return QueryResponse(response=filtered_answer)


@app.post("/recommend-courses", response_model=QueryResponse)
def recommend_courses(request: RecommenderQueryRequest):
    print("Entered course recommender bot...")

    # Dynamically load FAISS index and retriever only when the endpoint is called
    text_index_path = "faiss_index_recommender"  # Path to the FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    try:
        print("Dynamically loading the embeddings for the text docx...")
        recommender_db = FAISS.load_local(text_index_path, embeddings, allow_dangerous_deserialization=True)  # Load FAISS index
        recommender_retriever = recommender_db.as_retriever()  # Use the retriever for course search
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    print("Dynamic loading successfull!")

    # Extract the query and conversation history from the request
    query = request.query
    history = request.history

    # Build a system message for course recommendation
    system_prompt = SystemMessagePromptTemplate.from_template("""
        You are an academic advisor bot for an IIT Madras program.

        Based on the student's stated interests and preferences, recommend **4–6 courses** from the available courses in `course_availability.txt`.

        The student has completed **all core courses except BSCS3004 (Deep Learning)**, so recommend **BSCS3004** as it is a **required course** that must be completed before the student can take any electives. If the student hasn't taken it yet, ensure that it is recommended first.
        Also familiarise yourself with all the course codes and their abbreviations.
        For each recommendation, provide the following:
        - Course Code
        - Course Name
        - Term Offered (from the available data in `course_availability.txt`)
        - Credit Hours
        - Prerequisites (if any). If not provided, infer a logical sequence of courses(ONLY THE courses that are available in `course_availability.txt`) that leads to a comprehensive understanding of related concepts.
        - A brief explanation of why this course is a good match based on the student's preferences and inferred learning path.

        IMPORTANT:
        - **Only recommend courses that are available in `course_availability.txt`**.
        - **Do not suggest courses outside of the provided data**.
        - **The maximum number of courses the student can take per term is 4**.
        - The **electives** should be chosen based on the **student’s preferences** (e.g., if the student likes **maths**, recommend relevant courses like **Deep Learning** or **Mathematical Thinking**).
        - **If no prerequisites are mentioned for a course**, infer a logical learning path (e.g., recommend **Linear Algebra** before **Machine Learning**).
        - If **BSCS3004 (Deep Learning)** has not been taken by the student yet, it must be included in the course recommendations first, followed by electives.
    """)

    # Human prompt with context (conversation history) and query
    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Student preferences: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])

    # Use the OllamaLLM with your Llama3.1 model for processing
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    # Create the ConversationalRetrievalChain using our custom filtered memory.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=recommender_retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    # Ensure all items in history have the 'content' key before accessing it
    combined_input = " ".join([item.get("content", "") for item in history] + [query])

    # Retrieve relevant courses from the retriever based on the query
    #retrieved_courses = recommender_retriever.get_relevant_documents(query)
    result = qa_chain.invoke({"question": combined_input})

    # Add retrieved courses as context for course recommendations
    # context = history + [{"role": "System", "content": " ".join([doc.page_content for doc in retrieved_courses])}]

    # # Create the input structure for LangChain, passing 'question' and 'context'
    # inputs = {
    #     "question": combined_input,  # Combined query and context (history + course data)
    #     "context": context           # Conversation history passed as 'context'
    # }

    # # Invoke the chain with the structured input
    # result = qa_chain.invoke(inputs)

    # Filter out internal reasoning from the final answer before returning
    filtered_answer = remove_think_content(result["answer"])

    # Append the user query and bot's response to the history
    history.append({"role": "User", "content": query})
    history.append({"role": "Bot", "content": filtered_answer})

    print("Course Recommender bot responded successfully!")
    return QueryResponse(response=filtered_answer)

@app.post("/transcript", response_model=QueryResponse)
def transcript(request: TranscriptQueryRequest):
    print("Entered the transcript endpoint")

    if transcript_db is None or transcript_retriever is None:
        load_transcript_index()

    text_index_path = "faiss_index_transcript"  # Path to the FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    try:
        print("Dynamically loading the embeddings for the text docx...")
        transcript_db = FAISS.load_local(text_index_path, embeddings, allow_dangerous_deserialization=True)  # Load FAISS index
        transcript_retriever = transcript_db.as_retriever()  # Use the retriever for course search
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    print("Dynamic loading successfull!")

    if transcript_db is None or transcript_retriever is None:
        raise HTTPException(status_code=500, detail="Transcript PDFs not loaded. Try restarting the server.")

    history = request.history
    query = request.query

    # Build a chat prompt using a system message plus a human message
    system_message = SystemMessagePromptTemplate.from_template("""
        You are a teaching assistant for an IIT Madras Machine Learning Techniques course.
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
        Refer to the pdfs provided to you and answer to that context only.
    """)

    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>
        Student preferences: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Use the OllamaLLM with your Llama3.1 model for processing
    llm = OllamaLLM(model="llama3.1:8b", temperature=0)

    # Create the ConversationalRetrievalChain using our custom filtered memory.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=transcript_retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    # Ensure all items in history have the 'content' key before accessing it
    combined_input = " ".join([item.get("content", "") for item in history] + [query])

    result = qa_chain.invoke({"question": combined_input})

    filtered_answer = remove_think_content(result["answer"])

    history.append({"role": "User", "content": query})
    history.append({"role": "Bot", "content": filtered_answer})

    print("Transcript bot responded successfully!")
    return QueryResponse(response=filtered_answer)

    # Use the OllamaLLM with your Llama3.1 model for processing