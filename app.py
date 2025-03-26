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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
import re
import os
import time

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

# Set the path for saving the FAISS index (this index contains precomputed embeddings)
INDEX_PATH = "faiss_index"

# PDF_PATHS contains your documents
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
    "week4_rev.pdf",
    "course_availability.txt"  # Add your course availability file
]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("DEBUG: This is the top of app.py file.")

class QueryRequest(BaseModel):
    query: str
    history: list = []  # Add a history field to store the conversation history

class VideoQueryRequest(BaseModel):
    query: str
    history: list = []  # Add a history field to store the conversation history
    video: str

class QueryResponse(BaseModel):
    response: str

# Load PDFs and initialize FAISS
def load_pdfs():
    global db, retriever
    print("Checking if indexing already exists for FAISS")
    if os.path.exists(INDEX_PATH):
        print("FAISS index file found!")
        print("Loading FAISS index from disk...")
        embeddings = OllamaEmbeddings(model="llama3.1:8b")
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        print("FAISS index loaded successfully!")
        return

    print("Index file not found proceeding to create embeddings...")
    all_documents = []
    for path in PDF_PATHS:
        print(f"Processing {path}...")
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_documents.extend(text_splitter.split_documents(docs))

    print("Computing embeddings for all documents (this may take a while)...")
    start_time = time.time()
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    db = FAISS.from_documents(all_documents, embeddings)
    retriever = db.as_retriever()
    db.save_local(INDEX_PATH)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"FAISS index created and saved successfully @ {INDEX_PATH} in {elapsed_time:.2f} seconds!")

@app.on_event("startup")
async def startup_event():
    """Loads PDFs automatically when the server starts."""
    print("Starting up the app and loading document embeddings...")
    load_pdfs()

def remove_think_content(text: str) -> str:
    """Removes any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

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
    """Processes user queries with DeepSeek using a system prompt and filtered output."""
    print("Entered the academic helper endpoint : query")
    if db is None or retriever is None:
        raise HTTPException(status_code=500, detail="PDFs not loaded. Try restarting the server.")

    query = request.query
    history = request.history

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

@app.post("/recommend-courses", response_model=QueryResponse)
def recommend_courses(request: QueryRequest):
    print("Entered course recommender bot...")

    # Dynamically load FAISS index and retriever only when the endpoint is called
    text_index_path = "faiss_index_textfile_backup"  # Path to the FAISS index
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    
    try:
        print("Dynamically loading the embeddings for the text docx...")
        db = FAISS.load_local(text_index_path, embeddings, allow_dangerous_deserialization=True)  # Load FAISS index
        retriever = db.as_retriever()  # Use the retriever for course search
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {e}")

    print("Dynamic loading successfull!")

    # Extract the query and conversation history from the request
    query = request.query
    history = request.history

    # Build a system message for course recommendation
    system_prompt = SystemMessagePromptTemplate.from_template("""
        You are an academic advisor bot for an IIT Madras program.
        You have access to the official course availability data from `course_availability.txt` (which has been embedded for easy search).

        Based on the student's stated interests and preferences, recommend **4–6 courses** from the available courses in `course_availability.txt`.

        You should use your understanding of the general content of the courses (e.g., **Deep Learning** typically involves **neural networks**, **mathematical foundations**, and **algorithm design**, while **Deep Learning Practice** focuses on **practical applications of Deep Learning**).

        For each recommendation, provide the following:
        - Course Code
        - Course Name
        - Term Offered (from the available data in `course_availability.txt`)
        - Credit Hours
        - Prerequisites (if any). If not provided, infer a logical sequence of courses that leads to a comprehensive understanding of related concepts.
        - A brief explanation of why this course is a good match based on the student's preferences and inferred learning path.

        IMPORTANT:
        - **Only recommend courses that are available in `course_availability.txt`**.
        - **Do not suggest courses outside of the provided data**.
        - **The maximum number of courses the student can take per term is 4**.
        - If a student has already completed core courses (like **BSCS3001, BSCS3002, BSCS3003, BSCS3004, BSGN3001**), electives can be recommended along with the remaining core courses.
        - The **core courses** (e.g., **BSCS3001 to BSCS3004**) must be prioritized first before electives are suggested.
        - The **electives** should be chosen based on the **student’s preferences** (e.g., if the student likes **maths**, recommend relevant courses like **Deep Learning** or **Mathematical Thinking**).
        - **If no prerequisites are mentioned for a course**, infer a logical learning path (e.g., recommend **Linear Algebra** before **Machine Learning**).
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
        retriever=retriever,
        memory=FilteredConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )

    # Ensure all items in history have the 'content' key before accessing it
    combined_input = " ".join([item.get("content", "") for item in history] + [query])

    # Retrieve relevant courses from the retriever based on the query
    #retrieved_courses = retriever.get_relevant_documents(query)
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


@app.post("/query_video", response_model=QueryResponse)
def query_ai_assistant(request: VideoQueryRequest):
    """Processes user queries with video transcript context with DeepSeek using a system prompt and filtered output."""
    print("Entered the Query Video endpoint : query_video")

    query = request.query
    history = request.history
    video = request.video

    try:
        transcript = open("transcripts/" + video + ".txt", "r").read()
    except e:
        print("Error reading the video transcript file.")
        print(e)
        return {"response": "Error reading the video transcript file."}
    
    print("transcript", transcript)
    # Build a chat prompt using a system message plus a human message
    system_message = SystemMessagePromptTemplate.from_template(f"""
        You are a teaching assistant for an IIT Madras linear algebra course.
        Your role is to help students understand concepts, formulas, and theories without giving any direct answers to their problems.
        If any non-academic question is asked, ask the student to focus on the course and do not help them with non-academic topics.
        Provide clear explanations of definitions, relevant theories, and guiding formulas.
        Do not provide any direct numerical or computational answers.
        If a student asks for a direct answer, respond with guiding questions or conceptual clarifications.
        Keep your responses concise and focused solely on a specific video whose transcript is given to you below.
        If a question is out of scope or unacademic, politely redirect the student back to the content of the video/transcript.
        IMPORTANT: Do not include any internal chain-of-thought, processing details, or meta comments in your final output.
        Remove any text within "<think>" markers before outputting your final answer.
        I only want you to give the answers and nothing more—no think statements.
        <transcript>
        {transcript.replace("{", "").replace("}", "")}
        </transcript>
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

    print("Query Transcript Bot responded successfully!")
    return QueryResponse(response=filtered_answer)