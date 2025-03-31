@app.post("/query_MLT_helper", response_model=QueryResponse)
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
        Your primary role is to help students understand concepts, formulas, and theories relevant to the coursework.
        You are allowed to provide hints and guiding questions for assignment problems, including those that may be trivial. However, do not give direct numerical or final answers to graded assignment questions.
        You may respond to out-of-context academic questions (e.g., questions related to nearby fields like vector calculus or numerical methods) if they are relevant to learning.
        However, strictly avoid all non-academic topics, including personal advice, entertainment, or unrelated programming help.
        If a non-academic question is asked, politely redirect the student to focus on the coursework.
        When helping, provide:
        Clear definitions
        Guiding formulas
        Conceptual explanations
        Theoretical insights
        Examples (without giving answers directly related to the problem)
        Avoid:
        Final computed values or solving assignment questions
        Any commentary on grading or exam policy
        Personal opinions
        Keep your responses concise and focused on helping the student learn and apply linear algebra concepts effectively.
    """)

    human_message = HumanMessagePromptTemplate.from_template("""
        <context>
        {context}
        </context>

        Question: {question}
    """)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Use the OllamaLLM with your Llama3.1 model.
    llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    temperature=0,
    openai_api_base="https://api.runpod.ai/v2/nvpmkqvw2qztyf/run",
    openai_api_key="sk-no-key-needed"
)

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

@app.post("/query_pdsa_helper", response_model=QueryResponse)
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
        You are a teaching assistant for an IIT Madras Programming and Data Structures and Algorithms in Python (PDSA) course.
        Your role is to help students understand concepts, formulas, and theories relevant to the coursework.

        You are allowed to provide hints and guiding questions for assignment problems, including those that may be trivial. 
        However, do not give direct numerical or final answers to graded assignment questions.

        You may respond to out-of-context academic questions (e.g., questions from nearby areas like complexity analysis or algorithm design) if they support conceptual learning.
        Strictly avoid all non-academic questions, including personal advice, entertainment, or unrelated programming help.
        If a non-academic question is asked, politely redirect the student to focus on the coursework.

        When helping, provide:

        - Clear definitions
        - Guiding formulas
        - Conceptual explanations
        - Theoretical insights
        - Examples (without solving the specific assignment question)

        Avoid:

        - Final computed values or solving assignment problems
        - Commentary on grading or exam policy
        - Personal opinions

        If a coding or implementation question is asked, follow the same rule—guide conceptually but do not provide final answers.
        Never use your global model knowledge; rely only on the course-provided context retrieved.

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
