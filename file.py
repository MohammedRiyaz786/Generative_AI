from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine

class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score between 0 and 1")
    reasoning: str = Field(description="Explanation of why this score was given")

class ChatRequest(BaseModel):
    document_key: str
    prompt: str
    chat_history: Optional[List[Dict[str, str]]] = []
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.1

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None
    confidence_score: Optional[float] = None
    relevant_chunks: Optional[List[str]] = None

@rag.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    req: Request,
    document_key: str = Form(...),
    prompt: str = Form(...),
    chat_history: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(2000),
    temperature: Optional[float] = Form(0.1)
):
    try:
        user_ip = req.client.host
        logging.info(f"Chat request received from IP: {user_ip}")

        # Validate and parse chat history
        chat_history_list = []
        if chat_history:
            try:
                chat_history_list = json.loads(chat_history)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid chat history format")

        # Create request object
        chat_request = ChatRequest(
            document_key=document_key,
            prompt=prompt,
            chat_history=chat_history_list,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Validate document status
        doc_status = await async_db.documents.find_one({"document_key": chat_request.document_key})
        if not doc_status:
            raise HTTPException(status_code=404, detail="Document not found. Please upload the document first.")
        if doc_status.get("index_status") != "completed":
            raise HTTPException(status_code=400, detail="Document is still being processed. Please try again later.")

        # Get FAISS index
        vector_store = await get_faiss_index(chat_request.document_key)
        if not vector_store:
            raise HTTPException(status_code=500, detail="Error retrieving document index.")

        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        # Populate memory with chat history
        if chat_request.chat_history:
            for message in chat_request.chat_history:
                if message["type"] == "human":
                    memory.chat_memory.add_message(HumanMessage(content=message["content"]))
                elif message["type"] == "ai":
                    memory.chat_memory.add_message(AIMessage(content=message["content"]))

        # Create enhanced QA chain
        qa_chain = create_enhanced_qa_chain(
            vector_store, 
            memory,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens
        )

        # Get response with confidence scoring
        with get_openai_callback() as cb:
            result = qa_chain({"question": chat_request.prompt})
            
        response = result.get('answer', '').strip()
        source_docs = result.get('source_documents', [])

        if not response:
            return ChatResponse(
                response="I don't have enough information to answer this question.",
                confidence_score=0.0
            )

        # Calculate confidence score
        confidence_score = calculate_confidence_score(
            query=chat_request.prompt,
            response=response,
            source_docs=source_docs,
            llm=qa_chain.llm
        )

        # Get relevant chunks for transparency
        relevant_chunks = [doc.page_content for doc in source_docs[:3]]

        # Log query metrics
        await log_query_metrics(
            document_key=chat_request.document_key,
            prompt=chat_request.prompt,
            tokens_used=cb.total_tokens,
            confidence_score=confidence_score,
            user_ip=user_ip
        )

        return ChatResponse(
            response=response,
            confidence_score=confidence_score,
            relevant_chunks=relevant_chunks
        )

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return ChatResponse(
            response="",
            error=f"An error occurred: {str(e)}"
        )

def create_enhanced_qa_chain(vector_store: FAISS, memory, temperature: float = 0.1, max_tokens: int = 2000):
    """Create an enhanced QA chain with better retrieval and response generation"""
    
    # Create base retriever
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # Add contextual compression
    llm = Ollama(
        model="llama3.1",
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

    # Enhanced prompt template
    prompt_template = """You are a knowledgeable AI assistant. Answer questions based on the provided context.

    Context: {context}
    Chat History: {chat_history}
    Current Question: {question}

    Instructions:
    1. Answer directly and concisely using only the provided context
    2. If the context doesn't contain enough information, say so
    3. Maintain consistency with previous chat history
    4. Use specific details from the context to support your answer
    5. Avoid speculation beyond the provided information

    Question: {question}
    
    Answer: Let me help you with that."""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    # Create the chain with the enhanced components
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

async def log_query_metrics(
    document_key: str,
    prompt: str,
    tokens_used: int,
    confidence_score: float,
    user_ip: str
):
    """Log query metrics to database"""
    metrics = {
        "document_key": document_key,
        "prompt": prompt,
        "tokens_used": tokens_used,
        "confidence_score": confidence_score,
        "user_ip": user_ip,
        "timestamp": datetime.utcnow()
    }
    await async_db.query_metrics.insert_one(metrics)

def calculate_confidence_score(
    query: str,
    response: str,
    source_docs: List,
    llm: Any
) -> float:
    """Calculate a confidence score based on multiple factors"""
    
    if not source_docs:
        return 0.0

    # 1. Semantic similarity between query and response
    query_embedding = llm.embed_query(query)
    response_embedding = llm.embed_query(response)
    semantic_similarity = 1 - cosine(query_embedding, response_embedding)

    # 2. Source document relevance
    doc_scores = []
    for doc in source_docs:
        doc_embedding = llm.embed_query(doc.page_content)
        doc_scores.append(1 - cosine(query_embedding, doc_embedding))
    avg_doc_relevance = np.mean(doc_scores)

    # 3. Response consistency with sources
    consistency_scores = []
    for doc in source_docs:
        doc_embedding = llm.embed_query(doc.page_content)
        consistency_scores.append(1 - cosine(response_embedding, doc_embedding))
    response_consistency = np.mean(consistency_scores)

    # Combine scores with weights
    final_score = (
        0.3 * semantic_similarity +
        0.3 * avg_doc_relevance +
        0.4 * response_consistency
    )

    return round(float(final_score), 3)