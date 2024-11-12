from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime
import logging
import pickle
import gridfs
from pymongo import MongoClient
import torch
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "document_chat"

# Pydantic Models
class ChatRequest(BaseModel):
    question: str
    conversation_id: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: Optional[List[str]] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None

# MongoDB Connection
def get_mongodb_connection():
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        return db
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise

# Vector Store Operations
def save_vector_store_to_mongodb(vector_store, metadata=None):
    try:
        db = get_mongodb_connection()
        fs = gridfs.GridFS(db)
        
        serialized_vector_store = pickle.dumps(vector_store)
        
        file_metadata = {
            'type': 'faiss_index',
            'created_at': datetime.utcnow(),
            'custom_metadata': metadata or {}
        }
        
        file_id = fs.put(serialized_vector_store, **file_metadata)
        return str(file_id)
    
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        raise

def load_vector_store_from_mongodb():
    try:
        db = get_mongodb_connection()
        fs = gridfs.GridFS(db)
        
        latest_vector_store = fs.find_one(
            {'type': 'faiss_index'},
            sort=[('created_at', -1)]
        )
        
        if not latest_vector_store:
            return None
        
        vector_store = pickle.loads(latest_vector_store.read())
        return vector_store
    
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

# Document Processing Functions
def process_file(file_content: bytes, file_name: str) -> tuple[str, List[Document]]:
    """Process uploaded file and return extracted text and documents"""
    # Implement your existing file processing logic here
    # This should include your PDF, CSV, Excel, etc. processing functions
    # Return tuple of (extracted_text, documents)
    pass

def create_text_chunks(text: str, metadata: Dict) -> tuple[List[str], List[Dict]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata for _ in chunks]
    return chunks, metadata_chunks

def create_vector_store(text_chunks: List[str], metadata_chunks: List[Dict]):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings.client.to(device)
    
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadata_chunks
    )
    
    return vector_store

# Conversation Management
class ConversationManager:
    def __init__(self):
        self.conversations = {}
    
    def get_or_create_memory(self, conversation_id: str) -> ConversationBufferMemory:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
        return self.conversations[conversation_id]
    
    def clear_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            self.conversations[conversation_id].clear()

conversation_manager = ConversationManager()

# API Endpoints
@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
) -> ProcessingStatus:
    try:
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Process files in background
        background_tasks.add_task(process_documents_background, files, job_id)
        
        return ProcessingStatus(
            status="processing",
            message="Documents are being processed",
            job_id=job_id
        )
    
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/status/{job_id}")
async def get_processing_status(job_id: str) -> ProcessingStatus:
    # Implement job status checking logic
    # This could involve checking a status in MongoDB
    pass

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        vector_store = load_vector_store_from_mongodb()
        if not vector_store:
            raise HTTPException(status_code=404, detail="No processed documents found")
        
        memory = conversation_manager.get_or_create_memory(request.conversation_id)
        
        qa_chain = create_qa_chain(vector_store, memory)
        result = qa_chain({"question": request.question})
        
        return ChatResponse(
            answer=result['answer'],
            conversation_id=request.conversation_id,
            sources=[doc.metadata.get('source', '') for doc in result.get('source_documents', [])]
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    try:
        conversation_manager.clear_conversation(conversation_id)
        return {"status": "success", "message": "Conversation cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background Processing
async def process_documents_background(files: List[UploadFile], job_id: str):
    try:
        all_text = ""
        all_docs = []
        
        for file in files:
            content = await file.read()
            text, docs = process_file(content, file.filename)
            if text and docs:
                all_text += text + "\n\n"
                all_docs.extend(docs)
        
        text_chunks, metadata_chunks = [], []
        for doc in all_docs:
            chunks, meta_chunks = create_text_chunks(doc.page_content, doc.metadata)
            text_chunks.extend(chunks)
            metadata_chunks.extend(meta_chunks)
        
        vector_store = create_vector_store(text_chunks, metadata_chunks)
        save_vector_store_to_mongodb(vector_store, {"job_id": job_id})
        
        # Update job status in MongoDB
        update_job_status(job_id, "completed")
        
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        update_job_status(job_id, "failed", str(e))

def create_qa_chain(vector_store, memory):
    prompt_template = """[Your existing prompt template]"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def update_job_status(job_id: str, status: str, error: str = None):
    db = get_mongodb_connection()
    db.processing_jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": status,
                "updated_at": datetime.utcnow(),
                "error": error
            }
        },
        upsert=True
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)