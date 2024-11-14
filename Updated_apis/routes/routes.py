from fastapi import  UploadFile, HTTPException, BackgroundTasks,Request,APIRouter
from pydantic import BaseModel
from typing import Optional, List
from langchain.memory import ConversationBufferMemory
from threading import Thread
import logging
import jsonify
rag = APIRouter()

# from ..db import async_db,fs
from .routes_func import process_document_background,get_faiss_index,async_db,fs,processing_status
from utils.filefunctions import create_qa_chain





class ChatRequest(BaseModel):
    document_key: str
    prompt: str

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

@rag.post("/upload")
async def upload_document(
    req : Request,
    file: UploadFile,
    document_key: str,
    background_tasks: BackgroundTasks
):
    try:
        user_ip = req.client.host
        logging.info(f"Upload request recieved from IP {user_ip}")
        # Validate file type
        allowed_extensions = {
            'pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx',
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'
        }
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Check if document already exists
        existing_doc = await async_db.documents.find_one(
            {"document_key": document_key}
        )
        
        if existing_doc:
            raise HTTPException(
                status_code=400,
                detail="Document key already exists."
            )
            
        # Read file content
        file_content = await file.read()
        
        # Store file in GridFS
        file_id = fs.put(
            file_content,
            filename=f"document_{document_key}"
        )
        processing_status[document_key]={'status':"Document recieved !","Completed":1,"total":10}
        # Create document record
        await async_db.documents.insert_one({
            "document_key": document_key,
            "file_id": file_id,
            "filename": file.filename,
            "index_status": "processing"
        })
        
        # Start background processing with filename information
        thread = Thread(
            target=process_document_background,
            args=(file_content, file.filename, document_key)
        )
        thread.daemon = True  # Set daemon to True
        thread.start()
        
        return {
            "message": "Document uploaded and processing started, you can check status with document key",
            "document_key": document_key,
            "filename": file.filename
        }
        
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@rag.get('/status/{document_key}')
def check_status(document_key:str):
    status = processing_status.get(document_key, "not found")
    if status['status']==True:
        return {
            "document_key": document_key, **status
        }
    elif status['status']==False:
        return {
            "document_key": document_key,
            "status": "Error occured, contact admin"
            
             }  
    else:
        return {
            "document_key": document_key,
            **status
        }




@rag.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req:Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks
):
    try:
        user_ip = req.client.host
        logging.info(f"Chat request recieved from IP :{user_ip}")

        # Check if document exists and get status
        doc_status = await async_db.documents.find_one(
            {"document_key": chat_request.document_key}
        )
        
        if not doc_status:
            raise HTTPException(
                status_code=404,
                detail="Document not found. Please upload the document first."
            )
            
        if doc_status.get("index_status") != "completed":
            raise HTTPException(
                status_code=400,
                detail="Document is still being processed. Please try again later."
            )
            
        # Get FAISS index
        vector_store = await get_faiss_index(chat_request.document_key)
        if not vector_store:
            raise HTTPException(
                status_code=500,
                detail="Error retrieving document index."
            )
            
        # Create memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create QA chain using existing logic
        qa_chain = create_qa_chain(vector_store, memory)
        
        # Get response
        result = qa_chain({"question": chat_request.prompt})
        response = result.get('answer', '').strip()
        
        if not response:
            response = "I don't have enough information to answer this question."
            
        return ChatResponse(response=response)
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return ChatResponse(
            response="",
            error=f"An error occurred: {str(e)}"
        )