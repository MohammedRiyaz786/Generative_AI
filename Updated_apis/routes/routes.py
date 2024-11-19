from fastapi import  UploadFile, HTTPException, BackgroundTasks,Request,APIRouter,Form,UploadFile,File
from pydantic import BaseModel
from typing import Optional, List,Dict
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from threading import Thread
import logging
import asyncio
from fastapi.responses import StreamingResponse
import json
import jsonify
rag = APIRouter()

# from ..db import async_db,fs
from .routes_func import process_document_background,get_faiss_index,async_db,fs
from StatusTracker import status_tracker
from utils.filefunctions import create_qa_chain





class ChatRequest(BaseModel):
    document_key: str
    prompt: str
    chat_history: Optional[List[Dict[str, str]]] = []


class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None



@rag.post("/upload")
async def upload_document(
    req: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_key: str = Form(...),
):
    try:
        user_ip = req.client.host
        logging.info(f"Upload request received from IP {user_ip}")
        
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )

        allowed_extensions = {
            'pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx',
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'
        }
        
        # Check if document key exists
        existing_doc = await async_db.documents.find_one(
            {"document_key": document_key}
        )
        
        if existing_doc:
            raise HTTPException(
                status_code=400,
                detail=f"Document key {document_key} already exists."
            )

        file_details = []
        file_contents = []
        
        # Process all files
        for file in files:
            # Validate file type
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} type not allowed. Allowed types: {', '.join(allowed_extensions)}"
                )
            
            # Read file content
            file_content = await file.read()
            
            # Store in GridFS
            file_id = fs.put(
                file_content,
                filename=f"document_{document_key}_{file.filename}"
            )
            
            file_details.append({
                "file_id": file_id,
                "filename": file.filename
            })
            file_contents.append((file_content, file.filename))

        # Create single document record
        await async_db.documents.insert_one({
            "document_key": document_key,
            "files": file_details,
            "index_status": "processing"
        })

        # processing_status[document_key] = {
        #     'status': f"Received {len(files)} documents",
        #     'Completed': 1,
        #     'total': 8
        # }
        
        # Start processing thread with all files
        thread = Thread(
            target=process_document_background,
            args=(file_contents, document_key)
        )
        thread.daemon = True
        thread.start()
        
        return {
            "message": "Documents uploaded and processing started",
            "document_key": document_key,
            "files": [detail["filename"] for detail in file_details]
        }
        
    except Exception as e:
        logging.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )
    
@rag.get('/status/{document_key}')
async def check_status(document_key: str):
    status_history = status_tracker.get_status_history(document_key)
    if not status_history and not status_tracker.is_processed(document_key):
        raise HTTPException(status_code=404, detail="Document key not found")

    async def generate():
        if status_tracker.is_processed(document_key):
            # If already processed, just return completed status
            yield json.dumps({
                "document_key": document_key,
                "status": "Processing completed",
                "completed": 8,
                "total": 8
            }) + "\n"
            return

        # Stream all existing status updates
        for status in status_history:
            response_data = {
                "document_key": document_key,
                "status": status.message,
                "completed": status.completed,
                "total": status.total
            }
            if status.details:
                response_data.update(status.details)
            
            yield json.dumps(response_data) + "\n"
            await asyncio.sleep(0.1)  # Small delay between updates

        # If this was the last status check (completed), mark as processed
        if status_history and status_history[-1].completed == 8:
            status_tracker.mark_as_processed(document_key)

    return StreamingResponse(generate(), media_type="text/event-stream")



@rag.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    req:Request,
    document_key: str = Form(...),
    prompt: str = Form(...),
    chat_history: Optional[str] = Form(None),
):
    try:
        user_ip = req.client.host
        logging.info(f"Chat request recieved from IP :{user_ip}")

        # Check if document exists and get status
         # Convert chat_history from string to List[Dict[str, str]]
        chat_history_list = []
        if chat_history:
            try:
                chat_history_list = json.loads(chat_history)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid chat history format")

        # Create a ChatRequest object
        chat_request = ChatRequest(
            document_key=document_key,
            prompt=prompt,
            chat_history=chat_history_list
        )
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
        if chat_request.chat_history:
            for message in chat_request.chat_history:
                if message["type"] == "human":
                    memory.chat_memory.add_message(HumanMessage(content=message["content"]))
                elif message["type"] == "ai":
                    memory.chat_memory.add_message(AIMessage(content=message["content"]))

        
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
    

