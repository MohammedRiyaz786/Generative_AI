from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import asyncio
import logging
from threading import Thread
from utils.document_processor import process_pdf_document, create_document_hash
from utils.index_manager import (
    get_vector_store,
    create_qa_chain,
    store_index_in_mongodb,
    get_index_from_mongodb
)
from utils.cache_manager import faiss_cache

router = APIRouter()

class DocumentRequest(BaseModel):
    document: str
    key: str
    prompt: str

@router.post("/process_document")
async def process_document_api(
    request: Request,
    file: UploadFile = File(...),
    key: str = Form(...),
    prompt: str = Form(...)
    
):
    try:
        
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        
        file_content = await file.read()
        
        doc_hash = create_document_hash(file_content, key)
        
        # Check in-memory cache first
        if doc_hash in faiss_cache:
            print("Using in-memory cache")
            faiss_index, texts = faiss_cache[doc_hash]
        else:
            print("Checking MongoDB")
            faiss_index, texts = await get_index_from_mongodb(doc_hash, request.app.db)
            
            if not faiss_index:
                print("Creating new index")
                
                combined_text, documents = await process_pdf_document(file_content)
                if not combined_text:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract text from PDF"
                    )
                
                # Create vector store
                faiss_index, texts = await get_vector_store(documents)
                
                # Store in cache
                faiss_cache[doc_hash] = (faiss_index, texts)
                
                # Store in MongoDB asynchronously
                Thread(
                    target=lambda: asyncio.run(
                        store_index_in_mongodb(doc_hash, faiss_index, texts, request.app.db)
                    )
                ).start()
            else:
                faiss_cache[doc_hash] = (faiss_index, texts)
        
        # Create QA chain and get response
        qa_chain = await create_qa_chain(faiss_index, texts)
        response = qa_chain.run(prompt)
        
        return {
            "response": response,
            "source": "cache" if doc_hash in faiss_cache else "mongodb",
            "index_created": faiss_index is None,
            "filename": file.filename
        }
    
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Error processing document"
            }
        )