import logging
import io
import pickle
import asyncio
# from .routes import processing_status
processing_status = {}
from .db import sync_db,async_db,fs
from utils.filefunctions import extract_file_content,get_vector_store,get_text_chunks



class CustomBytesIO(io.BytesIO):
    """Custom BytesIO class that maintains filename information"""
    def __init__(self, content, filename, content_type=None):
        if isinstance(content, bytes):
            super().__init__(content)
        else:
            super().__init__(content.getvalue())  # Get bytes if content is BytesIO
        self.name = filename
        self._filename = filename
        self.content_type = content_type
        self.seek(0)  # Reset pointer to start

    @property
    def name(self):
        return self._filename

    def getvalue(self):
        self.seek(0)
        return super().getvalue()


async def store_faiss_index(document_key: str, vector_store):
    """Store FAISS index in MongoDB"""
    try:
        processing_status[document_key]={'status':f"storing the vector store with document key {document_key}","Completed":6,"total":8}
        # Serialize the vector store
        
        serialized_index = pickle.dumps(vector_store)
        # logging.info("\n\n------Serialize-------\n\n")
        # Store in GridFS
        file_id = sync_db.fs.files.find_one(
            {"filename": f"faiss_index_{document_key}"},
            {"_id": 1}
        )
        # logging.info(f"\n\n\nFileid : {file_id}\n\n\n")
        if file_id:
            
            # Update existing index
            sync_db.fs.files.delete_one({"_id": file_id["_id"]})
        
        
        fs.put(
            serialized_index,
            filename=f"faiss_index_{document_key}"
        )
        # Update document status
        sync_db.documents.update_one(
            {"document_key": document_key},
            {"$set": {"index_status": "completed"}},
            upsert=True
        )
        
        logging.info(f"FAISS index stored for document {document_key}")
        processing_status[document_key]={'status':"Stored successfully","Completed":7,"total":8}
        processing_status[document_key]={'status': True,"Completed":8,"total":8}
    except Exception as e:
        logging.error(f"Error storing FAISS index: {str(e)}")
        processing_status[document_key]={'status':False}
        sync_db.documents.update_one(
            {"document_key": document_key},
            {"$set": {"index_status": "failed"}},
            upsert=True
        )
        raise





async def process_uploaded_file(file_content: bytes, filename: str):
    """Process uploaded file content with filename information"""
    # Create a CustomBytesIO object that maintains the filename
    file_obj = CustomBytesIO(file_content, filename)
    return file_obj


def process_document_background(file_content: bytes, filename: str, document_key: str):
    """Background task for document processing"""
    try:
        processing_status[document_key]={'status':"Extracting document","Completed":2,"total":8}
        # Extract content using modified logic that handles filename
        text, docs = extract_file_content(file_content, filename)
        
        # Create text chunks
        text_chunks = []
        metadata_chunks = []
        processing_status[document_key]={'status':"Document extracted","Completed":3,"total":8}
        for doc in docs:
            # Ensure filename is in metadata
            if filename and 'filename' not in doc.metadata:
                doc.metadata['filename'] = filename
            chunks, meta_chunks = get_text_chunks(doc.page_content, doc.metadata)
            text_chunks.extend(chunks)
            metadata_chunks.extend(meta_chunks)
        processing_status[document_key]={'status':f"Chunks has been created {len(text_chunks)}","Completed":4,"total":8}


        # Create vector store
        # processing_status[document_key]={'status': "storing vector store","Completed":4,"total":10}
        vector_store = get_vector_store(text_chunks, metadata_chunks)
        processing_status[document_key]={'status':"vector store created succesfully !","Completed":5,"total":8}
        
        # Store in MongoDB using a new event loop
        try:
            # processing_status[document_key]={'status':"storing vector store in DB"
            asyncio.run(store_faiss_index(document_key, vector_store))

        finally:
            # future.result()
            pass
            
            
    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
        # Update document status to failed using a new event loop
        new_loop = asyncio.new_event_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(
            async_db.documents.update_one(
                {"document_key": document_key},
                {"$set": {"index_status": "failed", "error": str(e)}},
                upsert=True
            ),new_loop)
            future.result()
        finally:
            new_loop.close()



async def get_faiss_index(document_key: str):
    """Retrieve FAISS index from MongoDB"""
    try:
        # Check if index exists
        grid_out = fs.find_one({"filename": f"faiss_index_{document_key}"})
        if grid_out:
            # Deserialize the index
            return pickle.loads(grid_out.read())
        return None
    except Exception as e:
        logging.error(f"Error retrieving FAISS index: {str(e)}")
        raise



async def load_qa_chain(document_key: str):
    """Load QA chain from MongoDB"""
    try:
        # Retrieve the serialized QA chain from GridFS
        file = sync_db.fs.find_one({"filename": f"qa_chain_{document_key}"})
        if file:
            serialized_chain =  sync_db.fs.open_download_stream(file._id).read()
            return pickle.loads(serialized_chain)
        return None
    except Exception as e:
        logging.error(f"Error loading QA chain: {str(e)}")
        return None

async def store_qa_chain(document_key: str, qa_chain):
    """Store QA chain in MongoDB"""
    try:
        # Serialize the QA chain
        serialized_chain = pickle.dumps(qa_chain)
        
        # Store in GridFS
        file =  sync_db.fs.find_one({"filename": f"qa_chain_{document_key}"})
        if file:
            # Update existing chain
             sync_db.fs.delete(file._id)
        
        sync_db.fs.upload_from_stream(
            f"qa_chain_{document_key}",
            serialized_chain
        )
        
        logging.info(f"QA chain stored for document {document_key}")
    except Exception as e:
        logging.error(f"Error storing QA chain: {str(e)}")