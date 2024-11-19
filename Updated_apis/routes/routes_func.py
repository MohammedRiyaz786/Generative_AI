import logging
import io
import pickle
import asyncio
from typing import List

# from .routes import processing_status
processing_status = {}
from .db import sync_db,async_db,fs
from StatusTracker import status_tracker
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
        status_tracker.add_status(document_key=document_key,
                                  message="Starting vector store storage",
                                  completed=6
        )
        
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
        status_tracker.add_status(
            document_key,
            "Storing vector data in database",
            7
        )
        # Update document status
        sync_db.documents.update_one(
            {"document_key": document_key},
            {"$set": {"index_status": "completed"}},
            upsert=True
        )
        status_tracker.add_status(
            document_key,
            "Processing completed successfully",
            8
        )
        logging.info(f"FAISS index stored for document {document_key}")
        

        # status_tracker.update_status(
        #     document_key,
        #     ProcessingStage.COMPLETED,
        #     "Processing completed successfully",
        #     8
        # ) 
    except Exception as e:
        logging.error(f"Error storing FAISS index: {str(e)}")
        status_tracker.add_status(
            document_key,
            f"Error storing FAISS index: {str(e)}",
            -1
        )
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

def process_document_background(file_contents: List[tuple], document_key: str):
    """Background task for processing multiple documents"""
    try:
        status_tracker.add_status(
            document_key,
            "Starting document processing",
            1
        )
        
        all_text_chunks = []
        all_metadata_chunks = []
        total_files = len(file_contents)
        
        # Process each file one at a time
        for index, (file_content, filename) in enumerate(file_contents, 1):
            try:
                status_tracker.add_status(
                    document_key,
                    f"Processing {total_files} files",
                    2
                )
                
                # Extract content for single file using existing function
                text, docs = extract_file_content(file_content, filename)
                
                # Process chunks for this document
                for doc in docs:
                    if filename and 'filename' not in doc.metadata:
                        doc.metadata['filename'] = filename
                    chunks, meta_chunks = get_text_chunks(doc.page_content, doc.metadata)
                    all_text_chunks.extend(chunks)
                    all_metadata_chunks.extend(meta_chunks)
                
                status_tracker.add_status(
                    document_key,
                    "File processing completed",
                    3
                )
                        
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                continue  # Continue with next file even if one fails
        
        if not all_text_chunks:
            raise Exception("No documents were successfully processed")

        status_tracker.add_status(
            document_key,
            f"Created {len(all_text_chunks)} chunks from documents",
            4
        )

        # Create single vector store from all accumulated chunks
        vector_store = get_vector_store(all_text_chunks, all_metadata_chunks)
        
        status_tracker.add_status(
            document_key,
            "Vector store created successfully",
            5
        )
        
        # Store in MongoDB using a new event loop
        try:
            asyncio.run(store_faiss_index(document_key, vector_store))
            
        except Exception as e:
            raise Exception(f"Error storing FAISS index: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
        # status_tracker.add_status(
        #     document_key,
        #     f"Error occurred: {str(e)}",
        #     -1
        # )
        new_loop = asyncio.new_event_loop()
        try:
            future = asyncio.run_coroutine_threadsafe(
                async_db.documents.update_one(
                    {"document_key": document_key},
                    {"$set": {"index_status": "failed", "error": str(e)}},
                    upsert=True
                ),
                new_loop
            )
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