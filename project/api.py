from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient
import pickle
import base64
from utils import process_documents, create_qa_chain
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client.rag_database
collection = db.faiss_indexes

class Question(BaseModel):
    text: str

@app.post("/process-documents")
async def api_process_documents(files: List[UploadFile] = File(...)):
    try:
        # Process documents and get vector store
        vector_store = process_documents(files)
        
        # Serialize FAISS index
        serialized_index = pickle.dumps(vector_store)
        base64_index = base64.b64encode(serialized_index).decode()
        
        # Store in MongoDB
        await collection.update_one(
            {"index_id": "current"},
            {"$set": {"faiss_index": base64_index}},
            upsert=True
        )
        
        return {"message": "Documents processed and index stored successfully"}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/query")
async def query_documents(question: Question):
    try:
        # Retrieve FAISS index from MongoDB
        index_doc = await collection.find_one({"index_id": "current"})
        if not index_doc:
            return {"error": "No index found"}, 404
            
        # Deserialize index
        serialized_index = base64.b64decode(index_doc["faiss_index"])
        vector_store = pickle.loads(serialized_index)
        
        # Create QA chain and get response
        qa_chain = create_qa_chain(vector_store)
        result = qa_chain({"question": question.text})
        
        return {"answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/clear-conversation")
async def clear_conversation():
    # Clear conversation logic here
    return {"message": "Conversation cleared"}

@app.post("/clear-everything")
async def clear_everything():
    try:
        # Clear MongoDB index
        await collection.delete_many({})
        return {"message": "Everything cleared"}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0000", port=8000)