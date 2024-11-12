from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import uvicorn
from apis.routes import router
from utils.cache_manager import cleanup_cache
from config import MONGODB_URL

app = FastAPI(title="Document QA API")

@app.on_event("startup")
async def startup_event():
    # Initialize MongoDB connection
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    app.db = app.mongodb_client.document_qa_db
    
    # Start cache cleanup task
    asyncio.create_task(cleanup_cache())

@app.on_event("shutdown")
async def shutdown_event():
    app.mongodb_client.close()

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        # app,  # make sure this matches your file name and app variable
        "main:app",
        host="0.0.0.0",
        port=1111)