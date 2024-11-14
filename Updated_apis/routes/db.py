import motor
import motor.motor_asyncio
import os 
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
load_dotenv()


MONGO_URI = os.environ.get("MONGO_URI","mongodb://localhost:27017/test")
async_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
async_db = async_client.document_chat

sync_client = MongoClient(MONGO_URI)
sync_db = sync_client.document_chat
fs = GridFS(sync_db)

