from fastapi import FastAPI, File, UploadFile
from utils import *
from app import *

app = FastAPI()

@app.post("/process_documents")
async def process_documents_endpoint(files: list[UploadFile]):
    """
    Endpoint to process uploaded documents.
    """
    try:
        result = await process_documents(files)
        return {"success": True, "message": "Documents processed successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/chat")
async def chat_endpoint(question: str):
    """
    Endpoint to handle user questions.
    """
    try:
        response = await handle_user_input(question)
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
