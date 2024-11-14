# app.py
from fastapi import FastAPI
from APP_LOGGING import setup_logging
from  routes.routes import rag
# Configure logging
if not setup_logging():
    print("logging not setuped")



app = FastAPI()
app.include_router(rag,prefix="/rag/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
