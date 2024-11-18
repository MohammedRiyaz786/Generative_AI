# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from APP_LOGGING import setup_logging
import logging
from  routes.routes import rag
# Configure logging
if not setup_logging():
    print("logging not setuped")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag,prefix="/rag/v1")


if __name__ == "__main__":
    logging.info("Started FastAPI Server !")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
