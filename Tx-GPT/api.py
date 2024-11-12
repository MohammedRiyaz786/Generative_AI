# api.py
from fastapi import FastAPI, File, UploadFile
from pymongo import MongoClient
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISSVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import extract_file_content, create_prompt

app = FastAPI()

# Initialize MongoDB connection
mongo_client = MongoClient('mongodb://localhost:27017')
db = mongo_client['rag_app']
vector_store_collection = db['vector_store']

# Initialize vector store and QA chain
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISSVectorStore.from_documents(
    documents=[],
    embedding=embeddings,
    client=vector_store_collection
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=Ollama(model="llama3.1", temperature=0.1),
    retriever=vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    ),
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    ),
    combine_docs_chain_kwargs={
        "prompt": create_prompt()
    },
    return_source_documents=True
)

@app.post("/process_documents")
async def process_documents(files: list[UploadFile] = File(...)):
    text_chunks = []
    metadata_chunks = []
    for file in files:
        text, docs = extract_file_content(file)
        if text and docs:
            text_chunks.extend(docs[0].page_content.split('\n'))
            metadata_chunks.extend([docs[0].metadata] * len(docs[0].page_content.split('\n')))

    if text_chunks:
        vector_store.add_texts(text_chunks, metadatas=metadata_chunks)
        return {"message": "Documents processed successfully!"}
    else:
        return {"message": "No text content could be extracted from the uploaded files."}

@app.post("/ask_question")
async def ask_question(question: str):
    result = qa_chain({"question": question})
    response = result.get('answer', '').strip()
    if not response:
        response = "I don't have enough information to answer this question."
    return {"response": response}