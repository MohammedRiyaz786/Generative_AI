from typing import List, Tuple, Optional
# from document_processor import *
import faiss
from datetime import datetime
from gridfs import GridFS
from bson.binary import Binary
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List, Tuple
import torch



#from langchain.llms import OpenAI

# async def create_faiss_index(texts: List[str]) -> Tuple[faiss.Index, List[str]]:
#     """Create new FAISS index"""
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(texts, embeddings)
#     return vector_store.index, texts


async def get_vector_store(text_chunks: List[str], metadata_chunks: List) -> Tuple[FAISS, List[str]]:
    """Create and save FAISS index from text chunks and metadata."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #embeddings.client.to(device)
    
    batch_size = 32
    vector_store = None

    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i+batch_size]
        batch_metadata = metadata_chunks[i:i+batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch_texts, embedding=embeddings, metadatas=batch_metadata)
        else:
            vector_store.add_texts(batch_texts, metadatas=batch_metadata)
    
    #vector_store.save_local("faiss_index")
    
    
    return vector_store.index, text_chunks

# async def get_qa_chain(faiss_index, texts: List[str]):
#     """Create QA chain from FAISS index"""
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS(
#         embeddings.embed_query,
#         faiss_index,
#         texts=texts
#     )
    
#     return RetrievalQA.from_chain_type(
#         llm=OpenAI(),
#         chain_type="stuff",
#         retriever=vector_store.as_retriever()
#     )

from typing import List

async def create_qa_chain(faiss_index, texts: List[str]):
    """Create QA chain using FAISS index with customized prompt and retriever settings."""
    prompt_template = """You are a polite, respectful, and efficient AI assistant.

    IF the user's message matches ANY of these patterns:
    - "Hi", "Hello", "Hey", "Hii", "Hola" (just greeting)
    - "My name is [any name]" 
    - "I am [any name]"
    - "[any greeting] my name is [any name]"
    - "[any greeting] I am [any name]"
    THEN respond only with: "Hello! How can I assist you today?"

    OTHERWISE:
    1. Use only the provided information:
    - Context: {context}
    - Chat History: {chat_history} 
    - Current Question: {question}

    2. Your response must be:
    - Direct and to-the-point
    - Based only on given context and history
    - Without any explanations about your capabilities
    - Without mentioning sources or references

    3. If the answer cannot be found in context or history:
    Response should be only: "I apologize, but I don't have enough information to answer your question."

    4. Never start responses with:
    - "Based on..."
    - "According to..." 
    - "I understand..."
    - "Let me..."

    5. Never end responses with:
    - "Is there anything else..."
    - "Let me know if..."
    - "Feel free to..."

    6. If the user uses any abusive or inappropriate language, respond politely and avoid escalation:
    "I apologize, but I don't engage with that type of language. How else can I assist you today?"

    Question: {question}"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS(
        embeddings.embed_query,
        faiss_index,
        texts=texts
    )
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    llm = Ollama(model="llama3.1", temperature=0.1)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )



async def store_index_in_mongodb(doc_hash: str, faiss_index, texts: List[str], db):
    """Store FAISS index in MongoDB"""
    try:
        faiss_binary = faiss.serialize_index(faiss_index)
        
        document = {
            "doc_hash": doc_hash,
            "texts": texts,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
        
        if len(faiss_binary) > 16_000_000:
            fs = GridFS(db)
            file_id = await fs.put(
                faiss_binary,
                filename=f"faiss_index_{doc_hash}",
                metadata={"doc_hash": doc_hash}
            )
            document["index_file_id"] = file_id
        else:
            document["index_binary"] = Binary(faiss_binary)
        
        await db.faiss_indexes.insert_one(document)
        
    except Exception as e:
        print(f"Error storing index in MongoDB: {e}")

async def get_index_from_mongodb(doc_hash: str, db) -> Optional[Tuple[faiss.Index, List[str]]]:
    """Retrieve FAISS index from MongoDB"""
    try:
        doc = await db.faiss_indexes.find_one({"doc_hash": doc_hash})
        if not doc:
            return None, None
            
        await db.faiss_indexes.update_one(
            {"doc_hash": doc_hash},
            {"$set": {"last_accessed": datetime.utcnow()}}
        )
        
        if "index_file_id" in doc:
            fs = GridFS(db)
            file_data = await fs.get(doc["index_file_id"])
            faiss_binary = await file_data.read()
        else:
            faiss_binary = doc["index_binary"]
            
        faiss_index = faiss.deserialize_index(faiss_binary)
        return faiss_index, doc["texts"]
        
    except Exception as e:
        print(f"Error retrieving index from MongoDB: {e}")
        return None, None