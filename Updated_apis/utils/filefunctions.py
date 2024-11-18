
from .utils import (
    process_image,
    extract_tables_from_image,
    extract_formulas_from_image,
    get_pdf_text,
    get_non_table_pdf_text,
    get_csv_text,
    get_excel_text,
    get_ppt_text,
    get_word_text
)

from io import BytesIO
import logging
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileWrapper:
    """Wrapper class to maintain filename information for file-like objects"""
    def __init__(self, file_obj, filename):
        self.file_obj = file_obj
        self.name = filename

    def __getattr__(self, attr):
        return getattr(self.file_obj, attr)

    def seek(self, *args, **kwargs):
        return self.file_obj.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.file_obj.read(*args, **kwargs)


def extract_file_content(file_content: bytes, filename: str = None) -> tuple[str, list]:
    """Extract content from file bytes while maintaining filename information"""
    try:
        # Create BytesIO object wrapped with filename
        file_obj = BytesIO(file_content)
        wrapped_file = FileWrapper(file_obj, filename) if filename else file_obj
        # logging.info(f"Extracting the content of file {filename}")
        # Detect file type from content
        import magic
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(file_content)
        print(file_type)
        # Process based on file type
        if 'image' in file_type:
            return process_image(wrapped_file)
        elif 'pdf' in file_type:
            # logging.info(f"PDF detected ")
            # Reset file pointer before each use
            wrapped_file.seek(0)
            tabular_text, tabular_docs = get_pdf_text([wrapped_file])
            
            wrapped_file.seek(0)
            non_tabular_text, non_tabular_docs = get_non_table_pdf_text([wrapped_file])
            
            return tabular_text + non_tabular_text, tabular_docs + non_tabular_docs
        elif 'csv' in file_type:
            wrapped_file.seek(0)
            text = get_csv_text(wrapped_file)
            metadata = {'source': 'csv', 'filename': filename} if filename else {'source': 'csv'}
            return text, [Document(page_content=text, metadata=metadata)]
        elif 'excel' in file_type or 'spreadsheet' in file_type:
            wrapped_file.seek(0)
            text, docs = get_excel_text([wrapped_file])
            return text, docs
        elif 'powerpoint' in file_type:
            wrapped_file.seek(0)
            text, docs = get_ppt_text([wrapped_file])
            return text, docs
        elif 'word' in file_type or 'document' in file_type:
            wrapped_file.seek(0)
            text, docs = get_word_text([wrapped_file])
            return text, docs
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error extracting file content: {str(e)}")
        raise
    finally:
        file_obj.close()

def get_text_chunks(text: str, metadata: dict) -> tuple[list, list]:

    """Split text into chunks with metadata"""
    # logging.info(f"Creating chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata for _ in chunks]
    return chunks, metadata_chunks

def get_vector_store(text_chunks: list, metadata_chunks: list) -> FAISS:
    """Create FAISS vector store from text chunks"""
    try:
        # logging.info("creating vextor stores")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # embeddings.client.to(device)
        
        batch_size = 32
        vector_store = None
        
        for i in range(0, len(text_chunks), batch_size):
            batch_texts = text_chunks[i:i+batch_size]
            batch_metadata = metadata_chunks[i:i+batch_size]
            
            if vector_store is None:
                vector_store = FAISS.from_texts(
                    batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadata
                )
            else:
                vector_store.add_texts(batch_texts, metadatas=batch_metadata)
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def create_qa_chain(vector_store: FAISS, memory):
    """Create QA chain with vector store and memory"""
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

    Question: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain
