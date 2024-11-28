import streamlit as st
import re
import logging
from utils import*
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import torch
from PIL import Image
import io
import cv2
import numpy as np
import pytesseract
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import json
from datetime import datetime
from typing import List, Dict, Any


# Configure logging
logging.basicConfig(
    filename='query_metrics_new.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variable to track if documents are processed
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    try:
        if file_type in image_extensions:
            logging.info(f"Processing image file: {uploaded_file.name}")
            return process_image(uploaded_file)
        elif file_type == 'pdf':
            logging.info(f"Processing PDF file: {uploaded_file.name}")
            tabular_text, tabular_docs = get_pdf_text([uploaded_file])
            non_tabular_text, non_tabular_docs = get_non_table_pdf_text([uploaded_file])
            return tabular_text + non_tabular_text, tabular_docs + non_tabular_docs
        elif file_type == 'csv':
            logging.info(f"Processing CSV file: {uploaded_file.name}")
            text = get_csv_text(uploaded_file)
            return text, [Document(page_content=text, metadata={'source': 'csv'})]
        elif file_type in ['xls', 'xlsx']:
            logging.info(f"Processing Excel file: {uploaded_file.name}")
            text, docs = get_excel_text([uploaded_file])
            return text, docs
        elif file_type in ['pptx']:
            logging.info(f"Processing PowerPoint file: {uploaded_file.name}")
            text, docs = get_ppt_text([uploaded_file])
            return text, docs
        elif file_type in ['doc', 'docx']:
            logging.info(f"Processing Word file: {uploaded_file.name}")
            text, docs = get_word_text([uploaded_file])
            return text, docs
        else:
            error_msg = f"Unsupported file format: {file_type}"
            logging.error(error_msg)
            st.error(error_msg)
            return "", []
    except Exception as e:
        error_msg = f"Error processing file {uploaded_file.name}: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        return "", []

def calculate_relevance_score(query: str, context: str) -> float:
    """
    Calculate relevance score between query and context using cosine similarity
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Get embeddings for query and context
    query_embedding = embeddings.embed_query(query)
    context_embedding = embeddings.embed_query(context)
    
    # Calculate cosine similarity
    similarity = np.dot(query_embedding, context_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
    )
    
    return float(similarity)

def calculate_confidence_score(
    query: str,
    context: str,
    answer: str,
    relevance_weight: float = 0.4,
    length_weight: float = 0.3,
    coherence_weight: float = 0.3
) -> float:
    """
    Calculate confidence score based on multiple factors
    """
    # Relevance score between query and context
    relevance_score = calculate_relevance_score(query, context)
    
    # Length-based score (penalize too short or too long answers)
    optimal_length = 100  # Optimal answer length in characters
    length_ratio = min(len(answer), optimal_length) / optimal_length
    length_score = min(1.0, length_ratio)
    
    # Coherence score (based on answer structure)
    has_structure = bool(re.search(r'[.!?][\s]{1,2}[A-Z]', answer))  # Check for proper sentences
    has_paragraphs = '\n\n' in answer
    coherence_score = 0.5 + (0.25 if has_structure else 0) + (0.25 if has_paragraphs else 0)
    
    # Weighted average
    final_score = (
        relevance_weight * relevance_score +
        length_weight * length_score +
        coherence_weight * coherence_score
    )
    
    return min(1.0, max(0.0, final_score))

def log_query_metrics(
    query: str,
    answer: str,
    context: str,
    execution_time: float,
    relevance_score: float,
    confidence_score: float
) -> None:
    """
    Log query metrics for analysis
    """
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer_length': len(answer),
        'context_length': len(context),
        'execution_time_ms': execution_time * 1000,
        'relevance_score': relevance_score,
        'confidence_score': confidence_score
    }
    
    logging.info(f"Query Metrics: {metrics}")

def get_text_chunks(text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata for _ in chunks]
    return chunks, metadata_chunks

def get_vector_store(text_chunks, metadata_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings.client.to(device)
    
    batch_size = 32
    vector_store = None
    
    for i in range(0, len(text_chunks), batch_size):
        batch_texts = text_chunks[i:i+batch_size]
        batch_metadata = metadata_chunks[i:i+batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch_texts, embedding=embeddings, metadatas=batch_metadata)
        else:
            vector_store.add_texts(batch_texts, metadatas=batch_metadata)
    
    vector_store.save_local("faiss_index")

    return vector_store

def create_qa_chain():
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

    llm = Ollama(model="llama3.1", temperature=0.1)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Create the base retriever
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # Create LLM chain extractor for document compression
    chain_extractor = LLMChainExtractor.from_llm(llm)
    
    # Create contextual compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=chain_extractor,
        base_retriever=base_retriever
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,  # Use the compression retriever instead of base retriever
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def init_session_state():
    """Initialize session state variables"""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'user_context' not in st.session_state:
        st.session_state.user_context = {}
    if 'vector_store_path' not in st.session_state:
        st.session_state.vector_store_path = "faiss_index"



def handle_user_input(user_question: str):
    try:
        logging.info(f"Processing user question: {user_question}")
        
        # Handle basic greeting patterns (keep existing code)
        if re.match(r"^(?i)(hi|hello|hey)$", user_question):
            response = "Hello! How can I help you today?"
            
        # Handle other existing patterns...
        else:
            if not st.session_state.docs_processed:
                response = "Please upload and process some documents first."
            else:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    response = "I'm having trouble accessing the document knowledge base."
                else:
                    start_time = datetime.now()
                    
                    if not is_conversational_input(user_question):
                        result = qa_chain({"question": user_question})
                        response = result.get('answer', '').strip()
                        
                        # Calculate metrics
                        execution_time = (datetime.now() - start_time).total_seconds()
                        context = " ".join([doc.page_content for doc in result.get('source_documents', [])])
                        
                        relevance_score = calculate_relevance_score(user_question, context)
                        confidence_score = calculate_confidence_score(
                            user_question,
                            context,
                            response
                        )
                        
                        # Log metrics
                        log_query_metrics(
                            user_question,
                            response,
                            context,
                            execution_time,
                            relevance_score,
                            confidence_score
                        )
                        
                        # Add scores to response for display
                        response_with_metrics = (
                            f"{response}\n\n"
                            f"Relevance Score: {relevance_score:.2f}\n"
                            f"Confidence Score: {confidence_score:.2f}"
                        )
                        response = response_with_metrics
                    else:
                        response = handle_conversation(user_question)
        
        # Update conversation (keep existing code)
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        
        st.session_state.conversation.append({
            "user": user_question,
            "assistant": response
        })
        
        # Display conversation (keep existing code)
        for message in st.session_state.conversation:
            with st.chat_message("user", avatar="🧑"):
                st.write(message["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["assistant"])
                
    except Exception as e:
        logging.error(f"Error in handle_user_input: {str(e)}", exc_info=True)
        st.error(f"An error occurred while processing your question: {str(e)}")

def is_conversational_input(text: str) -> bool:
    """Determine if the input is conversational rather than a question about documents"""
    conversational_patterns = [
        r"(?i)my name is \w+",
        r"(?i)^(hi|hello|hey)$",
        r"(?i)what('s| is) my name\??",
        r"(?i)^(thanks|thank you|bye|goodbye)$",
        r"(?i)^(yes|no|maybe)$",
        r"(?i)^(good morning|good afternoon|good evening)$",
        r"(?i)how are you",
        r"(?i)nice to meet you"
    ]
    
    return any(re.match(pattern, text.strip()) for pattern in conversational_patterns)

def handle_conversation(text: str) -> str:
    """Handle conversational inputs that don't require document knowledge"""
    # Get user's name from context if available
    user_name = st.session_state.user_context.get('name', '')
    
    # Handle different types of conversational inputs
    text_lower = text.lower().strip()
    
    if "how are you" in text_lower:
        return f"I'm doing well{', ' + user_name if user_name else ''}! How can I help you today?"
    
    elif any(word in text_lower for word in ["thanks", "thank you"]):
        return f"You're welcome{', ' + user_name if user_name else ''}! Let me know if you need anything else."
    
    elif any(word in text_lower for word in ["bye", "goodbye"]):
        return f"Goodbye{', ' + user_name if user_name else ''}! Have a great day!"
    
    elif "nice to meet you" in text_lower:
        return f"Nice to meet you too{', ' + user_name if user_name else ''}!"
    
    elif any(greeting in text_lower for greeting in ["good morning", "good afternoon", "good evening"]):
        return f"{text.capitalize()}{', ' + user_name if user_name else ''}! How can I help you today?"
    
    # For other conversational inputs that don't match specific patterns
    return "How can I help you today?"

def process_documents(uploaded_files):
    """
    Process uploaded documents and create vector store
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        all_text = ""
        all_docs = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            logging.info(f"Processing file: {uploaded_file.name}")
            text, docs = extract_file_content(uploaded_file)
            if text and docs:
                all_text += text + "\n\n"
                all_docs.extend(docs)
        
        if not all_text.strip():
            logging.warning("No text content extracted from uploaded files")
            return False
            
        # Create text chunks with metadata
        text_chunks = []
        metadata_chunks = []
        
        for doc in all_docs:
            chunks, meta_chunks = get_text_chunks(doc.page_content, doc.metadata)
            text_chunks.extend(chunks)
            metadata_chunks.extend(meta_chunks)
        
        if not text_chunks:
            logging.warning("No text chunks created from documents")
            return False
            
        # Create and save vector store
        vector_store = get_vector_store(text_chunks, metadata_chunks)
        if vector_store is None:
            logging.error("Failed to create vector store")
            return False
            
        logging.info("Documents processed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in process_documents: {str(e)}", exc_info=True)
        return False
    
def clear_conversation():
    """Clear only the conversation history without affecting processed documents"""
    st.session_state.memory.clear()
    st.session_state.conversation = []
    st.success("Conversation cleared! You can continue asking questions about the processed documents.")

# def main():
#     try:
#         st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
#         init_session_state()
        
#         st.header("Chat with Documents and Images using LLAMA3 🦙")
        
#         # Sidebar
#         with st.sidebar:
#             st.title("Document Processing")
            
#             # File uploader
#             uploaded_files = st.file_uploader(
#                 "Upload documents or images",
#                 accept_multiple_files=True,
#                 type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
#             )
            
#             # Process button
#             if st.button("Process Documents"):
#                 if uploaded_files:
#                     with st.spinner("Processing documents..."):
#                         if process_documents(uploaded_files):
#                             st.session_state.docs_processed = True
#                             st.success("Documents processed successfully!")
#                         else:
#                             st.error("Error processing documents. Check logs for details.")
#                 else:
#                     st.warning("Please upload files before processing.")
            
#             # Separate buttons for clearing conversation and everything
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 if st.button("Clear Conversation"):
#                     clear_conversation()
            
#             with col2:
#                 if st.button("Clear Everything"):
#                     st.session_state.memory.clear()
#                     st.session_state.conversation = []
#                     st.session_state.docs_processed = False
#                     st.success("Everything has been cleared!")
            
#             # Status indicator
#             st.write("Status:")
#             st.write(f"Documents Processed: {'✅' if st.session_state.docs_processed else '❌'}")



#         # Chat interface
#         user_question = st.chat_input("Ask a question about your documents")
#         if user_question:
#             handle_user_input(user_question)
            
#     except Exception as e:
#         logging.error(f"Error in main: {str(e)}", exc_info=True)
#         st.error("An error occurred in the application. Please check the logs.")


# if __name__ == "__main__":
#     main()

def main():
    try:
        st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
        init_session_state()
        
        st.header("Chat with Documents and Images using LLAMA3 🦙")
        
        # Sidebar
        with st.sidebar:
            st.title("Document Processing")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload documents or images",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
            )
            
            # Process button
            if st.button("Process Documents"):
                if uploaded_files:
                    with st.spinner("Processing documents..."):
                        if process_documents(uploaded_files):
                            st.session_state.docs_processed = True
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Error processing documents. Check logs for details.")
                else:
                    st.warning("Please upload files before processing.")
            
            # Separate buttons for clearing conversation and everything
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Conversation"):
                    clear_conversation()
            
            with col2:
                if st.button("Clear Everything"):
                    st.session_state.memory.clear()
                    st.session_state.conversation = []
                    st.session_state.docs_processed = False
                    st.success("Everything has been cleared!")
            
            # Status indicator
            st.write("Status:")
            st.write(f"Documents Processed: {'✅' if st.session_state.docs_processed else '❌'}")

            # New sidebar metrics for displaying query metrics
            st.sidebar.markdown("### Query Metrics")
            st.sidebar.metric("Relevance Score", "-")
            st.sidebar.metric("Confidence Score", "-")

        # Chat interface
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            handle_user_input(user_question)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        st.error("An error occurred in the application. Please check the logs.")


if __name__ == "__main__":
    main()








































































