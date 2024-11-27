import streamlit as st
import re
import logging
from utils import (
    get_pdf_text, 
    get_csv_text, 
    get_excel_text, 
    get_non_table_pdf_text, 
    get_ppt_text, 
    get_word_text,
    process_image,
    extract_tables_from_image,
    extract_formulas_from_image
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch
from PIL import Image
import io
import cv2
import numpy as np
import pytesseract
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Any, Tuple
from datetime import datetime
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from scipy.spatial.distance import cosine
import numpy as np




logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    
def calculate_relevance_score(question_embedding: List[float], context_embedding: List[float]) -> float:
    """
    Calculate cosine similarity between question and context embeddings
    """
    return 1 - cosine(question_embedding, context_embedding)

def calculate_confidence_score(
    answer: str,
    question: str,
    context: str,
    llm_metadata: Dict[Any, Any]
) -> float:
    """
    Calculate confidence score based on multiple factors
    """
    # Initialize base score
    score = 1.0
    
    # Factor 1: Answer length relative to context
    length_ratio = len(answer) / len(context)
    if length_ratio < 0.01 or length_ratio > 0.8:
        score *= 0.8
    
    # Factor 2: Question terms present in answer
    question_terms = set(question.lower().split())
    answer_terms = set(answer.lower().split())
    term_overlap = len(question_terms.intersection(answer_terms)) / len(question_terms)
    score *= (0.5 + term_overlap)
    
    # Factor 3: LLM confidence metadata if available
    if llm_metadata.get('confidence'):
        score *= llm_metadata['confidence']
    
    # Factor 4: Source document relevance
    if llm_metadata.get('source_relevance'):
        score *= llm_metadata['source_relevance']
    
    return min(1.0, score)

def log_query_metrics(
    question: str,
    answer: str,
    relevance_score: float,
    confidence_score: float,
    response_time: float,
    metadata: Dict[str, Any]
) -> None:
    """
    Log query metrics to file and database
    """
    timestamp = datetime.now().isoformat()
    metrics = {
        'timestamp': timestamp,
        'question': question,
        'answer_length': len(answer),
        'relevance_score': relevance_score,
        'confidence_score': confidence_score,
        'response_time_ms': response_time,
        'metadata': metadata
    }
    
    # Log to file
    logging.info(f"Query Metrics: {metrics}")
    
    # You can also implement database logging here
    # store_metrics_in_db(metrics)

def create_enhanced_retriever(base_retriever, llm):
    """
    Create an enhanced retriever with LLMChainExtractor and ContextualCompressionRetriever
    """
    # Create LLMChainExtractor compressor
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Create ContextualCompressionRetriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
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

    3. Rate your confidence in the answer on a scale of 0-1
    4. Include relevant source document sections

    Question: {question}
    Confidence Score: {confidence_score}
    Relevant Context Score: {relevance_score}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history", "confidence_score", "relevance_score"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # Create enhanced retriever
    enhanced_retriever = create_enhanced_retriever(base_retriever, llm)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=enhanced_retriever,
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

# Modify handle_user_input function
def handle_user_input(user_question: str):
    try:
        start_time = datetime.now()
        
        if re.match(r"^(?i)(hi|hello|hey)$", user_question):
            response = "Hello! How can I help you today?"
            metadata = {'type': 'greeting'}
            
        elif name_match := re.match(r"(?i)my name is (\w+)", user_question):
            name = name_match.group(1)
            st.session_state.user_context['name'] = name
            response = f"Hello {name}! How can I help you today?"
            metadata = {'type': 'name_introduction', 'name': name}
            
        elif re.match(r"(?i)what('s| is) my name\??", user_question):
            name = st.session_state.user_context.get('name')
            response = f"Your name is {name}." if name else "You haven't told me your name yet."
            metadata = {'type': 'name_query'}
            
        else:
            if not st.session_state.docs_processed:
                response = "Please upload and process some documents first."
                metadata = {'type': 'error', 'reason': 'no_documents'}
            else:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    response = "I'm having trouble accessing the document knowledge base."
                    metadata = {'type': 'error', 'reason': 'qa_chain_creation_failed'}
                else:
                    if not is_conversational_input(user_question):
                        # Get embeddings for question and context
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        question_embedding = embeddings.embed_query(user_question)
                        
                        # Get response and relevant documents
                        result = qa_chain({"question": user_question})
                        response = result.get('answer', '').strip()
                        
                        # Calculate scores
                        relevant_docs = result.get('source_documents', [])
                        if relevant_docs:
                            context_embedding = embeddings.embed_query(relevant_docs[0].page_content)
                            relevance_score = calculate_relevance_score(question_embedding, context_embedding)
                            confidence_score = calculate_confidence_score(
                                response,
                                user_question,
                                relevant_docs[0].page_content,
                                result.get('metadata', {})
                            )
                        else:
                            relevance_score = 0.0
                            confidence_score = 0.0
                        
                        # Log metrics
                        end_time = datetime.now()
                        response_time = (end_time - start_time).total_seconds() * 1000
                        
                        metadata = {
                            'type': 'qa_response',
                            'relevance_score': relevance_score,
                            'confidence_score': confidence_score,
                            'response_time_ms': response_time,
                            'num_relevant_docs': len(relevant_docs)
                        }
                        
                        log_query_metrics(
                            user_question,
                            response,
                            relevance_score,
                            confidence_score,
                            response_time,
                            metadata
                        )
                        
                        # Add scores to response for display
                        response = f"{response}\n\nRelevance Score: {relevance_score:.2f}\nConfidence Score: {confidence_score:.2f}"
                    else:
                        response = handle_conversation(user_question)
                        metadata = {'type': 'conversation'}
        
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
            
        st.session_state.conversation.append({
            "user": user_question,
            "assistant": response,
            "metadata": metadata
        })
        
        for message in st.session_state.conversation:
            with st.chat_message("user", avatar="🧑"):
                st.write(message["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["assistant"])
                if 'metadata' in message and message['metadata'].get('type') == 'qa_response':
                    st.info(f"Response Time: {message['metadata']['response_time_ms']:.0f}ms")
                
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

            # Add Metrics Display Section
            if st.session_state.docs_processed:
                st.write("---")
                st.subheader("Query Metrics")
                if 'conversation' in st.session_state and st.session_state.conversation:
                    last_query = st.session_state.conversation[-1]
                    if 'metadata' in last_query and last_query['metadata'].get('type') == 'qa_response':
                        metrics = last_query['metadata']
                        st.metric("Response Time", f"{metrics['response_time_ms']:.0f}ms")
                        st.metric("Relevance Score", f"{metrics['relevance_score']:.2f}")
                        st.metric("Confidence Score", f"{metrics['confidence_score']:.2f}")
                        st.metric("Retrieved Documents", metrics['num_relevant_docs'])

        # Main chat interface with metrics
        if st.session_state.docs_processed:
            # Display conversation history with metrics
            for message in st.session_state.conversation:
                with st.chat_message("user", avatar="🧑"):
                    st.write(message["user"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["assistant"])
                    # Display metrics for QA responses
                    if 'metadata' in message and message['metadata'].get('type') == 'qa_response':
                        with st.expander("View Response Metrics"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Relevance", f"{message['metadata']['relevance_score']:.2f}")
                                st.metric("Response Time", f"{message['metadata']['response_time_ms']:.0f}ms")
                            with col2:
                                st.metric("Confidence", f"{message['metadata']['confidence_score']:.2f}")
                                st.metric("Retrieved Docs", message['metadata']['num_relevant_docs'])

        # Chat input
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            handle_user_input(user_question)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        st.error("An error occurred in the application. Please check the logs.")

if __name__ == "__main__":
    main()