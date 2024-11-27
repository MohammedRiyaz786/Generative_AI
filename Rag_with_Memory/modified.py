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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import torch
from PIL import Image
import io
import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Dict, Any, Tuple
import time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Initialize logging
logging.basicConfig(
    filename='app_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize session state variables
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

class EnhancedEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings.client.to(self.device)
    
    def embed_query(self, text: str) -> List[float]:
        cleaned_text = self._preprocess_text(text)
        try:
            embedding = self.embeddings.embed_query(cleaned_text)
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        max_length = 512
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        return text

def calculate_relevance_score(query: str, context: str) -> float:
    """
    Calculate relevance score between query and context using semantic similarity
    """
    try:
        embedder = EnhancedEmbeddings()
        query_embedding = embedder.embed_query(query)
        context_embedding = embedder.embed_query(context)
        
        similarity = np.dot(query_embedding, context_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
        )
        return float(similarity)
    except Exception as e:
        logging.error(f"Error calculating relevance score: {str(e)}")
        return 0.0

def calculate_confidence_score(
    relevance_score: float,
    response_length: int,
    context_length: int,
    token_match_ratio: float = 0.0
) -> float:
    """
    Calculate confidence score based on multiple factors
    """
    try:
        # Normalize response and context lengths
        length_ratio = min(response_length / max(context_length, 1), 1.0)
        
        # Weighted scoring
        weights = {
            'relevance': 0.5,
            'length_ratio': 0.3,
            'token_match': 0.2
        }
        
        confidence = (
            weights['relevance'] * relevance_score +
            weights['length_ratio'] * length_ratio +
            weights['token_match'] * token_match_ratio
        )
        
        return min(max(confidence, 0.0), 1.0)
    except Exception as e:
        logging.error(f"Error calculating confidence score: {str(e)}")
        return 0.0

def log_query_metrics(
    query: str,
    response: str,
    context: str,
    relevance_score: float,
    confidence_score: float,
    processing_time: float,
    additional_metrics: Dict[str, Any] = None
) -> None:
    """
    Log comprehensive metrics for each query
    """
    try:
        metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'query': query,
            'response_length': len(response),
            'context_length': len(context),
            'relevance_score': f"{relevance_score:.2f}",
            'confidence_score': f"{confidence_score:.2f}",
            'processing_time_ms': f"{processing_time * 1000:.2f}"
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        logging.info(f"Query Metrics: {metrics}")
        
        # Save metrics to session state for analysis
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = []
        st.session_state.metrics_history.append(metrics)
        
    except Exception as e:
        logging.error(f"Error logging metrics: {str(e)}")

def create_enhanced_qa_chain():
    """
    Create an enhanced QA chain with compression and metrics
    """
    try:
        prompt_template = """You are a polite, respectful, and efficient AI assistant.
        
        Consider the following metrics while formulating your response:
        - Relevance Score: {relevance_score}
        - Confidence Score: {confidence_score}
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        
        If confidence score is below 0.5, preface your response with:
        "Note: I'm not entirely confident about this answer, but based on available information:"
        
        Response should be:
        1. Direct and concise
        2. Based solely on provided context
        3. Include relevant quotes when possible
        4. Indicate uncertainty when appropriate
        
        Question: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history", "relevance_score", "confidence_score"]
        )

        # Initialize LLM
        llm = Ollama(model="llama3.1", temperature=0.1)
        
        # Create embeddings and load vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Create base retriever
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        # Create document compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
    
    except Exception as e:
        logging.error(f"Error creating QA chain: {str(e)}")
        return None

def handle_query_with_metrics(user_question: str) -> Tuple[str, Dict[str, float]]:
    """
    Process query and return response with metrics
    """
    try:
        start_time = time.time()
        
        if not st.session_state.docs_processed:
            return "Please upload and process documents first.", {}
        
        qa_chain = create_enhanced_qa_chain()
        if qa_chain is None:
            return "Error initializing QA system.", {}
        
        # Get relevant documents
        retrieved_docs = qa_chain.retriever.get_relevant_documents(user_question)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        
        # Calculate initial relevance score
        relevance_score = calculate_relevance_score(user_question, context)
        
        # Get response
        result = qa_chain({
            "question": user_question,
            "relevance_score": relevance_score,
            "confidence_score": 0.0  # Will be updated after response
        })
        
        response = result.get('answer', '').strip()
        
        # Calculate final scores
        confidence_score = calculate_confidence_score(
            relevance_score,
            len(response),
            len(context)
        )
        
        # Log metrics
        processing_time = time.time() - start_time
        log_query_metrics(
            user_question,
            response,
            context,
            relevance_score,
            confidence_score,
            processing_time
        )
        
        metrics = {
            'relevance_score': relevance_score,
            'confidence_score': confidence_score,
            'processing_time': processing_time
        }
        
        return response, metrics
    
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return f"An error occurred: {str(e)}", {}

def display_metrics(metrics: Dict[str, float]):
    """
    Display metrics in the Streamlit interface
    """
    if metrics:
        st.sidebar.subheader("Query Metrics")
        st.sidebar.markdown(f"""
        - Relevance Score: {metrics['relevance_score']:.2f}
        - Confidence Score: {metrics['confidence_score']:.2f}
        - Processing Time: {metrics['processing_time']*1000:.2f}ms
        """)

def main():
    try:
        st.set_page_config(page_title="Enhanced Document QA System", layout="wide")
        
        st.title("Enhanced Document QA System with Metrics 📊")
        
        # Sidebar for document processing
        with st.sidebar:
            st.header("Document Processing")
            
            uploaded_files = st.file_uploader(
                "Upload documents",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'xlsx', 'docx', 'txt']
            )
            
            if st.button("Process Documents"):
                if uploaded_files:
                    with st.spinner("Processing documents..."):
                        if process_documents(uploaded_files):
                            st.session_state.docs_processed = True
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Error processing documents.")
                else:
                    st.warning("Please upload documents first.")
        
        # Main chat interface
        if st.session_state.docs_processed:
            user_question = st.chat_input("Ask a question about your documents")
            if user_question:
                response, metrics = handle_query_with_metrics(user_question)
                
                # Display response
                st.chat_message("user").write(user_question)
                st.chat_message("assistant").write(response)
                
                # Display metrics
                display_metrics(metrics)
        else:
            st.info("Please upload and process documents to start asking questions.")
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        st.error("An error occurred. Please check the logs.")

if __name__ == "__main__":
    main()