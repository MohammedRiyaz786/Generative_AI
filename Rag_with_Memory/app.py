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


logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

def create_qa_chain():
    prompt_template = """You are a direct and efficient AI assistant.

    IF the user's message matches ANY of these patterns:
    - "Hi", "Hello", "Hey", "Hii", "Hola" (just greeting)
    - "My name is [any name]"
    - "I am [any name]"
    - "[any greeting] my name is [any name]"
    - "[any greeting] I am [any name]"
    THEN respond only with: "Hello! How can I help you today?"

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
    Response should be only: "I don't have enough information to answer this question."

    4. Never start responses with:
    - "Based on..."
    - "According to..."
    - "I understand..."
    - "Let me..."

    5. Never end responses with:
    - "Is there anything else..."
    - "Let me know if..."
    - "Feel free to..."

    Question: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def handle_user_input(user_question: str):
    """Handle user input with enhanced error handling, logging, and better conversation handling"""
    try:
        logging.info(f"Processing user question: {user_question}")
        
        # Handle basic greeting and conversation patterns
        if re.match(r"^(?i)(hi|hello|hey)$", user_question):
            response = "Hello! How can I help you today?"
            
        # Handle name introduction
        elif name_match := re.match(r"(?i)my name is (\w+)", user_question):
            name = name_match.group(1)
            st.session_state.user_context['name'] = name
            response = f"Hello {name}! How can I help you today?"
            # Skip the QA chain for conversational inputs
            
        # Handle name question
        elif re.match(r"(?i)what('s| is) my name\??", user_question):
            name = st.session_state.user_context.get('name')
            response = f"Your name is {name}." if name else "You haven't told me your name yet."
            # Skip the QA chain for conversational inputs
            
        # Handle document-based questions
        else:
            if not st.session_state.docs_processed:
                response = "Please upload and process some documents first."
            else:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    response = "I'm having trouble accessing the document knowledge base. Please make sure documents are processed."
                else:
                    logging.debug("Calling QA chain with question: %s", user_question)
                    try:
                        # Only use QA chain for actual questions, not conversational inputs
                        if not is_conversational_input(user_question):
                            result = qa_chain({"question": user_question})
                            logging.debug("QA chain result: %s", result)
                            response = result.get('answer', '').strip()
                            if not response:
                                response = "I don't have enough information to answer this question."
                        else:
                            response = handle_conversation(user_question)
                    except Exception as chain_error:
                        logging.error("Error in QA chain execution: %s", str(chain_error), exc_info=True)
                        raise
        
        # Update conversation
        st.session_state.conversation.append({
            "user": user_question,
            "assistant": response
        })
        
        # Display conversation
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

def main():
    st.set_page_config(page_title="Chat with Documents and Images")
    st.header("Chat with Documents and Images using LLAMA3🦙")

    
    if st.sidebar.button("Clear Conversation"):
        st.session_state.memory.clear()
        if 'conversation' in st.session_state:
            st.session_state.conversation = []
        st.success("Conversation history cleared!")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, CSV, Excel, PowerPoint, Word) or images (PNG, JPG, JPEG, GIF, BMP, TIFF)",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        )

        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    print("Creating chunks!\n")
                    all_text_chunks = []
                    all_metadata_chunks = []

                    for uploaded_file in uploaded_files:
                        try:
                            raw_text, docs = extract_file_content(uploaded_file)
                            
                            if raw_text.strip() and docs:
                                for doc in docs:
                                    text_chunks, metadata_chunks = get_text_chunks(doc.page_content, doc.metadata)
                                    all_text_chunks.extend(text_chunks)
                                    all_metadata_chunks.extend(metadata_chunks)
                            else:
                                st.warning(f"No content could be extracted from {uploaded_file.name}")
                                
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            logging.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue

                    if all_text_chunks:
                        get_vector_store(all_text_chunks, all_metadata_chunks)
                        print("Chunking Done!\n")
                        st.success(" File processed successfully!")
                    else:
                        st.error("No content could be extracted from any of the uploaded files.")
            else:
                st.warning("Please upload files before processing.")


    user_question = st.chat_input("Ask a question about your documents")

    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()