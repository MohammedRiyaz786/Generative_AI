import streamlit as st
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
import re
from typing import Dict, List
import logging

logging.basicConfig(filename='app_log_new.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        human_prefix="Human",
        ai_prefix="Assistant"
    )

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'user_context' not in st.session_state:
    st.session_state.user_context = {}

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

def update_user_context(question: str) -> None:
    """Update user context based on interactions"""
    name_patterns = [
        (r"(?i)my name is (\w+)", "name"),
        (r"(?i)i am (\w+)", "name"),
        (r"(?i)i'm (\w+)", "name"),
        (r"(?i)this is (\w+)", "name")
    ]
    
    for pattern, key in name_patterns:
        match = re.search(pattern, question)
        if match:
            st.session_state.user_context[key] = match.group(1)
            return

def get_context_aware_answer(question: str) -> str:
    """Generate context-aware answers based on user context"""
    name_questions = [
        r"(?i)what( is|'s) my name\??",
        r"(?i)who am i\??",
        r"(?i)do you (know|remember) (my name|who i am)\??"
    ]
    
    if any(re.match(pattern, question) for pattern in name_questions):
        if 'name' in st.session_state.user_context:
            return f"Your name is {st.session_state.user_context['name']}."
    return None

def create_qa_chain():
    prompt_template = """You are a direct and efficient AI assistant with a friendly personality. Use the following rules to process the question:

    Current User Context: {{"user_context": {user_context}}}

    IF the question matches ANY greeting patterns:
    - Basic greetings: "Hi", "Hello", "Hey", etc.
    THEN respond with: "Hello! How can I help you today?"

    ELSE IF the question is about user information (like their name):
    - Use the user context to provide a personalized response

    ELSE:
    1. Use ONLY the provided information:
    Context: {context}
    Chat History: {chat_history}
    Current Question: {question}

    2. Your response must be:
    - Direct and to-the-point
    - Based only on given context and history
    - Natural and conversational
    - Without mentioning sources or references
    
    3. If the answer cannot be found in context or history:
    Respond with: "I don't have enough information to answer this question."

    4. NEVER start responses with:
    - "Based on..."
    - "According to..."
    - "I understand..."
    - "Let me..."

    5. NEVER end responses with:
    - "Is there anything else..."
    - "Let me know if..."
    - "Feel free to..."

    Question: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history", "user_context"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={
                "prompt": PROMPT,
                "additional_kwargs": {"user_context": str(st.session_state.user_context)}
            },
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {str(e)}")
        return None

def handle_user_input(user_question):
    try:
        logging.info(f"User question: {user_question}")
        
        # Check for greeting patterns
        greeting_patterns = [r'^(?i)(hi|hello|hey|hii|hola)$']
        name_intro_patterns = [
            r'^(?i)(my name is|i am|i\'m)\s+\w+$',
            r'^(?i)(hi|hello|hey|hii|hola),?\s*(my name is|i am|i\'m)\s+\w+$'
        ]
        
        if any(re.match(pattern, user_question) for pattern in greeting_patterns):
            answer = "Hello! How can I help you today?"
        elif any(re.match(pattern, user_question) for pattern in name_intro_patterns):
            update_user_context(user_question)
            answer = f"Hello {st.session_state.user_context.get('name', '')}! How can I help you today?"
        else:
            # Check for context-aware answers first
            context_answer = get_context_aware_answer(user_question)
            if context_answer:
                answer = context_answer
            else:
                # Use RAG for other questions
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    answer = "I'm having trouble accessing the document knowledge base. Please make sure documents are properly processed."
                else:
                    response = qa_chain({"question": user_question})
                    answer = response.get('answer', '').strip()
                    
                    if not answer or answer == response.get('chat_history', ''):
                        answer = "I don't have enough information to answer this question."
        
        # Update conversation state
        st.session_state.conversation.append({
            "user": user_question,
            "assistant": answer
        })
        
        # Display conversation with clear user/assistant distinction
        for message in st.session_state.conversation:
            with st.chat_message("user", avatar="🧑"):
                st.write(message["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["assistant"])
                
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: I'm sorry, but I encountered an error while processing your question.")

def main():
    st.set_page_config(
        page_title="Chat with Documents and Images",
        page_icon="🦙",
        layout="wide"
    )
    
    st.header("Chat with Documents and Images using LLAMA3🦙")

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents or images",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        )
        
        # Process button
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    all_text_chunks = []
                    all_metadata_chunks = []

                    for uploaded_file in uploaded_files:
                        try:
                            raw_text, docs = extract_file_content(uploaded_file)
                            if raw_text.strip() and docs:
                                for doc in docs:
                                    text_chunks, metadata_chunks = get_text_chunks(
                                        doc.page_content, 
                                        doc.metadata
                                    )
                                    all_text_chunks.extend(text_chunks)
                                    all_metadata_chunks.extend(metadata_chunks)
                            else:
                                st.warning(f"No content extracted from {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue

                    if all_text_chunks:
                        get_vector_store(all_text_chunks, all_metadata_chunks)
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No content could be extracted from the uploaded files.")
            else:
                st.warning("Please upload files before processing.")
        
        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.memory.clear()
                st.session_state.conversation = []
                st.success("Chat cleared!")
                
        with col2:
            if st.button("Clear Context"):
                st.session_state.user_context = {}
                st.success("Context cleared!")
        

    # Main chat interface
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()