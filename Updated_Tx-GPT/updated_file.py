import streamlit as st
import logging
import os
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
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    filename='app2.log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize session states
def init_session_state():
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
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

def extract_file_content(uploaded_file) -> Tuple[str, List[Document]]:
    """Extract content from uploaded files"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    try:
        if file_type in image_extensions:
            return process_image(uploaded_file)
        elif file_type == 'pdf':
            tabular_text, tabular_docs = get_pdf_text([uploaded_file])
            non_tabular_text, non_tabular_docs = get_non_table_pdf_text([uploaded_file])
            return tabular_text + non_tabular_text, tabular_docs + non_tabular_docs
        elif file_type == 'csv':
            text = get_csv_text(uploaded_file)
            return text, [Document(page_content=text, metadata={'source': uploaded_file.name})]
        elif file_type in ['xls', 'xlsx']:
            return get_excel_text([uploaded_file])
        elif file_type in ['pptx']:
            return get_ppt_text([uploaded_file])
        elif file_type in ['doc', 'docx']:
            return get_word_text([uploaded_file])
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
    except Exception as e:
        logging.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        raise

def process_documents(uploaded_files) -> bool:
    """Process uploaded documents and create vector store"""
    try:
        all_text_chunks = []
        all_metadata_chunks = []
        
        for uploaded_file in uploaded_files:
            raw_text, docs = extract_file_content(uploaded_file)
            if raw_text.strip() and docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                for doc in docs:
                    chunks = text_splitter.split_text(doc.page_content)
                    metadata = [{'source': uploaded_file.name} for _ in chunks]
                    all_text_chunks.extend(chunks)
                    all_metadata_chunks.extend(metadata)
        
        if all_text_chunks:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings.client.to(device)
            
            vector_store = FAISS.from_texts(
                texts=all_text_chunks,
                embedding=embeddings,
                metadatas=all_metadata_chunks
            )
            vector_store.save_local("faiss_index")
            st.session_state.vector_store = vector_store
            return True
        return False
    except Exception as e:
        logging.error(f"Error in process_documents: {str(e)}")
        return False

def create_qa_chain() -> Optional[ConversationalRetrievalChain]:
    """Create the QA chain with the vector store"""
    try:
        if not os.path.exists("faiss_index"):
            return None
            
        prompt_template = """
        Context: {context}
        Chat History: {chat_history}
        User Context: {user_context}
        Current Question: {question}

        Instructions:
        1. If the question is a greeting or name introduction, respond naturally
        2. If asking about user information (like name), use the user context
        3. For other questions, use only the provided context and chat history
        4. If information isn't available, say "I don't have enough information to answer this question."
        5. Be concise and direct in your response
        6. Don't mention sources or references

        Question: {question}
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question", "user_context"]
        )

        llm = Ollama(model="llama3.1", temperature=0.1)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={
                "prompt": PROMPT,
                "additional_kwargs": {"user_context": str(st.session_state.user_context)}
            },
            return_source_documents=True
        )
    except Exception as e:
        logging.error(f"Error in create_qa_chain: {str(e)}")
        return None

def handle_user_input(user_question: str):
    """Handle user input and generate responses"""
    try:
        # Update user context for name-related inputs
        name_match = re.match(r"(?i)my name is (\w+)", user_question)
        if name_match:
            st.session_state.user_context['name'] = name_match.group(1)
            response = f"Hello {name_match.group(1)}! How can I help you today?"
        
        # Handle name questions
        elif re.match(r"(?i)what('s| is) my name\??", user_question):
            response = f"Your name is {st.session_state.user_context.get('name', 'not mentioned yet')}."
        
        # Handle greetings
        elif re.match(r"(?i)^(hi|hello|hey)$", user_question):
            response = "Hello! How can I help you today?"
        
        # Handle other questions using RAG
        else:
            qa_chain = create_qa_chain()
            if qa_chain is None:
                response = "I'm having trouble accessing the document knowledge base. Please make sure documents are processed."
            else:
                result = qa_chain({"question": user_question})
                response = result.get('answer', '').strip()
                if not response:
                    response = "I don't have enough information to answer this question."
        
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
        logging.error(f"Error in handle_user_input: {str(e)}")
        st.error("An error occurred while processing your question.")

def main():
    st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
    init_session_state()
    
    st.header("Chat with Documents and Images using LLAMA3 🦙")
    
    with st.sidebar:
        st.title("Document Processing")
        uploaded_files = st.file_uploader(
            "Upload documents or images",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    if process_documents(uploaded_files):
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No content could be extracted from the files.")
            else:
                st.warning("Please upload files before processing.")
        
        if st.button("Clear Conversation"):
            st.session_state.memory.clear()
            st.session_state.conversation = []
            st.success("Conversation cleared!")
    
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()