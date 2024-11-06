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

# Configure logging with more detailed output
logging.basicConfig(
    filename='app_log2.txt',
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)

# Global variable to track if documents are processed
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

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

def extract_file_content(uploaded_file) -> Tuple[str, List[Document]]:
    """Extract content from uploaded files with enhanced error checking"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        logging.info(f"Processing file: {uploaded_file.name} of type: {file_type}")
        
        image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        
        if file_type in image_extensions:
            text, docs = process_image(uploaded_file)
            logging.info(f"Processed image with text length: {len(text) if text else 0}")
            return text, docs
            
        elif file_type == 'pdf':
            tabular_text, tabular_docs = get_pdf_text([uploaded_file])
            non_tabular_text, non_tabular_docs = get_non_table_pdf_text([uploaded_file])
            combined_text = (tabular_text or "") + (non_tabular_text or "")
            combined_docs = (tabular_docs or []) + (non_tabular_docs or [])
            logging.info(f"Processed PDF with text length: {len(combined_text)}")
            return combined_text, combined_docs
            
        elif file_type == 'csv':
            text = get_csv_text(uploaded_file)
            docs = [Document(page_content=text, metadata={'source': uploaded_file.name})]
            logging.info(f"Processed CSV with text length: {len(text)}")
            return text, docs
            
        elif file_type in ['xls', 'xlsx']:
            text, docs = get_excel_text([uploaded_file])
            logging.info(f"Processed Excel with text length: {len(text)}")
            return text, docs
            
        elif file_type in ['pptx']:
            text, docs = get_ppt_text([uploaded_file])
            logging.info(f"Processed PowerPoint with text length: {len(text)}")
            return text, docs
            
        elif file_type in ['doc', 'docx']:
            text, docs = get_word_text([uploaded_file])
            logging.info(f"Processed Word with text length: {len(text)}")
            return text, docs
            
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
            
    except Exception as e:
        logging.error(f"Error processing file {uploaded_file.name}: {str(e)}", exc_info=True)
        raise

def process_documents(uploaded_files) -> bool:
    """Process uploaded documents and create vector store with enhanced error checking"""
    try:
        if not uploaded_files:
            logging.warning("No files uploaded")
            return False
            
        logging.info(f"Starting to process {len(uploaded_files)} documents")
        all_text_chunks = []
        all_metadata_chunks = []
        
        for uploaded_file in uploaded_files:
            try:
                raw_text, docs = extract_file_content(uploaded_file)
                if raw_text and docs:
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
                        
                    logging.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks created")
                else:
                    logging.warning(f"No content extracted from {uploaded_file.name}")
                    
            except Exception as e:
                logging.error(f"Error processing {uploaded_file.name}: {str(e)}", exc_info=True)
                continue
                
        if not all_text_chunks:
            logging.warning("No text chunks created from any documents")
            return False
            
        logging.info(f"Creating vector store with {len(all_text_chunks)} chunks")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings.client.to(device)
        
        vector_store = FAISS.from_texts(
            texts=all_text_chunks,
            embedding=embeddings,
            metadatas=all_metadata_chunks
        )
        
        # Save vector store
        vector_store.save_local(st.session_state.vector_store_path)
        st.session_state.docs_processed = True
        logging.info("Vector store created and saved successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in process_documents: {str(e)}", exc_info=True)
        return False

def create_qa_chain() -> Optional[ConversationalRetrievalChain]:
    """Create the QA chain with enhanced error handling and logging"""
    try:
        if not os.path.exists(st.session_state.vector_store_path):
            logging.error("Vector store path does not exist")
            return None
            
        if not st.session_state.docs_processed:
            logging.error("Documents not processed yet")
            return None
            
        logging.info("Creating QA chain")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = FAISS.load_local(
            st.session_state.vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        
        template = """
        Context: {context}
        Chat History: {chat_history}
        User Context: {user_context}
        Question: {question}

        Instructions:
        1. Use the provided context to answer the question
        2. If the answer isn't in the context, say "I don't have enough information to answer this question"
        3. Be concise and direct
        4. Don't mention sources

        Answer the question: {question}
        """

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question", "user_context"]
        )

        llm = Ollama(model="llama3.1", temperature=0.1)

        def get_chat_history(inputs) -> str:
            try:
                chat_history = inputs.get("chat_history", [])
                buffer = ""
                for message in chat_history:
                    if isinstance(message, tuple) and len(message) == 2:
                        human, ai = message
                        buffer += f"Human: {human}\nAssistant: {ai}\n"
                logging.debug(f"Processed chat history: {buffer}")
                return buffer
            except Exception as e:
                logging.error(f"Error in get_chat_history: {str(e)}", exc_info=True)
                return ""

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            get_chat_history=get_chat_history,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True  # Enable verbose mode for debugging
        )
        
        def new_call(inputs, *args, **kwargs):
            try:
                logging.debug(f"Chain inputs before modification: {inputs}")
                inputs["user_context"] = str(st.session_state.user_context)
                logging.debug(f"Chain inputs after modification: {inputs}")
                return chain._call(inputs, *args, **kwargs)
            except Exception as e:
                logging.error(f"Error in chain call: {str(e)}", exc_info=True)
                raise

        chain.__call__ = new_call
        
        logging.info("QA chain created successfully")
        return chain
        
    except Exception as e:
        logging.error(f"Error in create_qa_chain: {str(e)}", exc_info=True)
        return None

def handle_user_input(user_question: str):
    """Handle user input with enhanced error handling and logging"""
    try:
        logging.info(f"Processing user question: {user_question}")
        
        # Handle basic greeting
        if re.match(r"^(?i)(hi|hello|hey)$", user_question):
            response = "Hello! How can I help you today?"
            
        # Handle name introduction
        elif name_match := re.match(r"(?i)my name is (\w+)", user_question):
            name = name_match.group(1)
            st.session_state.user_context['name'] = name
            response = f"Hello {name}! How can I help you today?"
            
        # Handle name question
        elif re.match(r"(?i)what('s| is) my name\??", user_question):
            name = st.session_state.user_context.get('name')
            response = f"Your name is {name}." if name else "You haven't told me your name yet."
            
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
                        result = qa_chain({"question": user_question})
                        logging.debug("QA chain result: %s", result)
                        response = result.get('answer', '').strip()
                        if not response:
                            response = "I don't have enough information to answer this question."
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
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Error processing documents. Check logs for details.")
                else:
                    st.warning("Please upload files before processing.")
            
            # Clear button
            if st.button("Clear Conversation"):
                st.session_state.memory.clear()
                st.session_state.conversation = []
                st.success("Conversation cleared!")
            
            # Status indicator
            st.write("Status:")
            st.write(f"Documents Processed: {'✅' if st.session_state.docs_processed else '❌'}")
        
        # Chat interface
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            handle_user_input(user_question)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        st.error("An error occurred in the application. Please check the logs.")

if __name__ == "__main__":
    main()