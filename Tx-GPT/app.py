# app.py
import os
import logging
import streamlit as st
from utils import*
from pymongo import MongoClient
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISSVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from utils import create_prompt
from api import api_app



logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


mongo_client = MongoClient('mongodb://localhost:27017')
db = mongo_client['rag_app']
vector_store_collection = db['vector_store']


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

PROMPT = create_prompt()
def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore.from_documents(
        documents=[],
        embedding=embeddings,
        client=vector_store_collection
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model="llama3.1", temperature=0.1),
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={
            "prompt": PROMPT
        },
        return_source_documents=True
    )
    return qa_chain

def main():
    st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
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
                    process_documents(uploaded_files)
            else:
                st.warning("Please upload files before processing.")

    # Chat interface
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_user_input(user_question)

    # API integration
    api_app(st)

def process_documents(uploaded_files):
    text_chunks = []
    metadata_chunks = []
    for uploaded_file in uploaded_files:
        text, docs = extract_file_content(uploaded_file)
        if text and docs:
            text_chunks.extend(docs[0].page_content.split('\n'))
            metadata_chunks.extend([docs[0].metadata] * len(docs[0].page_content.split('\n')))

    if text_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISSVectorStore.from_texts(
            text_chunks,
            embedding=embeddings,
            metadatas=metadata_chunks,
            client=vector_store_collection
        )
        st.session_state.docs_processed = True
        st.success("Documents processed successfully!")
    else:
        st.error("No text content could be extracted from the uploaded files.")

def handle_user_input(user_question):
    try:
        logging.info(f"Processing user question: {user_question}")
        if not st.session_state.docs_processed:
            st.warning("Please upload and process some documents first.")
            return

        qa_chain = create_qa_chain()
        result = qa_chain({"question": user_question})
        response = result.get('answer', '').strip()
        if not response:
            response = "I don't have enough information to answer this question."

        st.session_state.conversation.append({
            "user": user_question,
            "assistant": response
        })

        for message in st.session_state.conversation:
            with st.chat_message("user", avatar="🧑"):
                st.write(message["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["assistant"])

    except Exception as e:
        logging.error(f"Error in handle_user_input: {str(e)}", exc_info=True)
        st.error(f"An error occurred while processing your question: {str(e)}")

if __name__ == "__main__":
    main()