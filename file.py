import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import requests
import logging
from typing import List
from fastapi import UploadFile
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import pickle
import base64

# Your existing utility functions here (process_image, extract_file_content, etc.)
# Make sure to modify them to work with FastAPI's UploadFile

def init_session_state():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'docs_processed' not in st.session_state:
        st.session_state.docs_processed = False

def handle_user_input(user_question: str):
    try:
        # Call API to get response
        response = requests.post(
            'http://localhost:8000/query',
            json={"text": user_question}
        )
        
        if response.status_code == 200:
            answer = response.json()["answer"]
            
            if 'conversation' not in st.session_state:
                st.session_state.conversation = []
                
            st.session_state.conversation.append({
                "user": user_question,
                "assistant": answer
            })
            
            for message in st.session_state.conversation:
                with st.chat_message("user", avatar="🧑"):
                    st.write(message["user"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["assistant"])
        else:
            st.error("Error getting response from API")
            
    except Exception as e:
        logging.error(f"Error in handle_user_input: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

def create_qa_chain(vector_store):
    prompt_template = """You are a polite, respectful, and efficient AI assistant.
    [Your existing prompt template here]
    """

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
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain