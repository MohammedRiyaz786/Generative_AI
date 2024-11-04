import streamlit as st
# Set page config as the first Streamlit command
st.set_page_config(page_title="Chat with Documents and Images", layout="wide")

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
import numpy as np

# Setup logger
logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Handle different file types
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
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the given context and chat history. 
    Provide a concise and point-to-point answer without mentioning sources or slides.

    Chat History: {chat_history}
    Context: {context}
    Question: {question}

    Instructions:
    1. Consider both the chat history and current context when formulating your answer.
    2. If referring to information from previous exchanges, be explicit about the connection.
    3. Provide a specific and concise answer to the question.
    4. If the context contains tables, structured data, or information from images, extract only the relevant information.
    5. Maintain the structure of bullet points or lists if present in the relevant information.
    6. Include mathematical formulas if relevant, using LaTeX notation.
    7. If you encounter text that appears to be from an image, interpret it in context.
    8. For tabular data from images, present it in a clear, structured format.
    9. If the answer is not in the context or chat history, respond: "I don't have enough information to answer this question."
    10. Do not mention sources, slide numbers, or any metadata in your answer.
    """

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

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        return_generated_question=True
    )
    
    return qa_chain

def handle_user_input(user_question):
    try:
        logging.info(f"User question: {user_question}")

        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = create_qa_chain()

        response = st.session_state.qa_chain({
            "question": user_question
        })

        answer = response.get('answer', '').strip()
        if not answer:
            answer = "I don't have enough information to answer this question."
        
        st.write("Reply: ", answer)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", answer))

        st.write("\nChat History:")
        for role, message in reversed(st.session_state.chat_history[:-2]):
            st.write(f"{role}: {message}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: I'm sorry, but I encountered an error while processing your question.")

def main():
    st.header("Chat with Documents and Images using LLAMA3🦙")

    # Add clear chat history button in sidebar
    if st.sidebar.button("Clear Chat History"):
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
        if 'qa_chain' in st.session_state:
            del st.session_state.qa_chain
        st.success("Chat history cleared!")

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
                        st.success("Documents and images processed successfully!")
                        
                        if 'qa_chain' in st.session_state:
                            del st.session_state.qa_chain
                    else:
                        st.error("No content could be extracted from any of the uploaded files.")
            else:
                st.warning("Please upload files before processing.")

    user_question = st.text_input("Ask a Question from the Uploaded Files", key="question_input")

    if st.button("Search") and user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()