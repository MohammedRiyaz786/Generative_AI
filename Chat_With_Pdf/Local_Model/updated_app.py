# import streamlit as st
# import logging
# from utils import get_pdf_text, get_csv_text, get_excel_text
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import os

# # Setup logger
# logging.basicConfig(filename='app_log.txt', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Helper function to handle multiple file types
# def extract_file_content(uploaded_file):
#     file_type = uploaded_file.name.split('.')[-1].lower()
    
#     if file_type == 'pdf':
#         return get_pdf_text([uploaded_file])
#     elif file_type == 'csv':
#         return get_csv_text(uploaded_file)  # Correctly handle the uploaded CSV file
#     elif file_type in ['xls', 'xlsx']:
#         text,_= get_excel_text([uploaded_file])
#         return text  # Correctly handle the uploaded Excel file
#     else:
#         st.error("Unsupported file format. Please upload PDF, CSV, or Excel files.")
#         return ""

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def create_qa_chain():
#     prompt_template = """
#     Use the following pieces of context to answer the question. If you cannot find the answer in the context, respond with "The answer is not available in the context." Do not make up or infer any information that is not explicitly stated in the context.

#     Context: {context}

#     Question: {question}

#     Instructions:
#     1. Only use information from the provided context
#     2. If the exact answer is in the context, provide it
#     3. If the answer is not in the context, say "The answer is not available in the context"
#     4. Keep the answer concise and relevant

#     Answer:
#     """

#     PROMPT = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     llm = Ollama(model="llama3.1", temperature=0.3)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 15
#         }
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         verbose=True,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={
#             "prompt": PROMPT,
#         },
#         return_source_documents=True
#     )
    
#     return qa_chain, retriever

# def handle_user_input(user_question):
#     try:
#         # Log the user's question
#         logging.info(f"User question: {user_question}")

#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         retrieved_docs = r.invoke(user_question)

#         # Log the number of retrieved chunks
#         logging.info(f"Number of retrieved chunks: {len(retrieved_docs)}")
        

#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
#         # Log the exception
#         logging.error(f"Error: {str(e)}")
#         st.error(f"An error occurred: {str(e)}")
#         st.write("Reply: The answer is not available in the context.")

# def main():
#     st.set_page_config(page_title="Chat with Documents")
#     st.header("Chat with Documents using LLAMA3🦙 ")

#     with st.sidebar:
#         st.title("Menu:")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF, CSV, or Excel files and Click on the Submit & Process Button",
#             accept_multiple_files=True
#         )
#         if st.button("Submit & Process"):
#             if uploaded_files:
#                 with st.spinner("Processing..."):
#                     raw_text = ""
#                     for uploaded_file in uploaded_files:
#                         raw_text += extract_file_content(uploaded_file)

#                     if raw_text:
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Documents processed successfully!")
#             else:
#                 st.warning("Please upload files before processing.")

#     user_question = st.text_input("Ask a Question from the Uploaded Files")
#     if user_question:
#         handle_user_input(user_question)

# if __name__ == "__main__":
#     main()
#-------------------------------------------------changes-1--------------------------
# import streamlit as st
# import logging
# from utils import get_pdf_text, get_csv_text, get_excel_text
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import os

# # Setup logger
# logging.basicConfig(filename='app_log.txt', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Helper function to handle multiple file types
# def extract_file_content(uploaded_file):
#     file_type = uploaded_file.name.split('.')[-1].lower()
    
#     if file_type == 'pdf':
#         return get_pdf_text([uploaded_file])
#     elif file_type == 'csv':
#         return get_csv_text(uploaded_file)  # Correctly handle the uploaded CSV file
#     elif file_type in ['xls', 'xlsx']:
#         text,_= get_excel_text([uploaded_file])
#         return text  # Correctly handle the uploaded Excel file
#     else:
#         st.error("Unsupported file format. Please upload PDF, CSV, or Excel files.")
#         return ""

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def create_qa_chain():
#     prompt_template = """
#     Use the following pieces of context to answer the question. If you cannot find the answer in the context, respond with "The answer is not available in the context." Do not make up or infer any information that is not explicitly stated in the context.

#     Context: {context}

#     Question: {question}

#     Instructions:
#     1. Only use information from the provided context
#     2. If the exact answer is in the context, provide it
#     3. If the answer is not in the context, say "The answer is not available in the context"
#     4. Keep the answer concise and relevant

#     Answer:
#     """

#     PROMPT = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     llm = Ollama(model="llama3.1", temperature=0.3)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 15
#         }
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         verbose=True,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={
#             "prompt": PROMPT,
#         },
#         return_source_documents=True
#     )
    
#     return qa_chain, retriever

# def handle_user_input(user_question):
#     try:
#         # Log the user's question
#         logging.info(f"User question: {user_question}")

#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         retrieved_docs = r.invoke(user_question)

#         # Log the number of retrieved chunks
#         logging.info(f"Number of retrieved chunks: {len(retrieved_docs)}")
        
#         # Print chunks in the backend
#         print("Retrieved chunks:")
#         for i, doc in enumerate(retrieved_docs):
#             print(f"Chunk {i + 1}:")
#             print(doc.page_content)
#             print("-" * 50)

#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
#         # Log the exception
#         logging.error(f"Error: {str(e)}")
#         st.error(f"An error occurred: {str(e)}")
#         st.write("Reply: The answer is not available in the context.")

# def main():
#     st.set_page_config(page_title="Chat with Documents")
#     st.header("Chat with Documents using LLAMA3🦙 ")

#     with st.sidebar:
#         st.title("Menu:")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF, CSV, or Excel files and Click on the Submit & Process Button",
#             accept_multiple_files=True
#         )
#         if st.button("Submit & Process"):
#             if uploaded_files:
#                 with st.spinner("Processing..."):
#                     raw_text = ""
#                     for uploaded_file in uploaded_files:
#                         raw_text += extract_file_content(uploaded_file)

#                     if raw_text:
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Documents processed successfully!")
#             else:
#                 st.warning("Please upload files before processing.")

#     user_question = st.text_input("Ask a Question from the Uploaded Files")
#     if user_question:
#         handle_user_input(user_question)

# if __name__ == "__main__":
#     main()
#--------------------changes 2----------------------
# import streamlit as st
# import logging
# from utils import get_pdf_text, get_csv_text, get_excel_text
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import os

# # Setup logger
# logging.basicConfig(filename='app_log.txt', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Helper function to handle multiple file types
# def extract_file_content(uploaded_file):
#     file_type = uploaded_file.name.split('.')[-1].lower()
    
#     if file_type == 'pdf':
#         return get_pdf_text([uploaded_file])
#     elif file_type == 'csv':
#         return get_csv_text(uploaded_file)  # Correctly handle the uploaded CSV file
#     elif file_type in ['xls', 'xlsx']:
#         text,_= get_excel_text([uploaded_file])
#         return text  # Correctly handle the uploaded Excel file
#     else:
#         st.error("Unsupported file format. Please upload PDF, CSV, or Excel files.")
#         return ""

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def create_qa_chain():
#     prompt_template = """
#     Use the following pieces of context to answer the question. If you cannot find the answer in the context, respond with "The answer is not available in the context." Do not make up or infer any information that is not explicitly stated in the context.

#     Context: {context}

#     Question: {question}

#     Instructions:
#     1. Only use information from the provided context
#     2. If the exact answer is in the context, provide it
#     3. If the answer is not in the context, say "The answer is not available in the context"
#     4. Keep the answer concise and relevant

#     Answer:
#     """

#     PROMPT = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     llm = Ollama(model="llama3.1", temperature=0.3)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 15
#         }
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         verbose=True,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={
#             "prompt": PROMPT,
#         },
#         return_source_documents=True
#     )
    
#     return qa_chain, retriever

# def handle_user_input(user_question):
#     try:
#         # Log the user's question
#         logging.info(f"User question: {user_question}")

#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         retrieved_docs = r.invoke(user_question)

#         # Log the number of retrieved chunks
#         logging.info(f"Number of retrieved chunks: {len(retrieved_docs)}")
        
#         # Print chunks in the backend with improved readability
#         print(f"\nRetrieved chunks for question: '{user_question}'\n")
#         for i, doc in enumerate(retrieved_docs):
#             print(f"Chunk {i + 1}:")
#             # Split the content into lines and print each line
#             for line in doc.page_content.split('\n'):
#                 print(line.strip())
#             print("-" * 50)

#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
#         # Log the exception
#         logging.error(f"Error: {str(e)}")
#         st.error(f"An error occurred: {str(e)}")
#         st.write("Reply: The answer is not available in the context.")

# def main():
#     st.set_page_config(page_title="Chat with Documents")
#     st.header("Chat with Documents using LLAMA3🦙 ")

#     with st.sidebar:
#         st.title("Menu:")
#         uploaded_files = st.file_uploader(
#             "Upload your PDF, CSV, or Excel files and Click on the Submit & Process Button",
#             accept_multiple_files=True
#         )
#         if st.button("Submit & Process"):
#             if uploaded_files:
#                 with st.spinner("Processing..."):
#                     raw_text = ""
#                     for uploaded_file in uploaded_files:
#                         raw_text += extract_file_content(uploaded_file)

#                     if raw_text:
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Documents processed successfully!")
#             else:
#                 st.warning("Please upload files before processing.")

#     user_question = st.text_input("Ask a Question from the Uploaded Files",key="question_input")
#     #print(user_question)
#     # if user_question:
#     #     handle_user_input(user_question)
#     if st.button("Search") and user_question:
#         handle_user_input(user_question)

# if __name__ == "__main__":
#     main()
# ---------------------changes 3 ----------------

import streamlit as st
import logging
from utils import get_pdf_text, get_csv_text, get_excel_text
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Setup logger
logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to handle multiple file types
def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        return get_pdf_text([uploaded_file])
    elif file_type == 'csv':
        return get_csv_text(uploaded_file)  # Correctly handle the uploaded CSV file
    elif file_type in ['xls', 'xlsx']:
        text, _ = get_excel_text([uploaded_file])
        return text  # Correctly handle the uploaded Excel file
    else:
        st.error("Unsupported file format. Please upload PDF, CSV, or Excel files.")
        return ""

# **CHANGED: Added metadata parameter and return metadata_chunks in sync**
def get_text_chunks(text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # **CHANGED: Create corresponding metadata for each chunk**
    metadata_chunks = [metadata for _ in chunks]
    
    return chunks, metadata_chunks

# **CHANGED: Now passing both text_chunks and metadata_chunks**
def get_vector_store(text_chunks, metadata_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata_chunks)
    vector_store.save_local("faiss_index")

def create_qa_chain():
    prompt_template = """
    Use the following pieces of context to answer the question. If the answer lies within a table, ensure you interpret the table's data correctly.

    Context: {context}

    Question: {question}

    Instructions:
    1. Only use information from the provided context.
    2. If the exact answer is in the context, provide it.
    3. If the answer is not in the context, say "The answer is not available in the context."
    4. For tabular data, interpret the data accurately and include relevant rows/columns in the answer.

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = Ollama(model="llama3.1", temperature=0.3)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=True,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain, retriever

def handle_user_input(user_question):
    try:
        # Log the user's question
        logging.info(f"User question: {user_question}")

        qa_chain, r = create_qa_chain()
        
        response = qa_chain({"query": user_question})
        retrieved_docs = r.invoke(user_question)

        # Log the number of retrieved chunks
        logging.info(f"Number of retrieved chunks: {len(retrieved_docs)}")
        
        # Print chunks in the backend with improved readability
        print(f"\nRetrieved chunks for question: '{user_question}'\n")
        for i, doc in enumerate(retrieved_docs):
            print(f"Chunk {i + 1}:")
            # Split the content into lines and print each line
            for line in doc.page_content.split('\n'):
                print(line.strip())
            print("-" * 50)

        answer = response.get('result', '').strip()
        if not answer:
            answer = "The answer is not available in the context."
        
        st.write("Reply: ", answer)
        
    except Exception as e:
        # Log the exception
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: The answer is not available in the context.")

def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("Chat with Documents using LLAMA3🦙 ")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, CSV, or Excel files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    all_text_chunks = []
                    all_metadata_chunks = []

                    # **CHANGED: Process each document individually and sync text/metadata**
                    for uploaded_file in uploaded_files:
                        raw_text, docs = extract_file_content(uploaded_file)
                        
                        # Process each document individually to keep text and metadata in sync
                        for doc in docs:
                            text_chunks, metadata_chunks = get_text_chunks(doc.page_content, doc.metadata)
                            all_text_chunks.extend(text_chunks)
                            all_metadata_chunks.extend(metadata_chunks)

                    if all_text_chunks:
                        get_vector_store(all_text_chunks, all_metadata_chunks)
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload files before processing.")

    user_question = st.text_input("Ask a Question from the Uploaded Files", key="question_input")

    if st.button("Search") and user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()

