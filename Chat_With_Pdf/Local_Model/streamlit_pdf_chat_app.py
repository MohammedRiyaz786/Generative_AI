
# import streamlit as st
# from utils import *
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import  pdfplumber

# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         with pdfplumber.open(pdf) as pdf_reader:
# #             for page in pdf_reader.pages:
# #                 text += page.extract_text() or ""
# #                 tables = page.extract_tables()
# #                 for table in tables:
# #                     for row in table:
# #                         # Filter out None values and join the rest into a string
# #                         filtered_row = [cell for cell in row if cell is not None]
# #                         if filtered_row:  # Check if the filtered row is not empty
# #                             text += " | ".join(filtered_row) + "\n"  # Join table rows into text
# #     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     # print(chunks)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     print(vector_store)
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
#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         # Printing Chunk Length
#         # retrieved_docs = response.get('source_documents', [])
#         retrieved_docs = r.invoke(user_question)
#         print("Number of retrieved chunks: ", len(retrieved_docs),"\n\n",retrieved_docs)
        
#         # for idx, doc in enumerate(retrieved_docs):
#         #     print(f"Chunk {idx + 1} (Length: {len(doc.page_content)}):")
#         #     print(doc.page_content)

        
#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         st.write("Reply: The answer is not available in the context.")

# def main():
#     st.set_page_config(page_title="Chat PDF")
#     st.header("Chat with PDF using LLAMA3🦙 ")

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button",
#             accept_multiple_files=True
#         )
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)

#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Done")
#             else:
#                 st.warning("Please upload PDF files before processing.")

#     user_question = st.text_input("Ask a Question from the PDF Files")
#     if user_question:
#         handle_user_input(user_question)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# from utils import *
# from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import pdfplumber
# import os

# # Helper function to handle multiple file types
# def extract_file_content(uploaded_file):
#     file_type = uploaded_file.name.split('.')[-1].lower()
    
#     if file_type == 'pdf':
#         return get_pdf_text([uploaded_file])
#     elif file_type == 'csv':
#         return get_csv_text(uploaded_file)
#     elif file_type == 'xls' or file_type == 'xlsx':
#         return get_excel_text([uploaded_file])
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
#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         retrieved_docs = r.invoke(user_question)
#         print("Number of retrieved chunks: ", len(retrieved_docs), "\n\n", retrieved_docs)
        
#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
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


# Changes -3

# import streamlit as st 
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

    # Instructions:
    # 1. Only use information from the provided context
    # 2. If the exact answer is in the context, provide it
    # 3. If the answer is not in the context, say "The answer is not available in the context"
    # 4. Keep the answer concise and relevant

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
#         qa_chain, r = create_qa_chain()
        
#         response = qa_chain({"query": user_question})
#         retrieved_docs = r.invoke(user_question)
#         print("Number of retrieved chunks: ", len(retrieved_docs), "\n\n", retrieved_docs)
        
#         answer = response.get('result', '').strip()
#         if not answer:
#             answer = "The answer is not available in the context."
        
#         st.write("Reply: ", answer)
        
#     except Exception as e:
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


import logging
import streamlit as st
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

# Setup Logger
def setup_logger():
    logger = logging.getLogger("DocumentQA")
    logger.setLevel(logging.INFO)
    
    # Create handlers for logging to file and console
    file_handler = logging.FileHandler("logs.txt")
    console_handler = logging.StreamHandler()
    
    # Set level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize the logger
logger = setup_logger()

# Helper function to handle multiple file types
def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return get_pdf_text([uploaded_file])
    elif file_type == 'csv':
        return get_csv_text(uploaded_file)  
    elif file_type in ['xls', 'xlsx']:
        text, _ = get_excel_text([uploaded_file])
        return text  
    else:
        st.error("Unsupported file format. Please upload PDF, CSV, or Excel files.")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def create_qa_chain():
    prompt_template = """
    Use the following pieces of context to answer the question. If you cannot find the answer in the context, respond with "The answer is not available in the context." Do not make up or infer any information that is not explicitly stated in the context.

    Context: {context}

    Question: {question}

    Instructions:
    1. Only use information from the provided context
    2. If the exact answer is in the context, provide it
    3. If the answer is not in the context, say "The answer is not available in the context"
    4. Keep the answer concise and relevant

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
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=True,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
        },
        return_source_documents=True
    )
    
    return qa_chain, retriever

def handle_user_input(user_question):
    try:
        qa_chain, r = create_qa_chain()
        a= "------"*20      
        # Log the user question
        logger.info("\n\n\n\n",a)
        logger.info(f"User question: {user_question}")
        
        #changes made
        # Retrieve documents based on the user question
        retrieved_docs = r.get_relevant_documents(user_question)

        # Print the retrieved chunks in the backend console
        print(f"Number of retrieved chunks: {len(retrieved_docs)}")
        for idx, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {idx + 1} ---")
            print(f"Length: {len(doc.page_content)}")
            print(f"Content:\n{doc.page_content}")
            print("\n---------------------")
        # till here
        # Perform QA
        response = qa_chain({"query": user_question})
        
        # Retrieve documents
        retrieved_docs = r.invoke(user_question)
        logger.info(f"\nRetrieved Documents: {retrieved_docs}\n")
        
        # Extract the answer
        answer = response.get('result', '').strip()
        if not answer:
            answer = "The answer is not available in the context."
        
        # Log the final answer
        logger.info(f"Answer: {answer}\n")
        logger.info(a)
        
        # Display the answer in Streamlit
        st.write("Reply: ", answer)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: The answer is not available in the context.")
        logger.error(f"An error occurred: {str(e)}")

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
                    raw_text = ""
                    for uploaded_file in uploaded_files:
                        raw_text += extract_file_content(uploaded_file)

                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload files before processing.")

    user_question = st.text_input("Ask a Question from the Uploaded Files")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
