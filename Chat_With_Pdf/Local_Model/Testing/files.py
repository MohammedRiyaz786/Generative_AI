import streamlit as st
import os
static_dir = os.path.join(os.getcwd(), 'static')
import logging
from utils1 import get_pdf_text, get_csv_text, get_excel_text
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Document


#creatin Directory
if not os.path.exists('static'):
    os.makedirs('static')

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

def get_text_chunks(text, metadata):
    """
    Split text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Create corresponding metadata for each chunk
    metadata_chunks = [metadata for _ in chunks]
    
    return chunks, metadata_chunks

def get_vector_store(text_chunks, metadata_chunks):
    """
    Create FAISS vector store from the given text chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata_chunks)
    vector_store.save_local("faiss_index")

def create_qa_chain():
    """
    Create a QA chain using the retrieved documents from FAISS.
    """
    prompt_template = """
    Use the following pieces of context to answer the question. 
    Provide only the exact information relevant to the question without any additional details or explanation.

    Context: {context}

    Question: {question}

    Instructions:
    1. Only use information from the provided context.
    2. Provide a concise and direct answer, without any extra information or explanation.
    3. Do not include any details about table structure or unrelated content.

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
    """
    Handle user's input and return answers using the QA chain.
    """
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
            for line in doc.page_content.split('\n'):
                print(line.strip())
            print("-" * 50)

        answer = response.get('result', '').strip()
        if not answer:
            answer = "The answer is not available in the context."
        
        st.write("Reply: ", answer)
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: The answer is not available in the context.")

def main():
    """
    Main function to run the Streamlit app.
    """
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

                    # Process each document individually and sync text/metadata
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
