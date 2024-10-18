import streamlit as st
import logging
from utils import get_pdf_text, get_csv_text, get_excel_text, get_non_table_pdf_text, get_ppt_text, get_word_text
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Setup logger
logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        # Use both tabular and non-tabular extraction methods
        tabular_text, tabular_docs = get_pdf_text([uploaded_file])
        non_tabular_text, non_tabular_docs = get_non_table_pdf_text([uploaded_file])
        return tabular_text + non_tabular_text, tabular_docs + non_tabular_docs
    elif file_type == 'csv':
        text = get_csv_text(uploaded_file)
        return text, [Document(page_content=text, metadata={'source': 'csv'})]
    elif file_type in ['xls', 'xlsx']:
        text, docs = get_excel_text([uploaded_file])
        return text, docs
    elif file_type in ['ppt', 'pptx']:
        text, docs = get_ppt_text([uploaded_file])
        return text, docs
    elif file_type in ['doc', 'docx']:
        text, docs = get_word_text([uploaded_file])
        return text, docs
    else:
        st.error("Unsupported file format. Please upload PDF, CSV, Excel, PowerPoint, or Word files.")
        return "", []

def get_text_chunks(text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata for _ in chunks]
    return chunks, metadata_chunks

def get_vector_store(text_chunks, metadata_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata_chunks)
    vector_store.save_local("faiss_index")

def create_qa_chain():
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the given context. The context may contain various types of information, such as academic text, bullet points, slides from presentations, tables, or data extracted from documents.

    Context: {context}

    Question: {question}

    Instructions:
    1. Carefully analyze the context and the question.
    2. If the context contains **tables** or structured data, identify and use the information appropriately in your answer. Mention the table data if relevant.
    3. If the context contains **slides** or bullet points (e.g., from a PowerPoint presentation), maintain their structure when presenting information.
    4. If the answer is clearly stated in the context, provide it directly.
    5. If the answer requires synthesizing information from multiple parts of the context, do so carefully.
    6. If the context contains relevant tabular data, incorporate it into your answer appropriately.
    7. If the answer is not contained within the context, or if you're unsure, say "I don't have enough information to answer that question accurately."
    8. Provide your answer in a clear and concise manner.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = Ollama(model="llama3.1", temperature=0.3)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 25}
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
        logging.info(f"User question: {user_question}")

        qa_chain, r = create_qa_chain()
        
        response = qa_chain({"query": user_question})
        retrieved_docs = r.invoke(user_question)

        logging.info(f"Number of retrieved chunks: {len(retrieved_docs)}")
        
        print(f"\nRetrieved chunks for question: '{user_question}'\n")
        for i, doc in enumerate(retrieved_docs):
            print(f"Chunk {i + 1}:")
            for line in doc.page_content.split('\n'):
                print(line.strip())
            print("-" * 50)

        answer = response.get('result', '').strip()
        if not answer:
            answer = "I don't have enough information to answer that question."
        
        st.write("Reply: ", answer)
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: I'm sorry, but I encountered an error while processing your question.")

def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("Chat with Documents using LLAMA3🦙 ")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, CSV, Excel, PowerPoint, or Word files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    all_text_chunks = []
                    all_metadata_chunks = []

                    for uploaded_file in uploaded_files:
                        raw_text, docs = extract_file_content(uploaded_file)
                        
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
