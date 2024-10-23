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
import torch

# Setup logger
logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_file_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        tabular_text, tabular_docs = get_pdf_text([uploaded_file])
        non_tabular_text, non_tabular_docs = get_non_table_pdf_text([uploaded_file])
        return tabular_text + non_tabular_text, tabular_docs + non_tabular_docs
    elif file_type == 'csv':
        text = get_csv_text(uploaded_file)
        return text, [Document(page_content=text, metadata={'source': 'csv'})]
    elif file_type in ['xls', 'xlsx']:
        text, docs = get_excel_text([uploaded_file])
        return text, docs
    elif file_type in ['pptx']:
        text, docs = get_ppt_text([uploaded_file])
        return text, docs
    elif file_type in ['doc', 'docx']:
        text, docs = get_word_text([uploaded_file])
        return text, docs
    else:
        st.error("Unsupported file format. Please upload PDF, CSV, Excel, PowerPoint, or Word files.")
        return "", []

def get_text_chunks(text, metadata):
    print("Started chunking!")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size for more granular retrieval
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata for _ in chunks]
    print("Chunking done!")
    return chunks, metadata_chunks

def get_vector_store(text_chunks, metadata_chunks):
    print("Storing chunks in Database!")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
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
        
        print(f"Processed batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}")

    vector_store.save_local("faiss_index")
    print("Stored chunks in Database!")
    return vector_store

def create_qa_chain():
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the given context. Provide a concise and point-to-point answer without mentioning sources or slides.

    Context: {context}

    Question: {question}

    Instructions:
    1. Analyze the context and question carefully.
    2. Provide a specific and concise answer to the question.
    3. If the context contains tables or structured data, extract only the relevant information.
    4. Maintain the structure of bullet points or lists if present in the relevant information.
    5. Include mathematical formulas if relevant, using LaTeX notation.
    6. Avoid unnecessary words or explanations. Stick to providing only the necessary information.
    7. If the answer is not in the context, respond: "I don't have enough information to answer this question."
    8. Do not mention sources, slide numbers, or any metadata in your answer.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = Ollama(model="llama3.1", temperature=0.1)  # Reduced temperature for more focused answers

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def handle_user_input(user_question):
    try:
        logging.info(f"User question: {user_question}")

        qa_chain = create_qa_chain()
        
        response = qa_chain({"query": user_question})

        answer = response.get('result', '').strip()
        if not answer:
            answer = "I don't have enough information to answer this question."
        
        st.write("Reply: ", answer)
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: I'm sorry, but I encountered an error while processing your question.")

def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("Chat with Documents using LLAMA3🦙")

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, CSV, Excel, PowerPoint (PPTX), or Word files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx']  # Specify pptx instead of ppt
)
        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    print("Creating chunks!\n")
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
                        print("Chunking Done!\n")
                        st.success("Documents processed successfully!")
            else:
                st.warning("Please upload files before processing.")

    user_question = st.text_input("Ask a Question from the Uploaded Files", key="question_input")

    if st.button("Search") and user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()