import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for more precise matching
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
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

    model = Ollama(model="llama2", temperature=0.1)  # Lower temperature for more focused answers
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Increase the number of retrieved documents and lower the distance threshold
        docs = new_db.similarity_search(
            user_question,
            k=5,  # Retrieve more documents
            fetch_k=20  # Consider more candidates
        )
        
        # Debug information (optional - comment out in production)
        st.write("Debug - Number of relevant documents found:", len(docs))
        
        if not docs:
            st.write("Reply: The answer is not available in the context.")
            return
        
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )
        
        output_text = response.get("output_text", "").strip()
        st.write("Reply: ", output_text)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Reply: The answer is not available in the context.")

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using LLAMA2🦙 ")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()