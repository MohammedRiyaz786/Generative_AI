
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BaseRetriever

# Load the webpage content
def load_content(url):
    loader = WebBaseLoader(web_paths=(url,))
    return loader.load()

# Split the text into smaller chunks
def split_content(content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(content)

# Create and initialize the retriever
def initialize_retriever(docs):
    embeddings = OllamaEmbeddings()
    vectore_store = FAISS.from_documents(docs, embeddings)
    return vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# App layout and logic
st.title("Simple RAG Search App")

# Input for webpage URL
url = st.text_input("Enter a URL to load content from:", 
                    "https://medium.com/@sahin.samia/what-is-retrieval-augmented-generation-rag-in-llm-and-how-it-works-a8c79e35a172")

# Load and process the content
if st.button("Load Content"):
    data = load_content(url)
    docs = split_content(data)
    retriever = initialize_retriever(docs)
    st.success("Content loaded and indexed!")

# Input for user's question
query = st.text_input("Ask a question:")

# Retrieve relevant documents based on the query
if query:
    if retriever:
        retrieved_docs = retriever.invoke(query)
        for i, doc in enumerate(retrieved_docs):
            st.write(f"Result {i + 1}:")
            st.write(doc.page_content)
    else:
        st.warning("Please load content first.")
