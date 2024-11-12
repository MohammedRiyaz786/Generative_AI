import streamlit as st
import requests
import logging
from utils import init_session_state, handle_user_input, process_documents

def main():
    try:
        st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
        init_session_state()
        
        st.header("Chat with Documents and Images using LLAMA3 🦙")
        
        # Sidebar
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
                        # Call API to process documents
                        files = [('files', file) for file in uploaded_files]
                        response = requests.post('http://localhost:8000/process-documents', files=files)
                        if response.status_code == 200:
                            st.session_state.docs_processed = True
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Error processing documents. Check logs for details.")
                else:
                    st.warning("Please upload files before processing.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Conversation"):
                    requests.post('http://localhost:8000/clear-conversation')
                    st.session_state.conversation = []
                    st.success("Conversation cleared!")
            
            with col2:
                if st.button("Clear Everything"):
                    requests.post('http://localhost:8000/clear-everything')
                    st.session_state.conversation = []
                    st.session_state.docs_processed = False
                    st.success("Everything has been cleared!")
            
            st.write("Status:")
            st.write(f"Documents Processed: {'✅' if st.session_state.docs_processed else '❌'}")

        # Chat interface
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            handle_user_input(user_question)
            
    except Exception as e:
        logging.error(f"Error in handle_user_input: {str(e)}", exc_info=True)
        st.error(f"An error occurred while processing your question: {str(e)}")


if __name__ == "__main__":
    main()