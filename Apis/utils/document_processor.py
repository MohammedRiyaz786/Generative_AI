from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from typing import List, Tuple
import pdfplumber
import hashlib
import logging
import io

def create_document_hash(document: bytes, key: str) -> str:
    """Create a unique hash for document and key combination"""
    return hashlib.md5(f"{document}{key}".encode()).hexdigest()

def get_text_chunks(text: str, metadata: dict) -> Tuple[List[str], List[dict]]:
    """Split text into chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    metadata_chunks = [metadata.copy() for _ in chunks]
    return chunks, metadata_chunks

async def process_pdf_document(file_content: bytes) -> Tuple[str, List[Document], List[dict]]:
    """Process PDF content and return extracted text, documents, and metadata"""
    print("processing pdf")
    pdf_file = io.BytesIO(file_content)
    
    try:
        # Get tabular content
        tabular_text, tabular_docs = get_pdf_text([pdf_file])
        pdf_file.seek(0)  
        
        # Get non-tabular content
        non_tabular_text, non_tabular_docs = get_non_table_pdf_text([pdf_file])
        
        combined_text = tabular_text + non_tabular_text
        combined_docs = tabular_docs + non_tabular_docs
        print(type(combined_docs))
        print(type(combined_text))
        if not combined_text.strip():
            logging.warning("No text extracted from PDF")
            return "", [], []
        
        # Create metadata for each chunk
        metadata_chunks = []
        
        # Add metadata for tabular chunks
        for i, doc in enumerate(tabular_docs):
            metadata_chunks.append({
                "source_type": "table",
                "chunk_index": i,
                "content_type": "tabular",
                "overall_index": i,
                **doc.metadata  # Include original metadata
            })
        
        # Add metadata for non-tabular chunks
        for i, doc in enumerate(non_tabular_docs):
            metadata_chunks.append({
                "source_type": "text",
                "chunk_index": i,
                "content_type": "non-tabular",
                "overall_index": i + len(tabular_docs),
                **doc.metadata  # Include original metadata
            })
            
        return combined_text, combined_docs, metadata_chunks
        
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise
    print("process done")


def get_pdf_text(pdf_docs) -> Tuple[str, List[Document]]:
    """Extract text and tables from PDF documents."""
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page_num, page in enumerate(pdf_reader.pages):
                    # Extract text
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = "Table:\n"
                        for row in table:
                            filtered_row = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                            if filtered_row:
                                table_text += " | ".join(filtered_row) + "\n"
                        text += table_text + "\n"
                        documents.append(Document(
                            page_content=table_text,
                            metadata={'source': 'table', 'page': page_num + 1, 'filename': getattr(pdf, 'name', 'unknown')}
                        ))
                    
                    if page_text.strip():
                        documents.append(Document(
                            page_content=page_text,
                            metadata={'source': 'pdf_text', 'page': page_num + 1, 'filename': getattr(pdf, 'name', 'unknown')}
                        ))
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            raise
                    
    return text, documents

def get_non_table_pdf_text(pdf_docs) -> Tuple[str, List[Document]]:
    """Extract non-tabular text from PDF documents."""
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                # Extract and clean the text
                page_text = page.extract_text() or ""
                cleaned_text = ' '.join(page_text.split())  
                #print(cleaned_text)
                print(type(cleaned_text))
                cleaned_text = cleaned_text.replace('-\n', '') 

                
                metadata = {
                    'source': 'pdf_non_table',
                    'page': page_num + 1,
                    'filename': getattr(pdf, 'name', 'unknown')
                }
                
                # Only process if there's text content
                if cleaned_text.strip():
                    # Add to total text
                    text += cleaned_text + "\n"
                    
                    # Create chunks of the cleaned text
                    chunk_size = 500
                    overlap = 100
                    
                    # Create chunks with overlap
                    for i in range(0, len(cleaned_text), chunk_size - overlap):
                        chunk = cleaned_text[i:i + chunk_size]
                        if chunk.strip():  # Only create document if chunk has content
                            documents.append(Document(
                                page_content=chunk,
                                metadata=metadata.copy()
                            ))
                            
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            raise
                
    return text, documents