# Function to extract text from PDFs and create document objects
import pdfplumber
import csv
import PyPDF2
import pandas as pd
from langchain_core.documents import Document
from io import StringIO
from PyPDF2 import PdfReader
i  

def get_non_table_pdf_text(pdf_docs):
    """Extract text from non-table PDFs."""
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        with open(pdf, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                # Extracting page text
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                documents.append(Document(page_content=page_text, metadata={'source': 'pdf'}))
                
    return text, documents

def get_pdf_text(pdf_docs):
    """Extract text from PDFs, including tables."""
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                # Extracting page text
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                text += page_text + "\n"
                
                # Extracting tables
                tables = page.extract_tables()
                for table in tables:
                    table_text = "Table:\n"
                    for row in table:
                        filtered_row = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                        if filtered_row:
                            table_text += " | ".join(filtered_row) + "\n"
                    text += table_text + "\n"
                    documents.append(Document(page_content=table_text, metadata={'source': 'table'}))
                    
    return text, documents


def get_csv_text(csv_file):
    text = ""
    csv_file.seek(0)
    content = csv_file.read().decode('utf-8')
    csv_reader = csv.reader(StringIO(content))
    for row in csv_reader:
        filtered_row = [cell.strip() for cell in row if cell.strip()]
        if filtered_row:
            text += " | ".join(filtered_row) + "\n"
    return text

def get_excel_text(excel_files):
    text = ""
    documents = []
    for excel_file in excel_files:
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        for sheet_name, sheet_data in excel_data.items():
            sheet_text = f"Sheet: {sheet_name}\n"
            for _, row in sheet_data.iterrows():
                filtered_row = [str(cell).strip() for cell in row if pd.notna(cell) and str(cell).strip()]
                if filtered_row:
                    sheet_text += " | ".join(filtered_row) + "\n"
            text += sheet_text + "\n"
            documents.append(Document(page_content=sheet_text, metadata={'source': f"Sheet {sheet_name}"}))
    return text, documents
