import pdfplumber
import fitz  # PyMuPDF for better text extraction from free-flowing PDFs
import pandas as pd
import csv
from io import StringIO
#from langchain_community.document_loaders import Document


def extract_text_pymupdf(pdf_path):
    """
    Use PyMuPDF (fitz) to extract text from the PDF.
    Handles free-flowing text better than pdfplumber.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract plain text
    return text

def get_pdf_text(pdf_docs):
    """
    Main function to extract text from PDFs using both pdfplumber and PyMuPDF.
    Extracts tables with pdfplumber and general text with PyMuPDF.
    """
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        # Use pdfplumber to extract tables
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                tables = page.extract_tables()
                for table in tables:
                    table_text = "Table:\n"
                    for row in table:
                        filtered_row = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                        if filtered_row:
                            table_text += " | ".join(filtered_row) + "\n"
                    text += table_text + "\n"
                    documents.append(Document(page_content=table_text, metadata={'source': 'table'}))
        
        # Use PyMuPDF for better text extraction
        non_table_text = extract_text_pymupdf(pdf)
        text += non_table_text  # Add non-tabular text to the output

    return text, documents

def get_csv_text(csv_file):
    """
    Extract text from a CSV file.
    """
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
    """
    Extract text from Excel files.
    """
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
