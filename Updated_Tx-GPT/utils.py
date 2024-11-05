import pdfplumber
import csv
import pandas as pd
from langchain_core.documents import Document
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument
from pptx.enum.shapes import MSO_SHAPE_TYPE
import re
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging

# Initialize OCR components
try:
    reader = easyocr.Reader(['en'])
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trocr_model.to(device)
except Exception as e:
    logging.error(f"Error initializing OCR components: {str(e)}")
    st.error("Error initializing OCR components. Some image processing features may be limited.")

def process_image(image_file):
    """
    Process image files and extract text using multiple OCR engines.
    """
    try:
        # Read image
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Initialize results list
        text_results = []
        
        # 1. Try Pytesseract
        try:
            pytesseract_text = pytesseract.image_to_string(gray)
            if pytesseract_text.strip():
                text_results.append(pytesseract_text)
        except Exception as e:
            logging.warning(f"Pytesseract OCR failed: {str(e)}")
            
        # 2. Try EasyOCR
        try:
            easy_result = reader.readtext(gray)
            easy_text = " ".join([t[1] for t in easy_result])
            if easy_text.strip():
                text_results.append(easy_text)
        except Exception as e:
            logging.warning(f"EasyOCR failed: {str(e)}")
            
        # 3. Try TrOCR for handwritten text
        try:
            pixel_values = trocr_processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = trocr_model.generate(pixel_values)
            trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if trocr_text.strip():
                text_results.append(trocr_text)
        except Exception as e:
            logging.warning(f"TrOCR failed: {str(e)}")
            
        # Extract tables
        table_text = extract_tables_from_image(gray)
        if table_text:
            text_results.append(table_text)
            
        # Extract formulas
        formula_text = extract_formulas_from_image(gray)
        if formula_text:
            text_results.append(formula_text)
            
        # Combine results
        combined_text = "\n".join(filter(None, text_results))
        
        if not combined_text.strip():
            raise Exception("No text could be extracted from the image")
            
        return combined_text, [Document(
            page_content=combined_text,
            metadata={'source': 'image', 'filename': image_file.name}
        )]
        
    except Exception as e:
        logging.error(f"Error processing image {image_file.name}: {str(e)}")
        raise

def extract_tables_from_image(gray_image):
    """
    Extract tables from images using OpenCV.
    """
    try:
        # Denoise image
        denoised = cv2.fastNlMeansDenoising(gray_image)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_text = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # Filter small contours
                roi = gray_image[y:y+h, x:x+w]
                
                # Try multiple OCR approaches for tables
                table_content = pytesseract.image_to_string(roi, config='--psm 6')
                if not table_content.strip():
                    easy_result = reader.readtext(roi)
                    table_content = " ".join([t[1] for t in easy_result])
                
                if table_content.strip():
                    table_text.append(f"Table Content:\n{table_content}")
        
        return "\n\n".join(table_text) if table_text else ""
        
    except Exception as e:
        logging.warning(f"Table extraction failed: {str(e)}")
        return ""

def extract_formulas_from_image(gray_image):
    """
    Extract mathematical formulas from images.
    """
    try:
        # Apply threshold to isolate formula
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        formula_text = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Filter small contours
                roi = gray_image[y:y+h, x:x+w]
                
                # Use specialized OCR settings for formulas
                formula_content = pytesseract.image_to_string(
                    roi,
                    config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/()={}[]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                )
                
                if formula_content.strip():
                    formula_text.append(f"Formula: {formula_content}")
        
        return "\n".join(formula_text) if formula_text else ""
        
    except Exception as e:
        logging.warning(f"Formula extraction failed: {str(e)}")
        return ""

# def get_pdf_text(pdf_docs):
#     """Extract text and tables from PDF documents."""
#     text = ""
#     documents = []
    
#     for pdf in pdf_docs:
#         try:
#             with pdfplumber.open(pdf) as pdf_reader:
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     # Extract text
#                     page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
#                     text += page_text + "\n"
                    
#                     # Extract tables
#                     tables = page.extract_tables()
#                     for table in tables:
#                         table_text = "Table:\n"
#                         for row in table:
#                             filtered_row = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
#                             if filtered_row:
#                                 table_text += " | ".join(filtered_row) + "\n"
#                         text += table_text + "\n"
#                         documents.append(Document(
#                             page_content=table_text,
#                             metadata={'source': 'table', 'page': page_num + 1, 'filename': pdf.name}
#                         ))
                    
#                     documents.append(Document(
#                         page_content=page_text,
#                         metadata={'source': 'pdf_text', 'page': page_num + 1, 'filename': pdf.name}
#                     ))
#         except Exception as e:
#             logging.error(f"Error processing PDF {pdf.name}: {str(e)}")
#             raise
                    
#     return text, documents
def get_pdf_text(pdf_docs):
    """
    Extract text and tables from PDF documents with enhanced table processing and structure preservation.
    Returns both raw text and structured documents for RAG pipeline.
    """
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page_num, page in enumerate(pdf_reader.pages):
                    # Extract and process tables first
                    tables = page.extract_tables()
                    for table in tables:
                        # Convert table to structured format
                        header_row = [str(cell).strip() if cell else "" for cell in table[0]]
                        table_data = []
                        
                        # Process each row and maintain column alignment
                        for row in table[1:]:
                            row_data = {}
                            for idx, cell in enumerate(row):
                                if idx < len(header_row):
                                    header = header_row[idx] if header_row[idx] else f"Column_{idx}"
                                    row_data[header] = str(cell).strip() if cell else ""
                            table_data.append(row_data)
                        
                        # Create searchable table text
                        table_text = "Table Content:\n"
                        # Add headers
                        table_text += " | ".join(header_row) + "\n"
                        # Add separator
                        table_text += "-" * 50 + "\n"
                        # Add data rows
                        for row in table_data:
                            table_text += " | ".join(row.values()) + "\n"
                        
                        # Store table as structured document
                        table_doc = Document(
                            page_content=table_text,
                            metadata={
                                'source': 'table',
                                'page': page_num + 1,
                                'filename': pdf.name,
                                'table_data': table_data,  # Store structured data for easier querying
                                'headers': header_row
                            }
                        )
                        documents.append(table_doc)
                        text += table_text + "\n"
                    
                    # Extract and process regular text
                    page_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,  # Preserve layout information
                        keep_blank_chars=True  # Maintain spacing
                    ) or ""
                    
                    # Clean and structure the text
                    cleaned_text = clean_text(page_text)
                    
                    # Create text document with enhanced metadata
                    text_doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            'source': 'pdf_text',
                            'page': page_num + 1,
                            'filename': pdf.name,
                            'layout_info': {
                                'width': page.width,
                                'height': page.height,
                                'orientation': 'portrait' if page.height > page.width else 'landscape'
                            }
                        }
                    )
                    documents.append(text_doc)
                    text += cleaned_text + "\n"
                    
        except Exception as e:
            logging.error(f"Error processing PDF {pdf.name}: {str(e)}")
            raise
                    
    return text, documents

def clean_text(text):
    """Clean and structure extracted text."""
    # Remove excessive whitespace while preserving meaningful spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Preserve list structures
    text = re.sub(r'(?<=\d)\.(?=\s)', '. ', text)
    
    # Preserve paragraph breaks
    text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
    
    # Handle special characters and encoding issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text

def get_non_table_pdf_text(pdf_docs):
    """Extract non-tabular text from PDF documents."""
    text = ""
    documents = []
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                
                # Clean text
                cleaned_text = ' '.join(page_text.split())
                cleaned_text = cleaned_text.replace('-\n', '')
                
                metadata = {
                    'source': 'pdf_non_table',
                    'page': page_num + 1,
                    'filename': pdf.name
                }
                
                # Split into chunks
                chunks = [cleaned_text[i:i+500] for i in range(0, len(cleaned_text), 400)]
                
                for chunk in chunks:
                    text += chunk + "\n"
                    documents.append(Document(page_content=chunk, metadata=metadata))
        except Exception as e:
            logging.error(f"Error processing PDF {pdf.name}: {str(e)}")
            raise
                
    return text, documents

def get_csv_text(csv_file):
    """Extract text from CSV files."""
    try:
        text = ""
        csv_file.seek(0)
        content = csv_file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        
        for row in csv_reader:
            filtered_row = [cell.strip() for cell in row if cell.strip()]
            if filtered_row:
                text += " | ".join(filtered_row) + "\n"
        return text
    except Exception as e:
        logging.error(f"Error processing CSV {csv_file.name}: {str(e)}")
        raise

def get_excel_text(excel_files):
    """Extract text from Excel files."""
    text = ""
    documents = []
    
    for excel_file in excel_files:
        try:
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            for sheet_name, sheet_data in excel_data.items():
                sheet_text = f"Sheet: {sheet_name}\n"
                
                for _, row in sheet_data.iterrows():
                    filtered_row = [str(cell).strip() for cell in row if pd.notna(cell) and str(cell).strip()]
                    if filtered_row:
                        sheet_text += " | ".join(filtered_row) + "\n"
                
                text += sheet_text + "\n"
                documents.append(Document(
                    page_content=sheet_text,
                    metadata={'source': f"Excel Sheet {sheet_name}", 'filename': excel_file.name}
                ))
        except Exception as e:
            logging.error(f"Error processing Excel file {excel_file.name}: {str(e)}")
            raise
            
    return text, documents

def get_ppt_text(ppt_files):
    """Extract text from PowerPoint files."""
    text = ""
    documents = []
    
    for ppt_file in ppt_files:
        try:
            prs = Presentation(ppt_file)
            
            for slide_num, slide in enumerate(prs.slides):
                slide_text = f"Slide {slide_num + 1}:\n"
                
                # Process shapes
                for shape in slide.shapes:
                    # Extract text from text-containing shapes
                    if hasattr(shape, 'text'):
                        shape_text = shape.text.strip()
                        if shape_text:
                            slide_text += shape_text + "\n"
                    
                    # Handle grouped shapes
                    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                        for subshape in shape.shapes:
                            if hasattr(subshape, 'text'):
                                subshape_text = subshape.text.strip()
                                if subshape_text:
                                    slide_text += subshape_text + "\n"
                    
                    # Handle tables
                    if hasattr(shape, 'has_table') and shape.has_table:
                        table = shape.table
                        table_text = "Table:\n"
                        for row in table.rows:
                            row_text = " | ".join([cell.text.strip() for cell in row.cells])
                            table_text += row_text + "\n"
                        slide_text += table_text + "\n"
                
                # Extract formulas
                formulas = re.findall(r'\$.*?\$', slide_text)
                if formulas:
                    slide_text += "Formulas: " + " ".join(formulas) + "\n"
                
                text += slide_text + "\n"
                documents.append(Document(
                    page_content=slide_text,
                    metadata={'source': f"PowerPoint Slide {slide_num + 1}", 'filename': ppt_file.name}
                ))
                
        except Exception as e:
            logging.error(f"Error processing PowerPoint file {ppt_file.name}: {str(e)}")
            raise
    
    return text, documents

def get_word_text(word_files):
    """Extract text from Word documents."""
    text = ""
    documents = []
    
    for word_file in word_files:
        try:
            doc = DocxDocument(word_file)
            doc_text = ""
            
            for element in doc.element.body:
                if element.tag.endswith('p'):
                    paragraph = element.text
                    doc_text += paragraph + "\n"
                    
                    # Extract formulas
                    formulas = re.findall(r'\$.*?\$', paragraph)
                    if formulas:
                        doc_text += "Formulas: " + " ".join(formulas) + "\n"
                
                elif element.tag.endswith('tbl'):
                    table_text = "Table:\n"
                    for row in element.findall('.//w:tr', namespaces=element.nsmap):
                        cells = [cell.text for cell in row.findall('.//w:t', namespaces=element.nsmap)]
                        table_text += " | ".join(cells) + "\n"
                    doc_text += table_text + "\n"
            
            text += doc_text
            documents.append(Document(page_content=doc_text, metadata={'source': f"Word Document"}))
        
        except Exception as e:
            print(f"Error processing {word_file}: {e}")
    
    return text, documents
