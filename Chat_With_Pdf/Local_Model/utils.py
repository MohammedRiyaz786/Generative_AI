# # prebuilt libraries
# import  pdfplumber
# import csv
# from langchain_core.documents import Document





# #custom lib



# # init





# # funcs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         with pdfplumber.open(pdf) as pdf_reader:
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
#                 tables = page.extract_tables()
#                 for table in tables:
#                     for row in table:
#                         # Filter out None values and join the rest into a string
#                         filtered_row = [cell for cell in row if cell is not None]
#                         if filtered_row:  # Check if the filtered row is not empty
#                             text += " | ".join(filtered_row) + "\n"  # Join table rows into text
#     return text

# import csv
# def get_csv_text(csv_file):
#     text=""
#     with open(csv_file,mode='r',encoding='utf-8') as file:
#         csv_reader=csv.reader(file)
#         for row in csv_reader:
#             filtered_row=[cell for cell in row if  cell]
#             if filtered_row:
#                 text+="|".join(filtered_row) +"\n"
#     return text

# #for excell files
# import pandas as pd
# from langchain_core.documents import Document
# def get_excel_text(excel_files):
#     text = ""
#     lis=[]
#     for excel_file in excel_files: 
#         excel_data = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
#         for sheet_name, sheet_data in excel_data.items():
#             text += f"Sheet: {sheet_name}\n"
#             for _, row in sheet_data.iterrows():
#                 filtered_row = [str(cell) for cell in row if pd.notna(cell)]
#                 if filtered_row:
#                     text += " | ".join(filtered_row) + "\n"
#             text += "\n"  #
#             lis.append(Document(page_content = text,metadata={'source' : f"Sheet {sheet_name}"}))
#     return lis

        






# if "__name__" == "__main__":
#     print("everything running properly")
# CHANGES -2
# import pdfplumber
# import csv
# import pandas as pd
# from langchain_core.documents import Document

# # Function to extract text from PDF files
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         with pdfplumber.open(pdf) as pdf_reader:
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
#                 tables = page.extract_tables()
#                 for table in tables:
#                     for row in table:
#                         filtered_row = [cell for cell in row if cell is not None]
#                         if filtered_row:
#                             text += " | ".join(filtered_row) + "\n"
#     return text

# # Function to extract text from CSV files
# def get_csv_text(csv_file):
#     text = ""
#     # Read from the uploaded file (Streamlit 'UploadedFile' object)
#     csv_reader = csv.reader(csv_file)
#     for row in csv_reader:
#         filtered_row = [cell for cell in row if cell]
#         if filtered_row:
#             text += "|".join(filtered_row) + "\n"
#     return text

# # Function to extract text from Excel files
# def get_excel_text(excel_files):
#     text = ""
#     documents = []
#     for excel_file in excel_files:
#         excel_data = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
#         for sheet_name, sheet_data in excel_data.items():
#             sheet_text = f"Sheet: {sheet_name}\n"
#             for _, row in sheet_data.iterrows():
#                 filtered_row = [str(cell) for cell in row if pd.notna(cell)]
#                 if filtered_row:
#                     sheet_text += " | ".join(filtered_row) + "\n"
#             text += sheet_text + "\n"
#             documents.append(Document(page_content=sheet_text, metadata={'source': f"Sheet {sheet_name}"}))
#     return text, documents
 # CHANGES -3
import pdfplumber
import csv
import pandas as pd
from langchain_core.documents import Document

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        filtered_row = [cell for cell in row if cell is not None]
                        if filtered_row:
                            text += " | ".join(filtered_row) + "\n"
    return text

# Function to extract text from CSV files
def get_csv_text(csv_file):
    text = ""
    # Read from the uploaded file (Streamlit 'UploadedFile' object)
    csv_file.seek(0)  # Reset the file pointer to the beginning
    csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines())  # Decode and read the content
    for row in csv_reader:
        filtered_row = [cell for cell in row if cell]
        if filtered_row:
            text += "|".join(filtered_row) + "\n"
    return text

# Function to extract text from Excel files
def get_excel_text(excel_files):
    text = ""
    documents = []
    for excel_file in excel_files:
        excel_data = pd.read_excel(excel_file, sheet_name=None)  # Load all sheets
        for sheet_name, sheet_data in excel_data.items():
            sheet_text = f"Sheet: {sheet_name}\n"
            for _, row in sheet_data.iterrows():
                filtered_row = [str(cell) for cell in row if pd.notna(cell)]
                if filtered_row:
                    sheet_text += " | ".join(filtered_row) + "\n"
            text += sheet_text + "\n"
            documents.append(Document(page_content=sheet_text, metadata={'source': f"Sheet {sheet_name}"}))
    return text, documents
