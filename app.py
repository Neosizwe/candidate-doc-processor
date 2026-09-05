import os
import re
import io
import zipfile
import pandas as pd
import pytesseract
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter

# 1. Page Configuration
st.set_page_config(
    page_title="Candidate Document Processor",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Document Extraction Logic
def extract_criminal_affidavit_details(ocr_text):
    fullname = None
    id_number = None

    name_match = re.search(r'Full\s*Names?\s*:\s*([A-Za-z\s]{3,50})', ocr_text, re.IGNORECASE)
    if name_match:
        raw_name = name_match.group(1).split('\n')[0].strip()
        clean_name = re.sub(r'[^a-zA-Z\s]', '', raw_name).strip()
        if len(clean_name) > 2:
            fullname = re.sub(r'\s+', '_', clean_name).title()

    id_match = re.search(r'Identity\s*Number\s*:\s*(\d{13})', ocr_text, re.IGNORECASE)
    if not id_match:
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        
    if id_match:
        id_number = id_match.group(1) if id_match.groups() else id_match.group(0)

    return fullname, id_number


def parse_document(ocr_text):
    text_lower = ocr_text.lower()
    
    if "criminal record status" in text_lower or "declaration of criminal" in text_lower:
        name, id_num = extract_criminal_affidavit_details(ocr_text)
        return name, id_num, "Criminal-Check-Affidavit"

    elif "national identity card" in text_lower or "republic of south africa" in text_lower:
        surname, names, id_num = None, None, None
        lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            l_lower = line.lower()
            if "surname:" in l_lower or l_lower == "surname":
                if i + 1 < len(lines):
                    surname = re.sub(r'[^a-zA-Z]', '', lines[i + 1]).title()
            elif "names:" in l_lower or l_lower == "names":
                if i + 1 < len(lines):
                    names = re.sub(r'[^a-zA-Z\s]', '', lines[i + 1]).strip().replace(' ', '_').title()
                    
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        fullname = f"{names}_{surname}" if names and surname else (names or surname)
        return fullname, id_num, "Smart-ID"

    elif "unemployment" in text_lower or "bbbee" in text_lower:
        line_match = re.search(r'I,?\s*([A-Za-z\s]{3,40})\s*,?\s*ID', ocr_text, re.IGNORECASE)
        name = None
        if line_match:
            raw_n = line_match.group(1).strip()
            if len(raw_n) > 2:
                name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', raw_n)).title()
                
        if not name:
            fn_match = re.search(r'Full\s*name\s*:\s*([A-Za-z\s]{3,40})', ocr_text, re.IGNORECASE)
            if fn_match:
                name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', fn_match.group(1))).title()
                
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        doc_type = "Unemployment-Affidavit" if "unemployment" in text_lower else "BBBEE-Affidavit"
        return name, id_num, doc_type

    elif "senior certificate" in text_lower or "awarded to" in text_lower:
        cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', ocr_text, re.IGNORECASE)
        name = None
        if cert_match:
            name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', cert_match.group(1))).title()
            
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        return name, id_num, "Senior-Certificate"

    id_match = re.search(r'\b\d{13}\b', ocr_text)
    return None, (id_match.group(0) if id_match else None), "Document"


def extract_ocr_from_single_pdf(page_pdf_bytes):
    extracted_text = ""
    with pdfplumber.open(io.BytesIO(page_pdf_bytes)) as pdf:
        if len(pdf.pages) > 0:
            extracted_text = pdf.pages[0].extract_text() or ""

    if not extracted_text.strip():
        images = convert_from_bytes(page_pdf_bytes)
        for img in images:
            extracted_text += pytesseract.image_to_string(img) + "\n"

    return extracted_text


# 3. Streamlit Interface
st.title("📄 Candidate Document Splitter & Renamer")
st.markdown("Upload candidate documents. Multi-page PDFs are automatically split into individual page files, categorized, renamed, and arranged into organized candidate folders.")

uploaded_files = st.file_uploader(
    "Upload Candidate Documents", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Split, Process & Package Files"):
        records = []
        zip_buffer = io.BytesIO()

        with st.spinner("Splitting PDFs and extracting document details..."):
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                file_counter = 1

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()

                    if uploaded_file.type == "application/pdf":
                        # Split PDF into single pages
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)

                        for page_idx in range(total_pages):
                            writer = PdfWriter()
                            writer.add_page(reader.pages[page_idx])
                            
                            single_page_io = io.BytesIO()
                            writer.write(single_page_io)
                            single_page_bytes = single_page_io.getvalue()

                            # Extract metadata for individual page
                            ocr_text = extract_ocr_from_single_pdf(single_page_bytes)
                            name, id_number, doc_type = parse_document(ocr_text)

                            clean_name = name or f"Candidate_{file_counter}"
                            clean_id = id_number or "NoID"
                            page_suffix = f"_pg{page_idx+1}" if total_pages > 1 else ""
                            new_filename = f"{clean_name}_{clean_id}_{doc_type}{page_suffix}.pdf"

                            folder_name = f"{clean_name}_{clean_id}"
                            zip_path = f"{folder_name}/{new_filename}"

                            zip_file.writestr(zip_path, single_page_bytes)

                            records.append({
                                "Source File": uploaded_file.name,
                                "Page Number": page_idx + 1,
                                "Renamed Filename": new_filename,
                                "Folder Path": folder_name,
                                "Extracted Name": name or "Unknown",
                                "Identity Number": id_number or "Unknown",
                                "Document Type": doc_type
                            })
                            file_counter += 1

                    else:
                        # Process image file
                        img = Image.open(io.BytesIO(file_bytes))
                        ocr_text = pytesseract.image_to_string(img)
                        name, id_number, doc_type = parse_document(ocr_text)

                        ext = os.path.splitext(uploaded_file.name)[1]
                        clean_name = name or f"Candidate_{file_counter}"
                        clean_id = id_number or "NoID"
                        new_filename = f"{clean_name}_{clean_id}_{doc_type}{ext}"

                        folder_name = f"{clean_name}_{clean_id}"
                        zip_path = f"{folder_name}/{new_filename}"

                        zip_file.writestr(zip_path, file_bytes)

                        records.append({
                            "Source File": uploaded_file.name,
                            "Page Number": 1,
                            "Renamed Filename": new_filename,
                            "Folder Path": folder_name,
                            "Extracted Name": name or "Unknown",
                            "Identity Number": id_number or "Unknown",
                            "Document Type": doc_type
                        })
                        file_counter += 1

                # Generate Summary Manifest CSV
                df = pd.DataFrame(records)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("Processing_Summary.csv", csv_bytes)

        st.success(f"Successfully processed and split into {len(records)} individual document file(s)!")
        
        st.dataframe(df)

        st.download_button(
            label="📥 Download Split & Renamed Files (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Split_Candidate_Documents.zip",
            mime="application/zip"
        )
