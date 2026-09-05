import os
import re
import cv2
import numpy as np
import pytesseract
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Candidate Document Processor",
    page_icon="📄",
    layout="wide"
)

# Set local Windows path for Tesseract if running locally
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Document Extraction Logic
def extract_criminal_affidavit_details(ocr_text):
    """Extracts details specifically from Criminal Record Status Affidavits."""
    fullname = None
    id_number = None

    # Capture Full Name following "Full Names:"
    name_match = re.search(r'Full\s*Names?\s*:\s*([A-Za-z\s]{3,50})', ocr_text, re.IGNORECASE)
    if name_match:
        raw_name = name_match.group(1).split('\n')[0].strip()
        clean_name = re.sub(r'[^a-zA-Z\s]', '', raw_name).strip()
        if len(clean_name) > 2:
            fullname = re.sub(r'\s+', '_', clean_name).title()

    # Capture 13-Digit ID
    id_match = re.search(r'Identity\s*Number\s*:\s*(\d{13})', ocr_text, re.IGNORECASE)
    if not id_match:
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        
    if id_match:
        id_number = id_match.group(1) if id_match.groups() else id_match.group(0)

    return fullname, id_number


def parse_document(ocr_text):
    """
    Unified extractor for supported document types.
    """
    text_lower = ocr_text.lower()
    
    # TYPE 1: Criminal Record Affidavit
    if "criminal record status" in text_lower or "declaration of criminal" in text_lower:
        name, id_num = extract_criminal_affidavit_details(ocr_text)
        return name, id_num, "Criminal Check Affidavit"

    # TYPE 2: SA Smart ID Card
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
        return fullname, id_num, "South African Smart ID"

    # TYPE 3: Unemployment / B-BBEE Affidavit
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
        
        doc_type = "Unemployment Affidavit" if "unemployment" in text_lower else "BBBEE Affidavit"
        return name, id_num, doc_type

    # TYPE 4: Senior Certificate
    elif "senior certificate" in text_lower or "awarded to" in text_lower:
        cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', ocr_text, re.IGNORECASE)
        name = None
        if cert_match:
            name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', cert_match.group(1))).title()
            
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        return name, id_num, "Senior Certificate"

    # General Fallback
    id_match = re.search(r'\b\d{13}\b', ocr_text)
    return None, (id_match.group(0) if id_match else None), "General Document"


def process_image(image_bytes):
    """Converts image bytes to text using Tesseract OCR."""
    img = Image.open(image_bytes)
    text = pytesseract.image_to_string(img)
    return text


def process_pdf(pdf_bytes):
    """Extracts text directly from PDF or converts pages to images for OCR."""
    extracted_text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                extracted_text += t + "\n"
                
    # Fallback to OCR if embedded text extraction returned nothing
    if not extracted_text.strip():
        images = convert_from_bytes(pdf_bytes.getvalue())
        for img in images:
            extracted_text += pytesseract.image_to_string(img) + "\n"
            
    return extracted_text


# 3. Streamlit Interface UI
st.title("📄 Candidate Document Processor")
st.markdown("Upload candidate documents (PDF, JPG, PNG) to extract name, ID number, and document type automatically.")

st.sidebar.header("Options")
show_raw_text = st.sidebar.checkbox("Show Raw OCR Text", value=False)

uploaded_file = st.file_uploader("Choose a document...", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Document Preview")
        if uploaded_file.type == "application/pdf":
            st.info("PDF document uploaded successfully.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with col2:
        st.subheader("Extraction Results")
        with st.spinner("Processing document with OCR..."):
            try:
                if uploaded_file.type == "application/pdf":
                    ocr_text = process_pdf(uploaded_file)
                else:
                    ocr_text = process_image(uploaded_file)
                
                name, id_number, doc_type = parse_document(ocr_text)

                st.success("Processing Complete!")
                
                st.metric(label="Document Type Detected", value=doc_type or "Unknown")
                st.metric(label="Extracted Name", value=name or "Not Found")
                st.metric(label="Extracted Identity Number", value=id_number or "Not Found")

                if show_raw_text:
                    st.text_area("Raw Extracted Text", ocr_text, height=250)

            except Exception as e:
                st.error(f"Error processing file: {e}")
