import io
import os
import re
import zipfile
import pdfplumber
import streamlit as st
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import easyocr

# Cache EasyOCR Reader to avoid reloading weights on every execution
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_easyocr()

def extract_text_hybrid(pdf_bytes):
    """
    Attempts native digital text extraction via pdfplumber.
    Falls back to OCR if little to no text is extracted.
    """
    full_text = ""
    
    # Strategy 1: Fast native text extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception:
        full_text = ""
        
    # Strategy 2: Fallback to OCR if page yields under 30 characters
    if len(full_text.strip()) < 30:
        # Convert PDF pages to PIL Images (dpi=200 balances speed and OCR accuracy)
        images = convert_from_bytes(pdf_bytes, dpi=200)
        ocr_texts = []
        
        for img in images:
            # Option A: EasyOCR (Recommended for handwritten text)
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0)
            text_page = " ".join(results)
            
            # Option B: PyTesseract (Faster alternative for clear scans)
            # Un-comment the line below if using PyTesseract instead of EasyOCR:
            # text_page = pytesseract.image_to_string(img)
            
            ocr_texts.append(text_page)
            
        full_text = "\n".join(ocr_texts)
        
    return full_text

# Document classification logic
def classify_doctype(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ["republic of south africa", "identity document", "identity card"]):
        return "ID"
    elif any(k in text_lower for k in ["senior certificate", "national senior certificate", "awarded to"]):
        return "SeniorCertificate"
    elif any(k in text_lower for k in ["bbbee", "b-bbee", "broad-based black"]):
        return "BBBEE-Affidavit"
    elif "unemployment" in text_lower:
        return "Unemployment-Affidavit"
    elif "cellphone" in text_lower or "cell phone" in text_lower:
        return "Cellphone-Affidavit"
    elif any(k in text_lower for k in ["criminal", "police clearance", "sap69"]):
        return "Criminal-Check-Affidavit"
    return "Document"

# Extraction using robust regular expressions
def extract_id_number(text):
    match = re.search(r'\b\d{13}\b', text)
    return match.group(0) if match else "UnknownID"

def extract_fullname(text):
    # Match names following "awarded to" or "this is to certify that"
    cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', text, re.IGNORECASE)
    if cert_match:
        name = cert_match.group(1).strip()
        # Clean up line breaks or multi-spaces
        return re.sub(r'\s+', '_', name)
    
    # Fallback to Affidavit / ID search patterns (e.g. "I, the undersigned [Name]")
    affidavit_match = re.search(r'I,?\s+the\s+undersigned\s+([A-Z\s]{3,40})', text, re.IGNORECASE)
    if affidavit_match:
        name = affidavit_match.group(1).strip()
        return re.sub(r'\s+', '_', name)

    return "UnknownCandidate"

def process_pdf_bytes(pdf_bytes, filename):
    extracted_text = extract_text_hybrid(pdf_bytes)
    
    doc_type = classify_doctype(extracted_text)
    id_num = extract_id_number(extracted_text)
    candidate_name = extract_fullname(extracted_text)
    
    return {
        "bytes": pdf_bytes,
        "candidate_name": candidate_name,
        "id_number": id_num,
        "doc_type": doc_type,
        "original_filename": filename
    }

# Streamlit Interface
st.set_page_config(page_title="Candidate Document OCR Processor", layout="wide")
st.title("Candidate Document Organizer (OCR Enabled)")

uploaded_files = st.file_uploader(
    "Upload PDFs or a ZIP archive (scanned, digital, or handwritten):",
    type=["pdf", "zip"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process & Structure Files"):
    processed_records = []
    
    with st.spinner("Extracting text and processing documents via OCR..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".zip"):
                with zipfile.ZipFile(uploaded_file, "r") as z:
                    for filename in z.namelist():
                        if filename.endswith(".pdf") and not filename.startswith("__MACOSX"):
                            pdf_bytes = z.read(filename)
                            processed_records.append(process_pdf_bytes(pdf_bytes, filename))
            elif uploaded_file.name.endswith(".pdf"):
                pdf_bytes = uploaded_file.read()
                processed_records.append(process_pdf_bytes(pdf_bytes, uploaded_file.name))
            
    # Build structured Zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for rec in processed_records:
            folder_name = f"{rec['candidate_name']}_{rec['id_number']}"
            new_file_name = f"{rec['candidate_name']}_{rec['id_number']}_{rec['doc_type']}.pdf"
            zip_path = os.path.join(folder_name, new_file_name)
            zip_out.writestr(zip_path, rec["bytes"])
            
    st.success(f"Successfully processed {len(processed_records)} document(s)!")
    
    st.download_button(
        label="Download Structured Zip Archive",
        data=zip_buffer.getvalue(),
        file_name="Structured_Candidate_Documents.zip",
        mime="application/zip"
    )
