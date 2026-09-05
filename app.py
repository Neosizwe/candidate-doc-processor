import io
import os
import re
import zipfile
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract

# Set page config
st.set_page_config(page_title="Candidate Document Organizer", layout="wide")

def extract_text_hybrid(pdf_bytes):
    """
    1. Tries direct PDF text extraction via pdfplumber (Fastest & lowest memory).
    2. Falls back to Tesseract OCR via pdf2image if text length < 30 chars.
    """
    full_text = ""
    
    # 1. Native Digital PDF Extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception:
        full_text = ""
        
    # 2. Fallback to OCR for Scanned/Handwritten PDFs
    if len(full_text.strip()) < 30:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=150) # Low DPI keeps memory low
            ocr_texts = []
            for img in images:
                text_page = pytesseract.image_to_string(img)
                ocr_texts.append(text_page)
            full_text = "\n".join(ocr_texts)
        except Exception as e:
            st.warning(f"OCR processing failed for a document: {e}")
            
    return full_text

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

def extract_id_number(text):
    match = re.search(r'\b\d{13}\b', text)
    return match.group(0) if match else "UnknownID"

def extract_fullname(text):
    cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', text, re.IGNORECASE)
    if cert_match:
        name = cert_match.group(1).strip()
        return re.sub(r'\s+', '_', name)
    
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

# App Header
st.title("Candidate Document Organizer")

uploaded_files = st.file_uploader(
    "Upload PDFs or a ZIP archive (scanned, digital, or handwritten):",
    type=["pdf", "zip"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process & Structure Files"):
    processed_records = []
    
    with st.spinner("Extracting text and organizing candidate documents..."):
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
            
    # Zip output generation
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for rec in processed_records:
            folder_name = f"{rec['candidate_name']}_{rec['id_number']}"
            new_file_name = f"{rec['candidate_name']}_{rec['id_number']}_{rec['doc_type']}.pdf"
            zip_path = os.path.join(folder_name, new_file_name)
            zip_out.writestr(zip_path, rec["bytes"])
            
    st.success(f"Processed {len(processed_records)} document(s)!")
    
    st.download_button(
        label="Download Structured Zip File",
        data=zip_buffer.getvalue(),
        file_name="Structured_Candidate_Documents.zip",
        mime="application/zip"
    )
