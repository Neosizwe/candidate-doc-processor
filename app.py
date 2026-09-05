import io
import os
import re
import zipfile
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from pypdf import PdfReader, PdfWriter

st.set_page_config(page_title="Candidate Document Splitter & Organizer", layout="wide")

def extract_text_from_page_bytes(page_pdf_bytes):
    """
    Extracts text from a single-page PDF.
    Uses pdfplumber first, falling back to pytesseract OCR if text is sparse.
    """
    full_text = ""
    try:
        with pdfplumber.open(io.BytesIO(page_pdf_bytes)) as pdf:
            if len(pdf.pages) > 0:
                full_text = pdf.pages[0].extract_text() or ""
    except Exception:
        full_text = ""
        
    # OCR Fallback for scanned/handwritten pages
    if len(full_text.strip()) < 30:
        try:
            images = convert_from_bytes(page_pdf_bytes, dpi=150)
            if images:
                full_text = pytesseract.image_to_string(images[0])
        except Exception as e:
            st.warning(f"OCR failed on a page: {e}")
            
    return full_text

def classify_doctype(text):
    text_lower = text.lower()
    if any(k in text_lower for k in ["republic of south africa", "identity document", "identity card", "national identity"]):
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
    return match.group(0) if match else None

def extract_fullname(text):
    cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', text, re.IGNORECASE)
    if cert_match:
        name = cert_match.group(1).strip()
        return re.sub(r'\s+', '_', name)
    
    affidavit_match = re.search(r'I,?\s+the\s+undersigned\s+([A-Z\s]{3,40})', text, re.IGNORECASE)
    if affidavit_match:
        name = affidavit_match.group(1).strip()
        return re.sub(r'\s+', '_', name)

    return None

def split_pdf_file(pdf_bytes):
    """
    Splits a multi-page PDF into single-page PDF byte streams.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    single_pages = []
    
    for page_idx in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_idx])
        
        page_io = io.BytesIO()
        writer.write(page_io)
        page_bytes = page_io.getvalue()
        
        single_pages.append(page_bytes)
        
    return single_pages

def process_pdf_batch(pdf_bytes_list):
    """
    Splits PDFs into pages, runs OCR, extracts metadata,
    and fills in missing candidate details across pages in the pack.
    """
    extracted_records = []
    
    # Global fallbacks per uploaded pack/batch
    batch_candidate_name = None
    batch_id_number = None

    # Step 1: Extract and classify every page
    for pdf_bytes in pdf_bytes_list:
        pages = split_pdf_file(pdf_bytes)
        
        for page_bytes in pages:
            text = extract_text_from_page_bytes(page_bytes)
            
            doc_type = classify_doctype(text)
            id_num = extract_id_number(text)
            name = extract_fullname(text)
            
            if id_num and not batch_id_number:
                batch_id_number = id_num
            if name and not batch_candidate_name:
                batch_candidate_name = name
                
            extracted_records.append({
                "page_bytes": page_bytes,
                "doc_type": doc_type,
                "id_number": id_num,
                "candidate_name": name
            })

    # Step 2: Fill missing candidate names/IDs with batch fallbacks
    final_records = []
    for rec in extracted_records:
        candidate_name = rec["candidate_name"] or batch_candidate_name or "UnknownCandidate"
        id_number = rec["id_number"] or batch_id_number or "UnknownID"
        
        final_records.append({
            "bytes": rec["page_bytes"],
            "candidate_name": candidate_name,
            "id_number": id_number,
            "doc_type": rec["doc_type"]
        })
        
    return final_records

# --- Streamlit UI ---
st.title("Candidate Document Splitter & Organizer")

uploaded_files = st.file_uploader(
    "Upload multi-page candidate PDF packs or a ZIP file:",
    type=["pdf", "zip"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Split, Rename & Structure Documents"):
    raw_pdf_streams = []
    
    # Collect all PDF streams
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                for filename in z.namelist():
                    if filename.endswith(".pdf") and not filename.startswith("__MACOSX"):
                        raw_pdf_streams.append(z.read(filename))
        elif uploaded_file.name.endswith(".pdf"):
            raw_pdf_streams.append(uploaded_file.read())

    with st.spinner("Splitting PDF pages, running OCR, and organizing files..."):
        processed_records = process_pdf_batch(raw_pdf_streams)

    # Build output ZIP
    zip_buffer = io.BytesIO()
    doc_type_counts = {}
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for idx, rec in enumerate(processed_records):
            c_name = rec["candidate_name"]
            id_num = rec["id_number"]
            d_type = rec["doc_type"]
            
            # Avoid duplicate file name collisions if a candidate has multiple pages of the same doc type
            doc_key = f"{c_name}_{id_num}_{d_type}"
            doc_type_counts[doc_key] = doc_type_counts.get(doc_key, 0) + 1
            suffix = f"_{doc_type_counts[doc_key]}" if doc_type_counts[doc_key] > 1 else ""
            
            folder_name = f"{c_name}_{id_num}"
            new_file_name = f"{c_name}_{id_num}_{d_type}{suffix}.pdf"
            zip_path = os.path.join(folder_name, new_file_name)
            
            zip_out.writestr(zip_path, rec["bytes"])

    st.success(f"Successfully extracted and split {len(processed_records)} individual documents!")
    
    st.download_button(
        label="Download Structured Zip File",
        data=zip_buffer.getvalue(),
        file_name="Structured_Candidate_Documents.zip",
        mime="application/zip"
    )
