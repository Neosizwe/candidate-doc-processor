import os
import re
import io
import zipfile
import cv2
import numpy as np
import pandas as pd
import pdfplumber
import easyocr
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter

st.set_page_config(page_title="Candidate Document Processor", page_icon="📄", layout="wide")

# Initialize EasyOCR reader once (loads deep learning weights into memory)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

HEADER_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded", "full", 
    "names", "name", "fication", "check", "or", "see", "fe", "ee", "se", "document",
    "residential", "address", "hereby", "confirm"
}

def clean_candidate_name(raw_text):
    if not raw_text:
        return None
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', str(raw_text))
    words = [w.capitalize() for w in cleaned.split() if len(w.strip()) > 1]
    filtered = [w for w in words if w.lower() not in HEADER_BLACKLIST]

    if len(filtered) >= 2:
        return "_".join(filtered[:3])
    elif len(filtered) == 1 and len(filtered[0]) > 2:
        return filtered[0]
    return None

def clean_candidate_id(raw_text):
    if not raw_text:
        return None
    cleaned = str(raw_text)
    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)

    digits = re.sub(r'\D', '', cleaned)
    match = re.search(r'\b(\d{13})\b', digits)
    return match.group(1) if match else (digits if len(digits) == 13 else None)

def extract_with_easyocr(pil_img):
    img_np = np.array(pil_img.convert('RGB'))
    
    # EasyOCR returns bounding boxes, text strings, and confidence scores
    results = reader.readtext(img_np, detail=1)
    
    full_text = []
    cand_name = None
    cand_id = None
    
    for i, res in enumerate(results):
        text_str = res[1].strip()
        full_text.append(text_str)
        
        # Look for "Full Names" or "Name" label and take the next detected bounding box
        if re.search(r'Full\s*Name|Name', text_str, re.IGNORECASE) and not cand_name:
            if i + 1 < len(results):
                cand_name = clean_candidate_name(results[i+1][1])
                
        # Look for "Identity" or "ID" label and take adjacent bounding box
        if re.search(r'Identity|ID\s*No|ID\s*Number', text_str, re.IGNORECASE) and not cand_id:
            if i + 1 < len(results):
                cand_id = clean_candidate_id(results[i+1][1])

    text_block = " ".join(full_text)
    
    # Fallback search if label offsets missed
    if not cand_id:
        cand_id = clean_candidate_id(text_block)
    if not cand_name:
        cand_name = clean_candidate_name(text_block)
        
    return cand_name, cand_id, text_block

def process_single_page(page_bytes):
    images = convert_from_bytes(page_bytes)
    if not images:
        return None, None, "Document"

    pil_img = images[0]

    # Step 1: Deep Learning Handwriting Extraction
    cand_name, cand_id, text_block = extract_with_easyocr(pil_img)

    # Step 2: Document Categorization
    text_lower = text_block.lower()
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "republic of south africa" in text_lower or "identity card" in text_lower:
        doc_type = "Smart-ID"
    elif "senior certificate" in text_lower:
        doc_type = "Senior-Certificate"

    return cand_name, cand_id, doc_type

st.title("📄 Candidate Pack Processor (EasyOCR Engine)")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Documents"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Analyzing handwriting via Deep Learning Neural Nets..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                reader_pdf = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader_pdf.pages)

                pages_data = []
                extracted_names = []
                extracted_ids = []

                for p_idx in range(total_pages):
                    writer = PdfWriter()
                    writer.add_page(reader_pdf.pages[p_idx])

                    p_io = io.BytesIO()
                    writer.write(p_io)
                    p_bytes = p_io.getvalue()

                    p_name, p_id, p_type = process_single_page(p_bytes)

                    if p_name:
                        extracted_names.append(p_name)
                    if p_id:
                        extracted_ids.append(p_id)

                    pages_data.append({"idx": p_idx, "bytes": p_bytes, "type": p_type})

                final_name = extracted_names[0] if extracted_names else "Candidate"
                final_id = extracted_ids[0] if extracted_ids else "NoID"

                for p in pages_data:
                    suffix = f"_pg{p['idx']+1}" if total_pages > 1 else ""
                    filename = f"{final_name}_{final_id}_{p['type']}{suffix}.pdf"
                    folder_name = f"{final_name}_{final_id}"

                    zip_file.writestr(f"{folder_name}/{filename}", p["bytes"])

                    records.append({
                        "Source File": uploaded_file.name,
                        "Page": p['idx'] + 1,
                        "Candidate Name": final_name,
                        "ID Number": final_id,
                        "Document Type": p['type'],
                        "Renamed Output": filename
                    })

            df = pd.DataFrame(records)
            zip_file.writestr("Processing_Summary.csv", df.to_csv(index=False).encode('utf-8'))

    st.success("Documents processed successfully!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Structured ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidates.zip",
        mime="application/zip"
    )
# Downscale image slightly to speed up CPU inference in EasyOCR
img_np = np.array(pil_img.convert('RGB'))
h, w, _ = img_np.shape
if max(h, w) > 1500:
    img_np = cv2.resize(img_np, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
