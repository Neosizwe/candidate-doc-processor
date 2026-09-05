import os
import re
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter
from google.cloud import vision
from google.oauth2 import service_account

st.set_page_config(page_title="Candidate Document Processor", page_icon="📄", layout="wide")

# Initialize Google Cloud Vision Client
@st.cache_resource
def get_vision_client():
    # 1. Try loading from Streamlit Secrets (for Streamlit Cloud)
    if "gcp_service_account" in st.secrets:
        key_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(key_info)
        return vision.ImageAnnotatorClient(credentials=credentials)
    
    # 2. Try loading from local file `gcp_key.json`
    elif os.path.exists("gcp_key.json"):
        credentials = service_account.Credentials.from_service_account_file("gcp_key.json")
        return vision.ImageAnnotatorClient(credentials=credentials)
    
    else:
        st.error("Google Cloud credentials not found! Please provide 'gcp_key.json' or configure Streamlit Secrets.")
        st.stop()

client = get_vision_client()

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

def extract_with_cloud_vision(pil_img):
    # Convert PIL Image to byte buffer
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    content = img_byte_arr.getvalue()

    image = vision.Image(content=content)
    # Execute document text detection optimized for handwriting
    response = client.document_text_detection(image=image)
    
    full_text = response.full_text_annotation.text if response.full_text_annotation else ""

    # Parse targeted fields using label regex
    cand_name = None
    cand_id = None

    name_match = re.search(r'(?:Full\s*Names?|First\s*Names?|Name)\s*[:\-\.]*\s*([A-Za-z\s]{3,35})(?=\n|ID|Identity|Address|$)', full_text, re.IGNORECASE)
    if name_match:
        cand_name = clean_candidate_name(name_match.group(1))

    if not cand_id:
        cand_id = clean_candidate_id(full_text)
        
    if not cand_name:
        cand_name = clean_candidate_name(full_text)

    return cand_name, cand_id, full_text

def process_single_page(page_bytes):
    images = convert_from_bytes(page_bytes)
    if not images:
        return None, None, "Document"

    pil_img = images[0]

    cand_name, cand_id, text_block = extract_with_cloud_vision(pil_img)

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

st.title("📄 Candidate Pack Processor (Google Vision Engine)")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Documents"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Processing documents via Google Cloud Vision API..."):
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
