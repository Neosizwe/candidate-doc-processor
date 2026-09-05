import os
import re
import io
import zipfile
import cv2
import numpy as np
import pandas as pd
import pytesseract
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter

st.set_page_config(page_title="Candidate Pack Processor", page_icon="📄", layout="wide")

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

HEADER_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded", "full", 
    "names", "name", "fication", "check", "or", "see", "fe", "ee", "se", "document",
    "residential", "address", "hereby", "confirm"
}


# 1. Text Sanitization Helpers
def clean_candidate_name(raw_text):
    """Strips title boilerplate and cleans candidate name."""
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
    """Sanitizes 13-digit South African ID numbers with character repair."""
    if not raw_text:
        return None
    cleaned = str(raw_text)
    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)

    digits = re.sub(r'\D', '', cleaned)
    match = re.search(r'\b(\d{13})\b', digits)
    return match.group(1) if match else (digits if len(digits) == 13 else None)


# 2. Crash-Proof Spatial Layout Extractor
def extract_fields_by_spatial_layout(pil_img):
    """
    Scans word positions using Tesseract OCR data.
    Finds label positions ('Full Name', 'Identity') and collects all words 
    printed/written in the same horizontal row to the right.
    """
    img_np = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Enhance contrast for handwritten ink
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Get OCR word coordinates dataframe
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DATAFRAME)

    # FIX FOR STREAMLIT CRASH: Safely convert 'text' column to string before filtering
    data = data.dropna(subset=['text']).copy()
    data['text'] = data['text'].astype(str).str.strip()
    data = data[data['text'] != '']

    if data.empty:
        return None, None

    found_name = None
    found_id = None

    # Search for Name Field Label
    name_labels = data[data['text'].str.contains(r'Name|Names|Full', case=False, regex=True)]
    if not name_labels.empty:
        for _, row in name_labels.iterrows():
            y_top = row['top'] - 15
            y_bottom = row['top'] + row['height'] + 20
            x_left = row['left'] + row['width']

            # Capture words on the same horizontal line to the right of label
            row_words = data[(data['top'] >= y_top) & (data['top'] <= y_bottom) & (data['left'] > x_left)]
            row_text = " ".join(row_words['text'].tolist())
            candidate = clean_candidate_name(row_text)
            if candidate:
                found_name = candidate
                break

    # Search for ID Field Label
    id_labels = data[data['text'].str.contains(r'Identity|ID|Number', case=False, regex=True)]
    if not id_labels.empty:
        for _, row in id_labels.iterrows():
            y_top = row['top'] - 15
            y_bottom = row['top'] + row['height'] + 25
            x_left = row['left'] + row['width']

            # Capture words on the same horizontal line to the right
            row_words = data[(data['top'] >= y_top) & (data['top'] <= y_bottom) & (data['left'] > x_left)]
            row_text = " ".join(row_words['text'].tolist())
            candidate_id = clean_candidate_id(row_text)
            if candidate_id:
                found_id = candidate_id
                break

    return found_name, found_id


# 3. Page Level Processing
def process_single_page(page_bytes):
    images = convert_from_bytes(page_bytes)
    if not images:
        return None, None, "Document"

    pil_img = images[0]

    # Step 1: Spatial layout extraction
    cand_name, cand_id = extract_fields_by_spatial_layout(pil_img)

    # Step 2: Full-text fallback parse
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    if not text.strip():
        text = pytesseract.image_to_string(pil_img)

    text_lower = text.lower()

    # Step 3: Document type classification
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "republic of south africa" in text_lower or "identity card" in text_lower:
        doc_type = "Smart-ID"
    elif "senior certificate" in text_lower:
        doc_type = "Senior-Certificate"

    # Fallback name and ID patterns if spatial extraction didn't find them
    if not cand_id:
        id_match = re.search(r'\b(\d{13})\b', text)
        if id_match:
            cand_id = id_match.group(1)

    if not cand_name:
        fn_match = re.search(r'(?:Full\s*Names?|First\s*Names?)\s*[:\-\.]*\s*([^\n]+)', text, re.IGNORECASE)
        if fn_match:
            cand_name = clean_candidate_name(fn_match.group(1))

    return cand_name, cand_id, doc_type


# 4. Streamlit Application
st.title("📄 Candidate Document Pack Processor")

uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Documents"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Extracting candidate metadata..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for uploaded_file in uploaded_files:
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader.pages)

                pages_data = []
                extracted_names = []
                extracted_ids = []

                for p_idx in range(total_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[p_idx])

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
