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

# Blacklist to strip out standard template headings if Tesseract bleeds text
HEADER_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded",
    "full", "name", "names", "fication", "check", "residential", "address"
}


# 1. Handwritten Field Extractors (PRIMARY FOCUS)
def extract_handwritten_name(text):
    """Targeted extraction focusing strictly on hand-filled name fields."""
    if not text:
        return None

    # Priority 1: Direct match after "Full Name(s):" or "First Name(s):"
    fn_match = re.search(r'(?:Full\s*Names?|First\s*Names?)\s*[:\-\.]*\s*([^\n]+)', text, re.IGNORECASE)
    if fn_match:
        raw_val = fn_match.group(1).strip()
        cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_val)
        words = [w.capitalize() for w in cleaned.split() if len(w) > 1 and w.lower() not in HEADER_BLACKLIST]
        if len(words) >= 1:
            return "_".join(words[:3])

    # Priority 2: Declaration match "I, <NAME>,"
    i_match = re.search(r'\bI,?\s+([A-Za-z\s]{3,40}?)(?:,|\s+ID|\s+identity|\s+hereby|\s+bearing|\s+declare)', text, re.IGNORECASE)
    if i_match:
        raw_val = i_match.group(1).strip()
        cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_val)
        words = [w.capitalize() for w in cleaned.split() if len(w) > 1 and w.lower() not in HEADER_BLACKLIST]
        if len(words) >= 1:
            return "_".join(words[:3])

    return None


def extract_handwritten_id(text):
    """Targeted extraction for handwritten 13-digit SA IDs (handles spaced digits)."""
    if not text:
        return None

    # Priority 1: ID number directly following ID labels
    id_label_match = re.search(r'(?:ID|Identity|Identity\s*Number)\s*[:\-\.]*\s*([0-9\sOiIl|SsBQZ]{13,25})', text, re.IGNORECASE)
    if id_label_match:
        raw_id = id_label_match.group(1)
        replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
        for char, digit in replacements.items():
            raw_id = raw_id.replace(char, digit)
        digits = re.sub(r'\D', '', raw_id)
        if len(digits) == 13:
            return digits

    # Priority 2: Any sequence of 13 digits across the text string
    digits_only = re.sub(r'\D', '', text)
    matches = re.findall(r'\d{13}', digits_only)
    if matches:
        return matches[0]

    return None


# 2. Document Type Classifier
def classify_document_type(text):
    text_lower = text.lower()
    if "republic of south africa" in text_lower or "identity card" in text_lower:
        return "Smart-ID"
    elif "senior certificate" in text_lower or "awarded to" in text_lower:
        return "Senior-Certificate"
    elif "bbbe" in text_lower or "unemployment" in text_lower:
        return "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        return "Criminal-Check-Affidavit"
    return "Document"


# 3. Processing Single Page
def process_single_page(page_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    # Denoise & OCR if pdfplumber returns empty/sparse output
    if len(text.strip()) < 30:
        images = convert_from_bytes(page_bytes)
        for img in images:
            cv_img = np.array(img.convert('L'))
            resized = cv2.resize(cv_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text += pytesseract.image_to_string(thresh, config='--oem 3 --psm 6') + "\n"

    # Extract handwritten values directly
    cand_name = extract_handwritten_name(text)
    cand_id = extract_handwritten_id(text)
    doc_type = classify_document_type(text)

    return cand_name, cand_id, doc_type


# 4. Main Application Interface
st.title("📄 Candidate Document Pack Splitter")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Candidate Packs"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Extracting handwritten data and renaming packs..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

            for uploaded_file in uploaded_files:
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader.pages)

                pages_data = []
                detected_names = []
                detected_ids = []

                # Pass 1: Extract handwritten fields across all pages
                for p_idx in range(total_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[p_idx])
                    
                    p_io = io.BytesIO()
                    writer.write(p_io)
                    p_bytes = p_io.getvalue()

                    p_name, p_id, p_type = process_single_page(p_bytes)

                    if p_name:
                        detected_names.append(p_name)
                    if p_id:
                        detected_ids.append(p_id)

                    pages_data.append({"idx": p_idx, "bytes": p_bytes, "type": p_type})

                # Pass 2: Pick best candidate identity for the entire PDF pack
                final_name = detected_names[0] if detected_names else "Candidate"
                final_id = detected_ids[0] if detected_ids else "NoID"

                # Pass 3: Construct filenames: candidatename_IDNumber_doctype.pdf
                for p in pages_data:
                    suffix = f"_pg{p['idx']+1}" if total_pages > 1 else ""
                    filename = f"{final_name}_{final_id}_{p['type']}{suffix}.pdf"
                    folder_name = f"{final_name}_{final_id}"

                    zip_file.writestr(f"{folder_name}/{filename}", p["bytes"])

                    records.append({
                        "Source File": uploaded_file.name,
                        "Page Number": p['idx'] + 1,
                        "Candidate Name": final_name,
                        "ID Number": final_id,
                        "Document Type": p['type'],
                        "Renamed Output": filename
                    })

            df = pd.DataFrame(records)
            zip_file.writestr("Processing_Summary.csv", df.to_csv(index=False).encode('utf-8'))

    st.success("Processing complete!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Processed ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
