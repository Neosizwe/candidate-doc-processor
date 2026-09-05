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

# 1. Configuration
st.set_page_config(
    page_title="Candidate Pack Processor",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Text Sanitation & Cleansing
def sanitize_candidate_name(raw_name):
    """Strips titles, template headers, and non-alphabetic noise."""
    if not raw_name:
        return None

    # Blacklist of template titles/headers that shouldn't be matched as names
    blacklist = [
        "affidavit", "declaration", "criminal", "record", "status", "undersigned",
        "bbbe", "certification", "unemployment", "republic", "south africa",
        "national", "identity", "card", "senior", "certificate", "awarded", "full name"
    ]

    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', raw_name)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]

    # Filter out blacklisted header words
    filtered_words = [w for w in words if w.lower() not in blacklist]

    if len(filtered_words) >= 2:
        return "_".join([w.capitalize() for w in filtered_words[:3]])
    elif len(filtered_words) == 1 and len(filtered_words[0]) > 2:
        return filtered_words[0].capitalize()

    return None


def clean_id_number(raw_id_str):
    """Cleans OCR misread digits in 13-digit SA IDs."""
    if not raw_id_str:
        return None

    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    cleaned = raw_id_str
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)

    digits = re.sub(r'\D', '', cleaned)
    return digits if len(digits) == 13 else None


# 3. High-Precision Target Extractors
def extract_smart_id(ocr_text):
    """Targeted extraction for RSA Smart ID Cards."""
    text_lower = ocr_text.lower()
    if "republic of south africa" not in text_lower and "national identity card" not in text_lower:
        return None, None

    # 1. Extract ID Number
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    extracted_id = id_match.group(1) if id_match else None

    # 2. Extract Names via Field Labels
    surname, names = None, None
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    for i, line in enumerate(lines):
        l_lower = line.lower()
        if "surname:" in l_lower or l_lower == "surname":
            if i + 1 < len(lines):
                surname = lines[i + 1]
        elif "names:" in l_lower or l_lower == "names":
            if i + 1 < len(lines):
                names = lines[i + 1]

    fullname = None
    if surname and names:
        fullname = sanitize_candidate_name(f"{names} {surname}")
    elif names:
        fullname = sanitize_candidate_name(names)

    return fullname, extracted_id


def extract_senior_certificate(ocr_text):
    """Targeted extraction for National Senior Certificates (Matric)."""
    text_lower = ocr_text.lower()
    if "senior certificate" not in text_lower and "awarded to" not in text_lower:
        return None

    # Capture text appearing directly under 'Awarded to'
    match = re.search(r'awarded\s+to\s*\n*\s*([A-Za-z\s]{5,60})', ocr_text, re.IGNORECASE)
    if match:
        return sanitize_candidate_name(match.group(1))

    return None


def extract_affidavit_details(ocr_text):
    """Targeted extraction for Criminal Check & BBBEE Affidavits."""
    # 1. ID Number Scan
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    id_num = id_match.group(1) if id_match else None

    if not id_num:
        fuzzy_match = re.search(r'(?:ID|Identity)\s*[\.:\s]*([0-9OiIl|SsBQZ\s]{13,20})', ocr_text, re.IGNORECASE)
        if fuzzy_match:
            id_num = clean_id_number(fuzzy_match.group(1))

    # 2. Name Extraction from Form Lines ('Full Names:', 'I, <Name>')
    name = None
    name_match = re.search(r'Full\s*Names?\s*:\s*([A-Za-z\s]{3,50})', ocr_text, re.IGNORECASE)
    if name_match:
        name = sanitize_candidate_name(name_match.group(1).split('\n')[0])

    if not name:
        line_match = re.search(r'I,?\s*([A-Za-z\s]{3,50})(?:,|\s+ID|\s+hereby)', ocr_text, re.IGNORECASE)
        if line_match:
            name = sanitize_candidate_name(line_match.group(1))

    return name, id_num


# 4. Image Processing & OCR Execution
def process_single_page(page_bytes):
    """Performs OCR and determines document type and extracted details."""
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    # Image OCR fallback if PDF text is sparse
    if len(text.strip()) < 30:
        images = convert_from_bytes(page_bytes)
        for img in images:
            # Contrast enhancement
            cv_img = np.array(img.convert('L'))
            resized = cv2.resize(cv_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text += pytesseract.image_to_string(thresh, config='--oem 3 --psm 6') + "\n"

    text_lower = text.lower()

    # Step A: Check for Smart ID Anchor
    smart_name, smart_id = extract_smart_id(text)
    if smart_name or smart_id:
        return smart_name, smart_id, "Smart-ID", True

    # Step B: Check for Senior Certificate Anchor
    cert_name = extract_senior_certificate(text)
    if cert_name:
        return cert_name, None, "Senior-Certificate", True

    # Step C: Form/Affidavit Extractors
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"

    aff_name, aff_id = extract_affidavit_details(text)
    return aff_name, aff_id, doc_type, False


# 5. UI & Application Workflow
st.title("📄 Candidate Document Pack Splitter")
st.markdown("Splits multi-page PDFs, anchors candidate identity using official documents, and renames all files and folders.")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process & Package Files"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Processing candidate packs..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

            for uploaded_file in uploaded_files:
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader.pages)

                pages_data = []
                anchor_name, anchor_id = None, None
                fallback_name, fallback_id = None, None

                # PASS 1: Read all pages and identify candidate master identity
                for p_idx in range(total_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[p_idx])
                    
                    p_io = io.BytesIO()
                    writer.write(p_io)
                    p_bytes = p_io.getvalue()

                    p_name, p_id, p_type, is_anchor = process_single_page(p_bytes)

                    if is_anchor:
                        if p_name and not anchor_name:
                            anchor_name = p_name
                        if p_id and not anchor_id:
                            anchor_id = p_id
                    else:
                        if p_name and not fallback_name:
                            fallback_name = p_name
                        if p_id and not fallback_id:
                            fallback_id = p_id

                    pages_data.append({"idx": p_idx, "bytes": p_bytes, "type": p_type})

                # PASS 2: Establish Master Identity for PDF Pack
                final_name = anchor_name or fallback_name or "Candidate"
                final_id = anchor_id or fallback_id or "NoID"

                # PASS 3: Write structured outputs to zip
                for p in pages_data:
                    suffix = f"_pg{p['idx']+1}" if total_pages > 1 else ""
                    filename = f"{final_name}_{final_id}_{p['type']}{suffix}.pdf"
                    folder = f"{final_name}_{final_id}"

                    zip_file.writestr(f"{folder}/{filename}", p["bytes"])

                    records.append({
                        "Source File": uploaded_file.name,
                        "Page": p['idx'] + 1,
                        "Candidate Name": final_name,
                        "ID Number": final_id,
                        "Document Type": p['type'],
                        "Renamed Output": filename
                    })

            # Add Summary CSV inside the zip file
            df = pd.DataFrame(records)
            zip_file.writestr("Processing_Summary.csv", df.to_csv(index=False).encode('utf-8'))

    st.success("Processing complete!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Structured ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
