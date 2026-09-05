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

# 1. Streamlit Configuration
st.set_page_config(
    page_title="Candidate Pack Splitter & Renamer",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Strict list of form title/header words that can NEVER be part of a candidate's name
TITLE_WORDS_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded",
    "full", "name", "names", "fication", "check", "residential", "address",
    "street", "hereby", "confirm", "programme", "seta", "funded"
}


# 2. Strict Field Extraction Logic
def clean_candidate_name(raw_text):
    """Cleans extracted name candidate string and ensures no header noise is present."""
    if not raw_text:
        return None

    # Keep only letters and spaces
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    
    # Remove blacklisted header words
    valid_words = [w for w in words if w.lower() not in TITLE_WORDS_BLACKLIST]

    if len(valid_words) >= 2:
        return "_".join([w.capitalize() for w in valid_words[:3]])
    elif len(valid_words) == 1 and len(valid_words[0]) > 2:
        return valid_words[0].capitalize()

    return None


def clean_id_number(raw_id_str):
    """Cleans OCR digit misreads in 13-digit SA IDs."""
    if not raw_id_str:
        return None

    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    cleaned = raw_id_str
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)

    digits = re.sub(r'\D', '', cleaned)
    return digits if len(digits) == 13 else None


def extract_from_smart_id(ocr_text):
    """Extracts Name and ID directly from Smart ID card layouts."""
    text_lower = ocr_text.lower()
    if "republic of south africa" not in text_lower and "identity card" not in text_lower:
        return None, None

    # ID Number Search
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    extracted_id = id_match.group(1) if id_match else None

    # Name Search: Targets text directly on lines following 'Surname' and 'Names'
    surname, names = None, None
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    for i, line in enumerate(lines):
        l_lower = line.lower()
        if "surname" in l_lower and i + 1 < len(lines):
            surname = lines[i + 1]
        elif "names" in l_lower and i + 1 < len(lines):
            names = lines[i + 1]

    fullname = None
    if surname and names:
        fullname = clean_candidate_name(f"{names} {surname}")
    elif names:
        fullname = clean_candidate_name(names)

    return fullname, extracted_id


def extract_from_senior_certificate(ocr_text):
    """Extracts Name from Matric / Senior Certificate layouts."""
    match = re.search(r'awarded\s+to\s*\n*\s*([A-Za-z\s]{5,60})', ocr_text, re.IGNORECASE)
    if match:
        # Take only the immediate first line under 'Awarded to'
        first_line = match.group(1).split('\n')[0]
        return clean_candidate_name(first_line)
    return None


def extract_from_affidavits(ocr_text):
    """Targeted line-by-line extraction for handwritten form fields."""
    # 1. ID Number Extraction
    id_num = None
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    if id_match:
        id_num = id_match.group(1)
    else:
        fuzzy_match = re.search(r'(?:ID|Identity)\s*[\.:\s]*([0-9OiIl|SsBQZ\s]{13,20})', ocr_text, re.IGNORECASE)
        if fuzzy_match:
            id_num = clean_id_number(fuzzy_match.group(1))

    # 2. Name Extraction - STRICT TARGETING ONLY
    name = None

    # Target Pattern A: "Full Names: <NAME>" or "Full name: <NAME>"
    fn_match = re.search(r'Full\s*Names?\s*:\s*([^\n]+)', ocr_text, re.IGNORECASE)
    if fn_match:
        name = clean_candidate_name(fn_match.group(1))

    # Target Pattern B: "I, <NAME>," or "I, <NAME> ID"
    if not name:
        i_match = re.search(r'I,?\s*([A-Za-z\s]+?)(?:,|\s+ID|\s+hereby)', ocr_text, re.IGNORECASE)
        if i_match:
            name = clean_candidate_name(i_match.group(1))

    return name, id_num


# 3. Document Processing Pipeline
def process_single_page(page_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    # Denoise & OCR if pdfplumber returns little/no text
    if len(text.strip()) < 30:
        images = convert_from_bytes(page_bytes)
        for img in images:
            cv_img = np.array(img.convert('L'))
            resized = cv2.resize(cv_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text += pytesseract.image_to_string(thresh, config='--oem 3 --psm 6') + "\n"

    text_lower = text.lower()

    # Priority 1: Smart ID Document
    smart_name, smart_id = extract_from_smart_id(text)
    if smart_name or smart_id:
        return smart_name, smart_id, "Smart-ID", True

    # Priority 2: Senior Certificate
    cert_name = extract_from_senior_certificate(text)
    if cert_name:
        return cert_name, None, "Senior-Certificate", True

    # Priority 3: Categorize Affidavits & Form Pages
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"

    aff_name, aff_id = extract_from_affidavits(text)
    return aff_name, aff_id, doc_type, False


# 4. Streamlit Application Core
st.title("📄 Candidate Document Pack Splitter")
st.markdown("Splits multi-page PDFs, anchors candidate identity, and outputs files named as `CandidateName_IDNumber_DocType.pdf`.")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Candidate Packs"):
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

                # PASS 1: Read all pages in the PDF pack and determine Candidate Identity
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

                # PASS 2: Establish Single Master Identity for PDF Pack
                final_candidate_name = anchor_name or fallback_name or "Candidate"
                final_candidate_id = anchor_id or fallback_id or "NoID"

                # PASS 3: Apply Master Identity to output filenames
                for p in pages_data:
                    suffix = f"_pg{p['idx']+1}" if total_pages > 1 else ""
                    filename = f"{final_candidate_name}_{final_candidate_id}_{p['type']}{suffix}.pdf"
                    folder_name = f"{final_candidate_name}_{final_candidate_id}"

                    zip_file.writestr(f"{folder_name}/{filename}", p["bytes"])

                    records.append({
                        "Source File": uploaded_file.name,
                        "Page Number": p['idx'] + 1,
                        "Candidate Name": final_candidate_name,
                        "ID Number": final_candidate_id,
                        "Document Type": p['type'],
                        "Renamed Output": filename
                    })

            df = pd.DataFrame(records)
            zip_file.writestr("Processing_Summary.csv", df.to_csv(index=False).encode('utf-8'))

    st.success("Candidate packs successfully processed!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Structured Candidate ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
