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

# 1. Page & Environment Configuration
st.set_page_config(
    page_title="Candidate Pack Processor",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Strict blacklist of words that must NEVER appear in candidate names
HEADER_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded",
    "full", "name", "names", "fication", "check", "or", "see", "fe", "ee", "se",
    "hereby", "confirm", "department", "home", "affairs", "residential", "address"
}


# 2. Text Cleansing Helpers
def clean_candidate_name(raw_text):
    """Strips header boilerplate and extracts a clean Candidate Name."""
    if not raw_text:
        return None

    # Strip out non-alphabetic characters
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    
    # Filter out blacklisted document headers
    valid_words = [w for w in words if w.lower() not in HEADER_BLACKLIST]

    if len(valid_words) >= 2:
        return "_".join([w.capitalize() for w in valid_words[:3]])
    elif len(valid_words) == 1 and len(valid_words[0]) > 2:
        return valid_words[0].capitalize()

    return None


def clean_id_number(raw_id_str):
    """Sanitizes and enforces 13-digit South African ID rules."""
    if not raw_id_str:
        return None

    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    cleaned = raw_id_str
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)

    digits = re.sub(r'\D', '', cleaned)
    return digits if len(digits) == 13 else None


# 3. Document Extractors
def extract_smart_id(ocr_text):
    """Targeted parser for Smart ID cards."""
    text_lower = ocr_text.lower()
    if "republic of south africa" not in text_lower and "identity card" not in text_lower:
        return None, None

    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    extracted_id = id_match.group(1) if id_match else None

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


def extract_senior_certificate(ocr_text):
    """Targeted parser for Senior Certificates / Matric results."""
    match = re.search(r'awarded\s+to\s*\n*\s*([A-Za-z\s]{5,60})', ocr_text, re.IGNORECASE)
    if match:
        first_line = match.group(1).split('\n')[0]
        return clean_candidate_name(first_line)
    return None


def extract_form_details(ocr_text):
    """Line-restricted extractor for handwritten affidavits and forms."""
    # 1. ID Number
    id_num = None
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    if id_match:
        id_num = id_match.group(1)
    else:
        fuzzy_match = re.search(r'(?:ID|Identity)\s*[\.:\s]*([0-9OiIl|SsBQZ\s]{13,20})', ocr_text, re.IGNORECASE)
        if fuzzy_match:
            id_num = clean_id_number(fuzzy_match.group(1))

    # 2. Strict Line-Bounded Name Target
    name = None
    
    # Target Pattern A: "Full Names: <Candidate Name>"
    fn_match = re.search(r'(?:Full\s*Names?|First\s*Names?)\s*[:\-\.]*\s*([^\n]+)', ocr_text, re.IGNORECASE)
    if fn_match:
        name = clean_candidate_name(fn_match.group(1))

    # Target Pattern B: "I, <Candidate Name>, hereby declare..."
    if not name:
        i_match = re.search(r'\bI,?\s+([A-Za-z\s]+?)(?:,|\s+ID|\s+identity|\s+hereby|\s+bearing)', ocr_text, re.IGNORECASE)
        if i_match:
            name = clean_candidate_name(i_match.group(1))

    return name, id_num


# 4. OCR Execution
def process_single_page(page_bytes):
    """Runs PDF/Image OCR and classifies document type."""
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    if len(text.strip()) < 30:
        images = convert_from_bytes(page_bytes)
        for img in images:
            cv_img = np.array(img.convert('L'))
            resized = cv2.resize(cv_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text += pytesseract.image_to_string(thresh, config='--oem 3 --psm 6') + "\n"

    text_lower = text.lower()

    # Step A: Smart ID
    smart_name, smart_id = extract_smart_id(text)
    if smart_name or smart_id:
        return smart_name, smart_id, "Smart-ID", True

    # Step B: Senior Certificate
    cert_name = extract_senior_certificate(text)
    if cert_name:
        return cert_name, None, "Senior-Certificate", True

    # Step C: Affidavits & Declarations
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"

    aff_name, aff_id = extract_form_details(text)
    return aff_name, aff_id, doc_type, False


# 5. Application UI & Flow
st.title("📄 Candidate Document Pack Splitter")

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

                # Pass 1: Extract details and locate Anchor identity across all pages
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

                # Pass 2: Establish unified identity for the whole pack
                final_name = anchor_name or fallback_name or "Candidate"
                final_id = anchor_id or fallback_id or "NoID"

                # Pass 3: Construct output filenames (CandidateName_IDNumber_DocType.pdf)
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

    st.success("Candidate packs processed successfully!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Structured Candidate ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
