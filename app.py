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

# Strict word blacklist to discard printed template headings
BLACK_LIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded",
    "full", "name", "names", "fication", "check", "or", "see", "fe", "ee", "se",
    "hereby", "confirm", "department", "home", "affairs", "residential", "address",
    "document", "southafriga", "on", "of", "noed", "noid"
}


def preprocess_handwriting(pil_img):
    """
    Enhances handwritten pen ink by converting image to HSV, 
    filtering for blue/dark ink, and applying adaptive thresholding.
    """
    img_np = np.array(pil_img.convert('RGB'))
    
    # 1. Convert to HSV color space to highlight blue/black ink
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Define color range for blue pen ink
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Convert to grayscale for dark/black ink
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Combine ink masks
    combined_mask = cv2.bitwise_or(mask_blue, mask_dark)
    
    # 2. Upscale 2x for OCR stroke recognition
    resized = cv2.resize(combined_mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 3. Morphological dilation to reconnect fragmented handwriting strokes
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(resized, kernel, iterations=1)
    
    return cv2.bitwise_not(processed)


def sanitize_extracted_name(raw_text):
    """Strips title words and cleans candidate name."""
    if not raw_text:
        return None

    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
    
    # Filter out template boilerplate words
    valid_words = [w.capitalize() for w in words if w.lower() not in BLACK_LIST]

    if len(valid_words) >= 2:
        return "_".join(valid_words[:3])
    elif len(valid_words) == 1 and len(valid_words[0]) > 2:
        return valid_words[0]

    return None


def sanitize_extracted_id(raw_text):
    """Extracts and verifies 13-digit South African ID structure."""
    if not raw_text:
        return None

    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    for char, digit in replacements.items():
        raw_text = raw_text.replace(char, digit)

    digits = re.sub(r'\D', '', raw_text)
    
    # Look for 13-digit sequence
    match = re.search(r'\b(\d{13})\b', digits)
    if match:
        return match.group(1)
        
    return digits if len(digits) == 13 else None


def extract_page_fields(pil_img):
    """Runs targeted OCR to extract handwritten names and ID numbers."""
    processed_img = preprocess_handwriting(pil_img)
    
    # Run OCR with sparse text mode (PSM 11) to capture isolated handwritten text
    ocr_text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 11')

    # Step 1: Strict Name Extraction using field labels
    name = None
    name_match = re.search(r'(?:Full\s*Names?|First\s*Names?|Name)\s*[:\-\.]*\s*([A-Za-z\s]{3,35})(?=\n|ID|Identity|Address|$)', ocr_text, re.IGNORECASE)
    if name_match:
        name = sanitize_extracted_name(name_match.group(1))

    if not name:
        i_match = re.search(r'\bI,?\s+([A-Za-z\s]{3,35}?)(?:,|\s+ID|\s+identity|\s+hereby|\s+bearing)', ocr_text, re.IGNORECASE)
        if i_match:
            name = sanitize_extracted_name(i_match.group(1))

    # Step 2: Strict ID Extraction
    id_num = sanitize_extracted_id(ocr_text)

    # Step 3: Classify document type
    text_lower = ocr_text.lower()
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "republic of south africa" in text_lower or "identity card" in text_lower:
        doc_type = "Smart-ID"
    elif "senior certificate" in text_lower or "awarded to" in text_lower:
        doc_type = "Senior-Certificate"

    return name, id_num, doc_type


# Streamlit Interface
st.title("📄 Candidate Document Pack Processor")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process & Rename Packs"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Filtering ink layers and extracting handwritten metadata..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

            for uploaded_file in uploaded_files:
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader.pages)

                pages_data = []
                detected_names = []
                detected_ids = []

                # Convert PDF pages to images
                images = convert_from_bytes(uploaded_file.getvalue())

                # Pass 1: Extract fields across all pages
                for p_idx in range(total_pages):
                    writer = PdfWriter()
                    writer.add_page(reader.pages[p_idx])

                    p_io = io.BytesIO()
                    writer.write(p_io)
                    p_bytes = p_io.getvalue()

                    pil_img = images[p_idx]
                    p_name, p_id, p_type = extract_page_fields(pil_img)

                    if p_name:
                        detected_names.append(p_name)
                    if p_id:
                        detected_ids.append(p_id)

                    pages_data.append({"idx": p_idx, "bytes": p_bytes, "type": p_type})

                # Pass 2: Select single identity anchor for entire pack
                final_name = detected_names[0] if detected_names else "Candidate"
                final_id = detected_ids[0] if detected_ids else "NoID"

                # Pass 3: Structure output files
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
        label="📥 Download Renamed ZIP Pack",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
