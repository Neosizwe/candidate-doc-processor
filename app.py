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

st.set_page_config(page_title="Candidate Document Processor", page_icon="📄", layout="wide")

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Blacklist terms preventing header leakage
HEADER_NOISE_BLACKLIST = {
    "affidavit", "declaration", "criminal", "record", "status", "undersigned",
    "bbbe", "bbbee", "certification", "unemployment", "republic", "south", "africa",
    "national", "identity", "card", "senior", "certificate", "awarded", "full", 
    "names", "name", "fication", "check", "or", "see", "fe", "ee", "se", "document"
}

def clean_extracted_name(raw_text):
    """Filters noisy characters and drops template headers."""
    if not raw_text:
        return None
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)
    words = [w.capitalize() for w in cleaned.split() if len(w.strip()) > 1]
    filtered_words = [w for w in words if w.lower() not in HEADER_NOISE_BLACKLIST]
    
    if len(filtered_words) >= 2:
        return "_".join(filtered_words[:3])
    elif len(filtered_words) == 1 and len(filtered_words[0]) > 2:
        return filtered_words[0]
    return None

def clean_extracted_id(raw_text):
    """Parses 13-digit South African ID numbers with OCR digit repair."""
    if not raw_text:
        return None
    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    for char, digit in replacements.items():
        raw_text = raw_text.replace(char, digit)
    digits = re.sub(r'\D', '', raw_text)
    
    # Locate valid 13-digit ID pattern
    match = re.search(r'\b(\d{13})\b', digits)
    return match.group(1) if match else (digits if len(digits) == 13 else None)

def extract_field_by_bounding_box(pil_img, keywords, is_id_field=False):
    """Locates anchor labels on the page and crops the adjacent target response box."""
    img_np = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Extract OCR data layout coordinates
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.text.notnull() & (data.text.str.strip() != "")]
    
    if data.empty:
        return None
        
    img_h, img_w = gray.shape[:2]

    for kw in keywords:
        matches = data[data['text'].str.contains(kw, case=False, regex=False)]
        if not matches.empty:
            for _, row in matches.iterrows():
                x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                
                # Crop parameters: right side of the label anchor
                crop_x1 = min(x + w + 5, img_w - 1)
                crop_y1 = max(0, y - 15)
                crop_x2 = min(x + w + int(img_w * 0.55), img_w - 1)
                crop_y2 = min(y + h + 30, img_h - 1)
                
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue

                crop_img = gray[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Image processing for handwritten ink enhancement
                resized = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                psm_mode = '--oem 3 --psm 7' if not is_id_field else '--oem 3 --psm 8'
                txt = pytesseract.image_to_string(thresh, config=psm_mode)
                
                if txt.strip():
                    return txt.strip()
    return None

def process_page_payload(page_bytes):
    images = convert_from_bytes(page_bytes)
    if not images:
        return None, None, "Document"
    
    pil_img = images[0]
    
    # 1. Coordinate-Based Bounding Box Extractions
    raw_name = extract_field_by_bounding_box(pil_img, ["Names", "Name", "Full"], is_id_field=False)
    raw_id = extract_field_by_bounding_box(pil_img, ["Identity", "ID", "Number"], is_id_field=True)
    
    candidate_name = clean_extracted_name(raw_name)
    candidate_id = clean_extracted_id(raw_id)
    
    # 2. Text Layer Processing Fallback
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""
            
    if not text.strip():
        text = pytesseract.image_to_string(pil_img)
        
    text_lower = text.lower()

    # 3. Document Categorization
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "republic of south africa" in text_lower or "identity card" in text_lower:
        doc_type = "Smart-ID"
    elif "senior certificate" in text_lower:
        doc_type = "Senior-Certificate"

    # Fallback to text parsing if coordinate cropping was empty
    if not candidate_id:
        id_match = re.search(r'\b(\d{13})\b', text)
        if id_match:
            candidate_id = id_match.group(1)
            
    if not candidate_name:
        fn_match = re.search(r'(?:Full\s*Names?|First\s*Names?)\s*[:\-\.]*\s*([^\n]+)', text, re.IGNORECASE)
        if fn_match:
            candidate_name = clean_extracted_name(fn_match.group(1))

    return candidate_name, candidate_id, doc_type

# Application Execution UI
st.title("📄 Candidate Document Pack Processor")

uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Documents"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Processing document layout boundaries..."):
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

                    p_name, p_id, p_type = process_page_payload(p_bytes)

                    if p_name:
                        extracted_names.append(p_name)
                    if p_id:
                        extracted_ids.append(p_id)

                    pages_data.append({"idx": p_idx, "bytes": p_bytes, "type": p_type})

                final_name = extracted_names[0] if extracted_names else "NoNameDetected"
                final_id = extracted_ids[0] if extracted_ids else "NoIDDetected"

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

    st.success("Extraction completed.")
    st.dataframe(df)

    st.download_button(
        label="📥 Download ZIP",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidates.zip",
        mime="application/zip"
    )
