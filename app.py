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

st.set_page_config(page_title="Candidate Pack Splitter", page_icon="📄", layout="wide")

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 1. Bounding-Box Targeted Region Extractor
def extract_field_by_crop(pil_img, label_keywords):
    """
    Locates specified label keywords on the page, crops the image region 
    directly to the right of the label, and runs isolated OCR on the cropped segment.
    """
    # Convert image to numpy array
    img_np = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Get word layout bounding boxes from Tesseract
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.text.notnull() & (data.text.str.strip() != "")]

    if data.empty:
        return None

    img_h, img_w = gray.shape[:2]

    for kw in label_keywords:
        # Match label keyword in word dataframe
        matches = data[data['text'].str.contains(kw, case=False, regex=False)]
        
        if not matches.empty:
            for _, row in matches.iterrows():
                x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])

                # Calculate bounding box for field area directly to the right of label
                crop_x1 = min(x + w + 10, img_w - 1)
                crop_y1 = max(0, y - 10)
                crop_x2 = min(x + w + int(img_w * 0.45), img_w - 1)
                crop_y2 = min(y + h + 25, img_h - 1)

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue

                # Crop region and preprocess for handwriting clarity
                cropped = gray[crop_y1:crop_y2, crop_x1:crop_x2]
                resized = cv2.resize(cropped, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Run OCR specifically on cropped target segment
                txt = pytesseract.image_to_string(thresh, config='--oem 3 --psm 7')
                txt_clean = re.sub(r'[^a-zA-Z0-9\s]', '', txt).strip()

                if len(txt_clean) > 1:
                    return txt_clean

    return None


# 2. Text Normalization Cleaners
def clean_candidate_name(raw_text):
    """Cleans extracted candidate string into proper format."""
    if not raw_text:
        return None

    cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_text)
    words = [w.capitalize() for w in cleaned.split() if len(w) > 1]

    blacklist = {"affidavit", "declaration", "criminal", "record", "status", "full", "names", "name", "identity"}
    words = [w for w in words if w.lower() not in blacklist]

    if len(words) >= 1:
        return "_".join(words[:3])

    return None


def clean_id_number(raw_text):
    """Sanitizes extracted ID number into 13 digits."""
    if not raw_text:
        return None

    replacements = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'i': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2'}
    for char, digit in replacements.items():
        raw_text = raw_text.replace(char, digit)

    digits = re.sub(r'\D', '', raw_text)
    return digits if len(digits) == 13 else None


# 3. Page Level Processing Pipeline
def process_single_page(page_bytes):
    # Convert PDF page bytes to PIL Image
    images = convert_from_bytes(page_bytes)
    if not images:
        return None, None, "Document"

    pil_img = images[0]

    # Step 1: Run region-targeted crops for Name and ID labels
    raw_name_crop = extract_field_by_crop(pil_img, ["Names", "Name", "Full"])
    raw_id_crop = extract_field_by_crop(pil_img, ["Identity", "ID", "Number"])

    candidate_name = clean_candidate_name(raw_name_crop)
    candidate_id = clean_id_number(raw_id_crop)

    # Step 2: Full-page fallback parse if crop returns empty
    text = ""
    with pdfplumber.open(io.BytesIO(page_bytes)) as pdf:
        if len(pdf.pages) > 0:
            text = pdf.pages[0].extract_text() or ""

    if len(text.strip()) < 20:
        cv_img = np.array(pil_img.convert('L'))
        thresh = cv2.adaptiveThreshold(cv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')

    text_lower = text.lower()

    # Step 3: Classification
    doc_type = "Document"
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "republic of south africa" in text_lower or "identity card" in text_lower:
        doc_type = "Smart-ID"
    elif "senior certificate" in text_lower:
        doc_type = "Senior-Certificate"

    # Step 4: Fallback full-text extraction if targeted crop yielded no name/ID
    if not candidate_id:
        id_match = re.search(r'\b(\d{13})\b', text)
        if id_match:
            candidate_id = id_match.group(1)

    if not candidate_name:
        fn_match = re.search(r'(?:Full\s*Names?|First\s*Names?)\s*[:\-\.]*\s*([^\n]+)', text, re.IGNORECASE)
        if fn_match:
            candidate_name = clean_candidate_name(fn_match.group(1))

    return candidate_name, candidate_id, doc_type


# 4. Streamlit Main Flow
st.title("📄 Candidate Document Pack Splitter")

uploaded_files = st.file_uploader("Upload Candidate PDF Packs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process & Rename Packs"):
    records = []
    zip_buffer = io.BytesIO()

    with st.spinner("Extracting handwritten data using visual region cropping..."):
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

            for uploaded_file in uploaded_files:
                reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
                total_pages = len(reader.pages)

                pages_data = []
                detected_names = []
                detected_ids = []

                # Pass 1: Region-crop parsing across all pages
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

                # Pass 2: Establish Master Identity for candidate pack
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

    st.success("Processing complete!")
    st.dataframe(df)

    st.download_button(
        label="📥 Download Renamed ZIP Pack",
        data=zip_buffer.getvalue(),
        file_name="Processed_Candidate_Packs.zip",
        mime="application/zip"
    )
