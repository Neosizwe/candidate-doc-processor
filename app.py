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

# 1. Page Configuration
st.set_page_config(
    page_title="Candidate Document Splitter & Cross-Matcher",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Advanced Preprocessing for Handwritten Text
def preprocess_for_handwriting(image):
    """
    Applies scaling, blurring, and adaptive thresholding to maximize 
    Tesseract's ability to extract handwritten text and spaced-out numbers.
    """
    open_cv_image = np.array(image.convert('L'))
    
    # Resize 2x for better character clarity
    resized = cv2.resize(open_cv_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Denoise and threshold
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh


def clean_handwritten_id(id_candidate_str):
    """
    Cleans OCR misreads common in handwritten IDs:
    Converts 'O', 'o', 'Q' -> '0', 'I', 'l', 'i' -> '1', 'S' -> '5', 'B' -> '8'
    """
    replacements = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2'
    }
    cleaned = id_candidate_str
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)
    
    # Strip non-digits
    digits_only = re.sub(r'\D', '', cleaned)
    if len(digits_only) == 13:
        return digits_only
    return None


def extract_smart_id_details(ocr_text):
    surname, names = None, None
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    for i, line in enumerate(lines):
        l_lower = line.lower()
        if "surname" in l_lower and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not any(k in next_line.lower() for k in ["names", "sex", "nationality"]):
                surname = re.sub(r'[^a-zA-Z]', '', next_line).title()
        
        if "names" in l_lower and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not any(k in next_line.lower() for k in ["sex", "nationality", "country"]):
                names = re.sub(r'[^a-zA-Z\s]', '', next_line).strip().replace(' ', '_').title()

    if not names and not surname:
        cap_words = re.findall(r'\b[A-Z]{3,20}\b', ocr_text)
        ignore_words = {"REPUBLIC", "SOUTH", "AFRICA", "NATIONAL", "IDENTITY", "CARD", "SEX", "CITIZEN"}
        filtered = [w.title() for w in cap_words if w not in ignore_words]
        if len(filtered) >= 2:
            names, surname = filtered[0], filtered[1]

    fullname = f"{names}_{surname}" if names and surname else (names or surname)
    return fullname


def parse_page_details(ocr_text):
    text_lower = ocr_text.lower()

    # 1. Standard 13-digit SA ID Extraction
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    id_num = id_match.group(1) if id_match else None

    # 2. Fuzzy Extraction for Handwritten ID numbers (e.g., ID.: 0102141451082)
    if not id_num:
        fuzzy_id_match = re.search(r'(?:ID|Identity)\s*[\.:\s]*([0-9OiIl|SsBQZ\s]{13,20})', ocr_text, re.IGNORECASE)
        if fuzzy_id_match:
            id_num = clean_handwritten_id(fuzzy_id_match.group(1))

    # TYPE 1: B-BBEE / Unemployment Affidavit
    if "bbbe" in text_lower or "unemployment" in text_lower or "affidavit" in text_lower:
        top_half_text = ocr_text[:1000]
        name = None

        line_match = re.search(r'I,?\s*([A-Za-z\s_]{3,50})(?:,|\s+ID|\s+hereby)', top_half_text, re.IGNORECASE)
        if line_match:
            raw_n = line_match.group(1).replace('_', '').strip()
            clean_n = re.sub(r'[^a-zA-Z\s]', '', raw_n).strip()
            if len(clean_n) > 2:
                name = re.sub(r'\s+', '_', clean_n).title()

        if not name:
            table_match = re.search(r'Full\s*names?\s*:\s*([A-Za-z\s]{3,60})', top_half_text, re.IGNORECASE)
            if table_match:
                raw = table_match.group(1).split('\n')[0].strip()
                clean = re.sub(r'[^a-zA-Z\s]', '', raw).strip()
                if len(clean) > 2:
                    name = re.sub(r'\s+', '_', clean).title()

        doc_type = "BBBEE-Unemployment-Affidavit" if "unemployment" in text_lower or "bbbe" in text_lower else "Affidavit"
        return name, id_num, doc_type

    # TYPE 2: Criminal Check Affidavit
    elif "criminal" in text_lower or "declaration" in text_lower:
        name = None
        name_match = re.search(r'Full\s*Names?\s*:\s*([A-Za-z\s]{3,50})', ocr_text, re.IGNORECASE)
        if name_match:
            raw = name_match.group(1).split('\n')[0].strip()
            clean = re.sub(r'[^a-zA-Z\s]', '', raw).strip()
            if len(clean) > 2:
                name = re.sub(r'\s+', '_', clean).title()

        return name, id_num, "Criminal-Check-Affidavit"

    # TYPE 3: Smart ID Card
    elif "identity card" in text_lower or "republic of south africa" in text_lower or "smart id" in text_lower:
        extracted_name = extract_smart_id_details(ocr_text)
        return extracted_name, id_num, "Smart-ID"

    return None, id_num, "Document"


def extract_ocr_from_single_pdf(page_pdf_bytes):
    extracted_text = ""
    with pdfplumber.open(io.BytesIO(page_pdf_bytes)) as pdf:
        if len(pdf.pages) > 0:
            extracted_text = pdf.pages[0].extract_text() or ""

    # Run OpenCV handwriting preprocessing if direct text extraction is short or empty
    if len(extracted_text.strip()) < 50:
        images = convert_from_bytes(page_pdf_bytes)
        for img in images:
            processed_img = preprocess_for_handwriting(img)
            # PSM 6 optimizes Tesseract for uniform blocks of text
            extracted_text += pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6') + "\n"

    return extracted_text


# 3. Streamlit Interface UI
st.title("📄 Candidate Document Splitter & Cross-Matcher")
st.markdown("Upload candidate documents. Multi-page PDFs are automatically split, preprocessed for handwriting recognition, and cross-matched.")

uploaded_files = st.file_uploader(
    "Upload Candidate Documents", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Split, Cross-Match & Package Files"):
        records = []
        zip_buffer = io.BytesIO()

        with st.spinner("Processing documents, running handwriting OCR, and cross-matching..."):
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()

                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)

                        pdf_pages_data = []
                        file_candidate_name = None
                        file_candidate_id = None

                        for page_idx in range(total_pages):
                            writer = PdfWriter()
                            writer.add_page(reader.pages[page_idx])
                            
                            single_page_io = io.BytesIO()
                            writer.write(single_page_io)
                            single_page_bytes = single_page_io.getvalue()

                            ocr_text = extract_ocr_from_single_pdf(single_page_bytes)
                            page_name, page_id, doc_type = parse_page_details(ocr_text)

                            if page_name and not file_candidate_name:
                                file_candidate_name = page_name
                            if page_id and not file_candidate_id:
                                file_candidate_id = page_id

                            pdf_pages_data.append({
                                "page_idx": page_idx,
                                "bytes": single_page_bytes,
                                "page_name": page_name,
                                "page_id": page_id,
                                "doc_type": doc_type
                            })

                        final_candidate_name = file_candidate_name or "Candidate"
                        final_candidate_id = file_candidate_id or "NoID"

                        for pdata in pdf_pages_data:
                            matched_name = pdata["page_name"] or final_candidate_name
                            matched_id = pdata["page_id"] or final_candidate_id
                            
                            page_suffix = f"_pg{pdata['page_idx']+1}" if total_pages > 1 else ""
                            new_filename = f"{matched_name}_{matched_id}_{pdata['doc_type']}{page_suffix}.pdf"

                            folder_name = f"{matched_name}_{matched_id}"
                            zip_path = f"{folder_name}/{new_filename}"

                            zip_file.writestr(zip_path, pdata["bytes"])

                            records.append({
                                "Source File": uploaded_file.name,
                                "Page Number": pdata["page_idx"] + 1,
                                "Renamed Filename": new_filename,
                                "Folder Path": folder_name,
                                "Extracted Name": matched_name,
                                "Identity Number": matched_id,
                                "Document Type": pdata["doc_type"]
                            })

                    else:
                        img = Image.open(io.BytesIO(file_bytes))
                        processed_img = preprocess_for_handwriting(img)
                        ocr_text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6')
                        name, id_number, doc_type = parse_page_details(ocr_text)

                        matched_name = name or "Candidate"
                        matched_id = id_number or "NoID"
                        ext = os.path.splitext(uploaded_file.name)[1]
                        new_filename = f"{matched_name}_{matched_id}_{doc_type}{ext}"

                        folder_name = f"{matched_name}_{matched_id}"
                        zip_path = f"{folder_name}/{new_filename}"

                        zip_file.writestr(zip_path, file_bytes)

                        records.append({
                            "Source File": uploaded_file.name,
                            "Page Number": 1,
                            "Renamed Filename": new_filename,
                            "Folder Path": folder_name,
                            "Extracted Name": matched_name,
                            "Identity Number": matched_id,
                            "Document Type": doc_type
                        })

                df = pd.DataFrame(records)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("Processing_Summary.csv", csv_bytes)

        st.success(f"Successfully processed {len(records)} page(s) across uploaded file(s)!")
        
        st.dataframe(df)

        st.download_button(
            label="📥 Download Split & Matched Files (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Cross_Matched_Candidate_Documents.zip",
            mime="application/zip"
        )
