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
    page_title="Candidate Document Pack Splitter",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Text Preprocessing & Cleaning
def preprocess_for_handwriting(image):
    """Enhances image contrast for handwriting and faint text OCR."""
    open_cv_image = np.array(image.convert('L'))
    resized = cv2.resize(open_cv_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def clean_handwritten_id(id_candidate_str):
    """Fixes digit/letter confusion in handwritten 13-digit SA IDs."""
    replacements = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1', '|': '1',
        'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2'
    }
    cleaned = id_candidate_str
    for char, digit in replacements.items():
        cleaned = cleaned.replace(char, digit)
    
    digits_only = re.sub(r'\D', '', cleaned)
    return digits_only if len(digits_only) == 13 else None


def sanitize_candidate_name(raw_name):
    """Filters out form headings and OCR noise from candidate names."""
    if not raw_name:
        return None

    # Common form noise and template headers to ignore
    noise_patterns = [
        r'declaration', r'criminal', r'record', r'status', r'affidavit',
        r'administered', r'capaciti', r'programme', r'participant', r'confirm',
        r'unemployment', r'certification', r'republic', r'south africa', r'identity'
    ]

    cleaned_str = re.sub(r'[^a-zA-Z\s]', '', raw_name).strip()
    words = cleaned_str.split()

    # Reject if text matches known header patterns or contains fewer than 2 words
    if len(words) < 2:
        return None

    lower_check = cleaned_str.lower()
    for noise in noise_patterns:
        if noise in lower_check:
            return None

    return "_".join([w.capitalize() for w in words[:4]])


# 3. High-Accuracy Document & Anchor Extractors
def extract_anchor_document_details(ocr_text):
    """
    Extracts Candidate Name & ID from official anchor documents:
    Smart ID, Green ID Book, and Senior Certificates (Matric).
    """
    text_lower = ocr_text.lower()
    is_anchor = False
    doc_type = "Document"

    # Detect Official Anchor Document Types
    if "national identity card" in text_lower or "republic of south africa" in text_lower or "identity document" in text_lower:
        is_anchor = True
        doc_type = "ID-Document"
    elif "senior certificate" in text_lower or "national senior certificate" in text_lower or "matric" in text_lower or "umalusi" in text_lower:
        is_anchor = True
        doc_type = "Senior-Certificate"

    # Extract 13-Digit SA ID Number
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    extracted_id = id_match.group(1) if id_match else None

    # Name Extraction for ID Documents
    extracted_name = None
    if is_anchor:
        surname, names = None, None
        lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

        for i, line in enumerate(lines):
            l_lower = line.lower()
            if "surname" in l_lower and i + 1 < len(lines):
                next_l = lines[i + 1]
                if not any(k in next_l.lower() for k in ["names", "sex", "nationality", "date"]):
                    surname = re.sub(r'[^a-zA-Z]', '', next_l).strip()
            elif "names" in l_lower and i + 1 < len(lines):
                next_l = lines[i + 1]
                if not any(k in next_l.lower() for k in ["sex", "nationality", "country"]):
                    names = re.sub(r'[^a-zA-Z\s]', '', next_l).strip()

        if surname and names:
            extracted_name = sanitize_candidate_name(f"{names} {surname}")
        elif names or surname:
            extracted_name = sanitize_candidate_name(names or surname)

        # Senior Certificate Name Pattern ("This is to certify that <NAME>...")
        if not extracted_name and "Senior-Certificate" in doc_type:
            cert_match = re.search(r'(?:certify that|awarded to)\s+([A-Za-z\s]{5,60})', ocr_text, re.IGNORECASE)
            if cert_match:
                extracted_name = sanitize_candidate_name(cert_match.group(1))

    return is_anchor, extracted_name, extracted_id, doc_type


def parse_page_details(ocr_text):
    """Parses individual non-anchor pages (Affidavits, Supporting Docs)."""
    text_lower = ocr_text.lower()

    # Identify document type
    if "bbbe" in text_lower or "unemployment" in text_lower:
        doc_type = "BBBEE-Unemployment-Affidavit"
    elif "criminal" in text_lower or "declaration of criminal" in text_lower:
        doc_type = "Criminal-Check-Affidavit"
    elif "affidavit" in text_lower:
        doc_type = "Affidavit"
    else:
        doc_type = "Document"

    # Extract ID
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    id_num = id_match.group(1) if id_match else None

    if not id_num:
        fuzzy_id = re.search(r'(?:ID|Identity)\s*[\.:\s]*([0-9OiIl|SsBQZ\s]{13,20})', ocr_text, re.IGNORECASE)
        if fuzzy_id:
            id_num = clean_handwritten_id(fuzzy_id.group(1))

    # Name fallback from top-half form lines
    name = None
    line_match = re.search(r'I,?\s*([A-Za-z\s]{4,40})(?:,|\s+ID|\s+hereby)', ocr_text[:800], re.IGNORECASE)
    if line_match:
        name = sanitize_candidate_name(line_match.group(1))

    return name, id_num, doc_type


def extract_ocr_from_single_pdf(page_pdf_bytes):
    extracted_text = ""
    with pdfplumber.open(io.BytesIO(page_pdf_bytes)) as pdf:
        if len(pdf.pages) > 0:
            extracted_text = pdf.pages[0].extract_text() or ""

    if len(extracted_text.strip()) < 40:
        images = convert_from_bytes(page_pdf_bytes)
        for img in images:
            proc_img = preprocess_for_handwriting(img)
            extracted_text += pytesseract.image_to_string(proc_img, config='--oem 3 --psm 6') + "\n"

    return extracted_text


# 4. Streamlit Application Core
st.title("📄 Candidate Pack Splitter & Renamer")
st.markdown("Upload candidate PDF packs. The app detects **ID Documents & Senior Certificates** to establish candidate identity and renames all files and folders in the PDF pack.")

uploaded_files = st.file_uploader(
    "Upload Candidate PDF Packs", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Candidate Packs"):
        records = []
        zip_buffer = io.BytesIO()

        with st.spinner("Processing PDF candidate packs and matching anchor documents..."):
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()

                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)

                        pdf_pages_data = []
                        anchor_candidate_name = None
                        anchor_candidate_id = None
                        
                        fallback_candidate_name = None
                        fallback_candidate_id = None

                        # PASS 1: Scan entire PDF for anchor documents (ID / Senior Cert)
                        for page_idx in range(total_pages):
                            writer = PdfWriter()
                            writer.add_page(reader.pages[page_idx])
                            
                            single_page_io = io.BytesIO()
                            writer.write(single_page_io)
                            single_page_bytes = single_page_io.getvalue()

                            ocr_text = extract_ocr_from_single_pdf(single_page_bytes)
                            is_anchor, a_name, a_id, doc_type = extract_anchor_document_details(ocr_text)

                            if not is_anchor:
                                page_name, page_id, doc_type = parse_page_details(ocr_text)
                                if page_name and not fallback_candidate_name:
                                    fallback_candidate_name = page_name
                                if page_id and not fallback_candidate_id:
                                    fallback_candidate_id = page_id
                            else:
                                if a_name and not anchor_candidate_name:
                                    anchor_candidate_name = a_name
                                if a_id and not anchor_candidate_id:
                                    anchor_candidate_id = a_id

                            pdf_pages_data.append({
                                "page_idx": page_idx,
                                "bytes": single_page_bytes,
                                "doc_type": doc_type,
                                "ocr_name": a_name if is_anchor else page_name,
                                "ocr_id": a_id if is_anchor else page_id
                            })

                        # PASS 2: Establish Single Master Identity for the PDF Pack
                        final_pack_name = anchor_candidate_name or fallback_candidate_name or "Candidate"
                        final_pack_id = anchor_candidate_id or fallback_candidate_id or "NoID"

                        # PASS 3: Apply Master Identity across every page file and directory
                        for pdata in pdf_pages_data:
                            page_suffix = f"_pg{pdata['page_idx']+1}" if total_pages > 1 else ""
                            new_filename = f"{final_pack_name}_{final_pack_id}_{pdata['doc_type']}{page_suffix}.pdf"
                            folder_name = f"{final_pack_name}_{final_pack_id}"
                            zip_path = f"{folder_name}/{new_filename}"

                            zip_file.writestr(zip_path, pdata["bytes"])

                            records.append({
                                "Source PDF Pack": uploaded_file.name,
                                "Page Number": pdata["page_idx"] + 1,
                                "Renamed Filename": new_filename,
                                "Folder Path": folder_name,
                                "Candidate Name": final_pack_name,
                                "Identity Number": final_pack_id,
                                "Document Type": pdata["doc_type"]
                            })

                    else:
                        img = Image.open(io.BytesIO(file_bytes))
                        proc_img = preprocess_for_handwriting(img)
                        ocr_text = pytesseract.image_to_string(proc_img, config='--oem 3 --psm 6')
                        
                        is_anchor, name, id_num, doc_type = extract_anchor_document_details(ocr_text)
                        if not is_anchor:
                            name, id_num, doc_type = parse_page_details(ocr_text)

                        matched_name = name or "Candidate"
                        matched_id = id_num or "NoID"
                        ext = os.path.splitext(uploaded_file.name)[1]
                        
                        new_filename = f"{matched_name}_{matched_id}_{doc_type}{ext}"
                        folder_name = f"{matched_name}_{matched_id}"
                        zip_path = f"{folder_name}/{new_filename}"

                        zip_file.writestr(zip_path, file_bytes)

                        records.append({
                            "Source PDF Pack": uploaded_file.name,
                            "Page Number": 1,
                            "Renamed Filename": new_filename,
                            "Folder Path": folder_name,
                            "Candidate Name": matched_name,
                            "Identity Number": matched_id,
                            "Document Type": doc_type
                        })

                df = pd.DataFrame(records)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("Processing_Summary.csv", csv_bytes)

        st.success(f"Processed {len(records)} document page(s) into candidate packs!")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Structured Candidate Packs (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Candidate_Packs_Processed.zip",
            mime="application/zip"
        )
