import os
import re
import io
import zipfile
import pandas as pd
import pytesseract
import pdfplumber
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader, PdfWriter

# 1. Page Configuration
st.set_page_config(
    page_title="Candidate Document Splitter & Renamer",
    page_icon="📄",
    layout="wide"
)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 2. Document Extraction & Cross-Matching Logic
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
    """
    Extracts document type, candidate name, and ID from an individual page.
    For B-BBEE affidavits, it focuses explicitly on the top half of the text.
    """
    text_lower = ocr_text.lower()

    # Extract 13-digit SA ID number (handles printed or clear OCR handwritten digits)
    id_match = re.search(r'\b(\d{13})\b', ocr_text)
    id_num = id_match.group(1) if id_match else None

    # TYPE 1: B-BBEE / Unemployment Affidavit
    if "bbbe" in text_lower or "unemployment" in text_lower or "affidavit" in text_lower:
        # Focus scan on the upper portion of the page text (first ~800 chars)
        top_half_text = ocr_text[:800]
        name = None

        # 1. Primary: Look for "Full name:" or "I, <Name>" in the top region
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

    # Fallback
    return None, id_num, "Document"


def extract_ocr_from_single_pdf(page_pdf_bytes):
    extracted_text = ""
    with pdfplumber.open(io.BytesIO(page_pdf_bytes)) as pdf:
        if len(pdf.pages) > 0:
            extracted_text = pdf.pages[0].extract_text() or ""

    if not extracted_text.strip():
        # Fallback to OCR engine
        images = convert_from_bytes(page_pdf_bytes)
        for img in images:
            extracted_text += pytesseract.image_to_string(img) + "\n"

    return extracted_text


# 3. Streamlit Interface UI
st.title("📄 Candidate Document Splitter & Cross-Matcher")
st.markdown("Upload documents. Multi-page PDFs are automatically split, analyzed, and cross-matched to apply candidate details across all pages in the file.")

uploaded_files = st.file_uploader(
    "Upload Candidate Documents", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Split, Cross-Match & Package Files"):
        records = []
        zip_buffer = io.BytesIO()

        with st.spinner("Processing documents, extracting metadata, and cross-matching candidates..."):
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()

                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(io.BytesIO(file_bytes))
                        total_pages = len(reader.pages)

                        # PHASE 1: Extract details per page & establish PDF-wide candidate profile
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

                            # Record best candidate metadata found across pages in this PDF
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

                        # PHASE 2: Cross-check & backfill missing metadata using PDF-level profile
                        final_candidate_name = file_candidate_name or "Candidate"
                        final_candidate_id = file_candidate_id or "NoID"

                        for pdata in pdf_pages_data:
                            # Prefer page-specific details; fall back to file-level matched profile
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
                        # Processing individual image file
                        img = Image.open(io.BytesIO(file_bytes))
                        ocr_text = pytesseract.image_to_string(img)
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

                # Append processing manifest CSV to output zip
                df = pd.DataFrame(records)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("Processing_Summary.csv", csv_bytes)

        st.success(f"Successfully processed {len(records)} page(s) across uploaded file(s) with cross-page matching!")
        
        st.dataframe(df)

        st.download_button(
            label="📥 Download Split & Matched Files (.ZIP)",
            data=zip_buffer.getvalue(),
            file_name="Cross_Matched_Candidate_Documents.zip",
            mime="application/zip"
        )
