def parse_document(ocr_text):
    text_lower = ocr_text.lower()

    # SPECIFIC HANDLER: BBBEE Certification / Unemployment Affidavit
    if "bbbe certification" in text_lower or "affidavit to confirm unemployment" in text_lower or "unemployment" in text_lower:
        name = None
        id_num = None

        # 1. Primary Extraction: Table row for "Full name:" (Captures complete name including middle names)
        table_name_match = re.search(r'Full\s*name\s*:\s*([A-Za-z\s]{3,60})', ocr_text, re.IGNORECASE)
        if table_name_match:
            raw_name = table_name_match.group(1).split('\n')[0].strip()
            clean_name = re.sub(r'[^a-zA-Z\s]', '', raw_name).strip()
            if len(clean_name) > 2:
                name = re.sub(r'\s+', '_', clean_name).title()

        # 2. Secondary Extraction: Opening line "I, <Name> ..., ID.: <ID>"
        if not name:
            line_match = re.search(r'I,\s*([A-Za-z\s_]+?)(?:,|\s+ID)', ocr_text, re.IGNORECASE)
            if line_match:
                raw_n = line_match.group(1).replace('_', '').strip()
                if len(raw_n) > 2:
                    name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', raw_n)).title()

        # 3. ID Number Extraction (Handles "ID.: 9205120475088" or standalone 13-digit pattern)
        id_match = re.search(r'(?:ID\.?:?|Identity\s*Number\s*:?)\s*(\d{13})', ocr_text, re.IGNORECASE)
        if id_match:
            id_num = id_match.group(1)
        else:
            fallback_id = re.search(r'\b\d{13}\b', ocr_text)
            if fallback_id:
                id_num = fallback_id.group(0)

        clean_name = name or "Unknown_Candidate"
        clean_id = id_num or "NoID"
        return clean_name, clean_id, "BBBEE-Unemployment-Affidavit"

    # --- Keep remaining document types below ---
    elif "criminal record status" in text_lower or "declaration of criminal" in text_lower:
        name, id_num = extract_criminal_affidavit_details(ocr_text)
        return name, id_num, "Criminal-Check-Affidavit"

    elif "national identity card" in text_lower or "republic of south africa" in text_lower:
        # Smart ID processing...
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        return "Smart_ID_Candidate", (id_match.group(0) if id_match else None), "Smart-ID"

    # General Fallback
    id_match = re.search(r'\b\d{13}\b', ocr_text)
    return None, (id_match.group(0) if id_match else None), "Document"
