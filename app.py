import re

def parse_document_all_types(ocr_text):
    """
    Unified extractor supporting:
    1. Criminal Record Affidavits
    2. SA National ID Cards
    3. B-BBEE / Unemployment Affidavits
    4. Senior Certificates
    """
    text_lower = ocr_text.lower()
    
    # --- TYPE 1: Criminal Record Affidavit ---
    if "criminal record status" in text_lower or "declaration of criminal" in text_lower:
        name, id_num = extract_criminal_affidavit_details(ocr_text)
        return name, id_num, "Criminal-Check-Affidavit"

    # --- TYPE 2: SA Smart ID Card ---
    elif "national identity card" in text_lower or "republic of south africa" in text_lower:
        surname, names, id_num = None, None, None
        lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
        
        for i, line in enumerate(lines):
            l_lower = line.lower()
            if "surname:" in l_lower or l_lower == "surname":
                if i + 1 < len(lines):
                    surname = re.sub(r'[^a-zA-Z]', '', lines[i + 1]).title()
            elif "names:" in l_lower or l_lower == "names":
                if i + 1 < len(lines):
                    names = re.sub(r'[^a-zA-Z\s]', '', lines[i + 1]).strip().replace(' ', '_').title()
                    
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        
        fullname = f"{names}_{surname}" if names and surname else (names or surname)
        return fullname, id_num, "ID"

    # --- TYPE 3: Unemployment / B-BBEE Affidavit ---
    elif "unemployment" in text_lower or "bbbee" in text_lower:
        # Check inline sentence: "I, [Name] , ID.: [ID]"
        line_match = re.search(r'I,?\s*([A-Za-z\s]{3,40})\s*,?\s*ID', ocr_text, re.IGNORECASE)
        name = None
        if line_match:
            raw_n = line_match.group(1).strip()
            if len(raw_n) > 2:
                name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', raw_n)).title()
                
        # Check form field fallback: "Full name:"
        if not name:
            fn_match = re.search(r'Full\s*name\s*:\s*([A-Za-z\s]{3,40})', ocr_text, re.IGNORECASE)
            if fn_match:
                name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', fn_match.group(1))).title()
                
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        
        doc_type = "Unemployment-Affidavit" if "unemployment" in text_lower else "BBBEE-Affidavit"
        return name, id_num, doc_type

    # --- TYPE 4: Senior Certificate ---
    elif "senior certificate" in text_lower or "awarded to" in text_lower:
        cert_match = re.search(r'(?:awarded\s+to|certify\s+that)\s+([A-Z\s]{3,40})', ocr_text, re.IGNORECASE)
        name = None
        if cert_match:
            name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z\s]', '', cert_match.group(1))).title()
            
        id_match = re.search(r'\b\d{13}\b', ocr_text)
        id_num = id_match.group(0) if id_match else None
        return name, id_num, "SeniorCertificate"

    # Fallback default
    id_match = re.search(r'\b\d{13}\b', ocr_text)
    return None, (id_match.group(0) if id_match else None), "Document"
