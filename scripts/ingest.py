# import os
# import re
# from pathlib import Path
# import fitz
# from docx import Document
# import pytesseract
# from PIL import Image

# # Your local Tesseract path


# def extract_pdf_text(filepath):
#     try:
#         doc = fitz.open(filepath)
#         return "".join(page.get_text() for page in doc)
#     except Exception as e:
#         print(f"Error parsing PDF {filepath}: {e}")
#         return ""

# def extract_docx_text(filepath):
#     try:
#         doc = Document(filepath)
#         text = "\n".join(p.text for p in doc.paragraphs)
#         for table in doc.tables:
#             for row in table.rows:
#                 text += "\n" + " ".join(cell.text for cell in row.cells)
#         return text
#     except Exception as e:
#         print(f"Error parsing DOCX {filepath}: {e}")
#         return ""

# def extract_image_text(filepath):
#     try:
#         return pytesseract.image_to_string(Image.open(filepath))
#     except Exception as e:
#         print(f"Error executing OCR on {filepath}: {e}")
#         return ""

# def extract_text(filepath: str) -> str:
#     ext = Path(filepath).suffix.lower()
#     if ext == ".pdf":
#         return extract_pdf_text(filepath)
#     elif ext == ".docx":
#         return extract_docx_text(filepath)
#     elif ext in (".jpg", ".jpeg", ".png"):
#         return extract_image_text(filepath)
#     return ""

# def extract_contact_info(text):
#     if not text:
#         return None, None
#     phone = None
    
#     # Smarter phone regex with \b (word boundaries) to stop it from grabbing zip codes
#     PHONE_PATTERNS = [
#         r'\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', 
#         r'\b(?:\+?91)?[6-9]\d{9}\b',                                    
#     ]
#     for pattern in PHONE_PATTERNS:
#         match = re.search(pattern, text)
#         if match:
#             phone = match.group(0)
#             break

#     # Smarter email regex that catches OCR errors like "NAKAMURAGEXAMPLE.COM" 
#     # or ignores stray brackets like "(@cxample.com"
#     email_clean = re.sub(r'[()\[\]{}]', '', text) # Strip OCR brackets first
#     email_match = re.search(r'[a-zA-Z0-9._%+-]+(?:@|G|g)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_clean)
#     email = email_match.group(0).replace('G', '@').replace('g', '@') if email_match else None
    
#     return phone, email

# def extract_name_by_layout(filepath: str, validator) -> str | None:
#     if not str(filepath).lower().endswith(".pdf"):
#         return None
#     try:
#         from extract import TAXONOMY # Import here to avoid circular imports
#         doc = fitz.open(filepath)
#         page = doc[0]
#         blocks = page.get_text("dict").get("blocks", [])
#         page_height = page.rect.height

#         candidates = []
#         for block in blocks:
#             if "lines" not in block:
#                 continue
#             for line in block["lines"]:
#                 for span in line["spans"]:
#                     text = span["text"].strip()
#                     if not text:
#                         continue
#                     y_pos = span["bbox"][1]
#                     if y_pos > page_height * 0.25:
#                         continue
#                     candidates.append((span["size"], y_pos, text))

#         candidates.sort(key=lambda c: (-c[0], c[1]))
#         for size, y, text in candidates:
#             if validator(text, TAXONOMY):
#                 return text
#         return None
#     except Exception as e:
#         print(f"Layout name extraction failed for {filepath}: {e}")
#         return None