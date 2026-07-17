import json
from pathlib import Path
from ingest import extract_text
from extract import extract_structured

for filepath in Path("uploads").iterdir():
    print(f"\n{'='*60}\n{filepath.name}\n{'='*60}")
    raw_text = extract_text(str(filepath))
    print("\n--- Raw OCR/extracted text (first 600 chars) ---\n", raw_text[:600])
    print("\n--- Raw LLM output (before any post-processing) ---")
    print(json.dumps(extract_structured(raw_text), indent=2))