import os
import json
import time
from pathlib import Path
from extract import extract_with_gemini

def process_all_resumes() -> None:
    upload_dir = Path("uploads")
    files = [f for f in upload_dir.iterdir() if f.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']]
    
    print(f"[INFO] Initializing batch document ingestion for {len(files)} files.\n")
    results = []
    
    for i, filepath in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {filepath.name}")
        
        candidate_data = extract_with_gemini(str(filepath))
        candidate_data["source_file"] = filepath.name
        
        results.append(candidate_data)
        print(f"    [SUCCESS] Entity: {candidate_data.get('name')} | Skills: {candidate_data.get('skills')}")
        
        # Throttling to remain within free tier rate boundaries
        if i < len(files) - 1:
            time.sleep(6)
            
    os.makedirs("data", exist_ok=True)
    output_path = "data/candidates.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[INFO] Batch execution complete. Output serialized to {output_path}")

if __name__ == "__main__":
    process_all_resumes()