import os
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TAXONOMY_SYNONYMS = [
    "Electrician", "Plumber", "Carpenter", "Welder", "HVAC Technician",
    "Mason/Bricklayer", "Construction Laborer", "Auto Mechanic", "Diesel Mechanic",
    "Driver (Heavy/Light Vehicle)", "Machine/Plant Operator", "Engineering/Lab Technician",
    "Fabricator/Metal Worker", "Fitter", "Cook/Chef", "Food Preparation/Service",
    "Housekeeping/Cleaner", "Security Guard", "Gardener/Landscaper",
    "General Technician/Helper", "Healthcare Support", "Social/Clinical Worker", "Other"
]

class CandidateSchema(BaseModel):
    name: Optional[str] = Field(description="Candidate's full legal name")
    raw_title: Optional[str] = Field(description="Primary job title or headline")
    experience_years: Optional[int] = Field(description="Total years of experience as an integer")
    phone: Optional[str] = Field(description="Contact phone number")
    email: Optional[str] = Field(description="Contact email address")
    skills: List[str] = Field(description="List of matched skill categories from taxonomy")

def extract_with_gemini(filepath: str, max_retries: int = 6) -> dict:
    file_ext = Path(filepath).suffix.lower()
    mime_type = 'application/pdf' if file_ext == '.pdf' else f'image/{file_ext.strip(".")}'
    
    prompt = f"""Extract resume fields strictly adhering to the provided schema. 
Map candidate skills to one or more of these standard categories: {TAXONOMY_SYNONYMS}. 
If no category fits perfectly, use ["Other"]."""
    
    file_bytes = Path(filepath).read_bytes()
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash', 
                contents=[
                    prompt, 
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=CandidateSchema, 
                )
            )
            return CandidateSchema.model_validate_json(response.text).model_dump()
            
        except Exception as e:
            error_msg = str(e)
            if '503' in error_msg or '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg:
                wait_time = (2 ** attempt) + 2  
                print(f"      [WARNING] API busy/Rate limit. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"      [ERROR] API request failed for {filepath}: {e}")
                break
                
    return {
        "name": None, 
        "raw_title": None, 
        "experience_years": None, 
        "phone": None, 
        "email": None, 
        "skills": ["Other"]
    }