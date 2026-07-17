import os
import time
from pathlib import Path
from google import genai
from google.genai import types

# 1. Setup using the NEW 2026 SDK
# Best practice: Fetch from environment variable, but hardcoding for this test is fine if you don't share it!
client = genai.Client(api_key="API_key")

def test_cloud_extraction(image_path, max_retries=3):
    print(f"🚀 Sending {image_path} to Gemini 3.5 Flash Vision...")
    
    prompt = """You are a strict resume extraction AI. Read this resume image and return ONLY valid JSON.
    SCHEMA:
    - "name": (string or null) The candidate's real full name.
    - "raw_title": (string or null) The candidate's primary job title.
    - "phone": (string or null) The candidate's phone number.
    - "email": (string or null) The candidate's email address.
    """
    
    image_bytes = Path(image_path).read_bytes()
    
    # Fault Tolerance: Exponential Backoff Retry Loop
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash', 
                contents=[
                    prompt, 
                    types.Part.from_bytes(data=image_bytes, mime_type='image/png')
                ],
                config=types.GenerateContentConfig(
                    temperature=0, 
                    response_mime_type="application/json", 
                )
            )
            
            print("\n✅ Extraction Complete:")
            print(response.text)
            return # Exit the function successfully
            
        except Exception as e:
            error_msg = str(e)
            # Catch server overload (503) or rate limits (429)
            if '503' in error_msg or '429' in error_msg:
                wait_time = 2 ** attempt # Waits 1s, then 2s, then 4s
                print(f"⚠️ Server busy (Attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # If it's a different error (like a bad API key), crash immediately
                print(f"❌ Fatal Error: {e}")
                break
                
    print("\n❌ Max retries reached. API is completely unresponsive.")

if __name__ == "__main__":
    test_cloud_extraction("uploads/001.png")