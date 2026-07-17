# from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from sentence_transformers import SentenceTransformer

import zipfile
import io
import docx
import json
import time
from fastapi import Query
import google.generativeai as genai

# replace
# genai.configure(api_key="API_key")
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {"response_mime_type": "application/json"}
# gemini_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config=generation_config)
app = FastAPI(title="Blue Collar Portal AI Engine")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Booting up AI Engine...")
model = SentenceTransformer('all-MiniLM-L6-v2')

import os
# Supabase Transaction Pooler Connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "6543"),
}

def calibrate_score(raw_sim: float) -> tuple[float, str]:
    """
    Normalizes raw cosine similarity (typically 0.2 - 0.8 range for MiniLM)
    into a human-readable 0-100% score and assigns a confidence level.
    """
    normalized = min(max((raw_sim - 0.2) / 0.6 * 100, 0), 100)
    
    if normalized >= 70:
        confidence = "High Match"
    elif normalized >= 40:
        confidence = "Moderate Match"
    else:
        confidence = "Low Match"
        
    return round(normalized, 1), confidence

@app.get("/search")
def search_candidates(query: str, limit: int = 5):
    try:
        # 1. Convert search query into vector representation
        query_vector = model.encode(query).tolist()
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 2. Query pgvector using Cosine Distance (<=> operator)
        # sql = """
        #     SELECT id, name, raw_title, experience_years, skills, 
        #            1 - (embedding <=> %s::vector) AS raw_similarity 
        #     FROM candidates 
        #     WHERE name IS NOT NULL
        #     ORDER BY embedding <=> %s::vector 
        #     LIMIT %s;
        # """
        sql = """
            SELECT id, name, raw_title, experience_years, skills, 
                   1 - (embedding <=> %s::vector) AS raw_similarity,
                   phone, email 
            FROM candidates 
            WHERE name IS NOT NULL
            ORDER BY embedding <=> %s::vector 
            LIMIT %s;
        """
        
        cursor.execute(sql, (str(query_vector), str(query_vector), limit))
        results = cursor.fetchall()
        
        # 3. Format and calibrate output
        matches = []
        for row in results:
            raw_sim = float(row[5])
            calibrated_score, confidence_tier = calibrate_score(raw_sim)
            
            matches.append({
                "id": row[0],
                "name": row[1],
                "title": row[2],
                "experience_years": row[3],
                "skills": row[4],
                "raw_similarity": round(raw_sim, 3),
                "match_score": calibrated_score,
                "confidence_tier": confidence_tier,
                "phone": row[6],
                "email": row[7]
            })
            
        return {
            "query": query, 
            "total_evaluated": len(results),
            "results": matches
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


def extract_candidate_data(file_bytes, file_name, extension):
    prompt = """
    You are an expert technical recruiter AI. Extract the following information from this resume and return ONLY a valid JSON object. 
    Do not include markdown blocks or any other text.
    {
        "name": "Full Name",
        "email": "Email Address",
        "phone": "Phone Number",
        "skills": ["Skill 1", "Skill 2", "Skill 3"],
        "experience_years": Total years of experience as an integer (e.g., 5),
        "title": "Current or most recent job title"
    }
    If any field is missing, use null for strings/integers, or an empty array [] for skills.
    """
    
    try:
        if extension == '.docx':
            # Route A: Word Document Text Extraction
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n".join([para.text for para in doc.paragraphs])
            response = gemini_model.generate_content([prompt, full_text])
        else:
            # Route B: Multimodal Vision for PDFs and Images
            mime_types = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png'
            }
            blob = {
                "mime_type": mime_types[extension],
                "data": file_bytes
            }
            response = gemini_model.generate_content([prompt, blob])
            
        return json.loads(response.text)
    
    except Exception as e:
        print(f"Failed to process {file_name} with Gemini: {e}")
        return None


@app.post("/upload-resumes/")
async def upload_resumes(file: UploadFile = File(...)):
    global conn, cursor
    # 1. Read the uploaded ZIP file directly into RAM
    zip_bytes = await file.read()
    
    valid_file_count = 0
    extracted_names = []
    
    # 2. Define the multimodal formats we support
    allowed_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.docx')

    # 3. Open the ZIP file from the memory buffer
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_ref:
            # Loop through every file inside the ZIP
            for file_name in zip_ref.namelist():
                # Ignore hidden macOS files (like __MACOSX or .DS_Store)
                if file_name.startswith('__MACOSX') or file_name.split('/')[-1].startswith('.'):
                    continue
                    
                # Check if it matches our allowed multimodal extensions
                if file_name.lower().endswith(allowed_extensions):
                    valid_file_count += 1
                    extracted_names.append(file_name)
                    
                    # Read the individual file into RAM
                    # file_data = zip_ref.read(file_name)
                    # Read the individual file into RAM
                    file_data = zip_ref.read(file_name)
                    extension = '.' + file_name.lower().split('.')[-1]
                    
                    # 1. Send the file to Gemini 3.5 Flash
                    candidate_data = extract_candidate_data(file_data, file_name, extension)
                    
                    if candidate_data:
                        # 2. Prevent DB Crashes from missing data
                        email = candidate_data.get('email')
                        phone = candidate_data.get('phone')
                        title = candidate_data.get('title', 'Unknown Title')
                        skills = candidate_data.get('skills', [])
                        
                        # 3. Create the text string for the vector embedding
                        # (We embed the title and skills together for semantic meaning)
                        text_to_embed = f"{title}. Skills: {', '.join(skills)}"
                        
                        # 4. Generate the 384-dimensional vector using your existing model
                        # (Assuming your sentence-transformer model is named 'model')
                        embedding = model.encode(text_to_embed).tolist()
                        
                        import psycopg2
                        conn = psycopg2.connect(**DB_CONFIG)
                        cursor = conn.cursor()

                        # 5. Push to Supabase pgvector
                        try:
                            # Using your existing database cursor connection
                            sql = """
                                INSERT INTO candidates (name, raw_title, experience_years, skills, phone, email, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            """
                            cursor.execute(sql, (
                                candidate_data.get('name', 'Anonymous Profile'),
                                title,
                                candidate_data.get('experience_years', 0),
                                skills,
                                phone,
                                email,
                                str(embedding)
                            ))
                            conn.commit()
                        except Exception as db_error:
                            print(f"Database error for {email}: {db_error}")
                            conn.rollback() # Rollback on duplicate email/error so the loop doesn't crash
                        
                        finally:
                            # Ensure the connection closes cleanly
                            if 'conn' in locals():
                                cursor.close()
                                conn.close()

                    print(f"Waiting 15 seconds before processing the next file...")
                    time.sleep(15)

        return {
            "message": "ZIP file processed successfully in RAM.",
            "total_valid_files_found": valid_file_count,
            "files_queued": extracted_names
        }
        
    except zipfile.BadZipFile:
        return {"error": "Invalid file format. Please upload a valid .zip archive."}
    

    
@app.get("/health")
async def health_check():
    return {"status": "online"}