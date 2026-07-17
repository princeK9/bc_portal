import psycopg2
from sentence_transformers import SentenceTransformer

# Your Supabase Connection
import os
# Supabase Transaction Pooler Connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "6543"),
}

def generate_and_store_vectors():
    print("[INFO] Loading the local embedding model (all-MiniLM-L6-v2)...")
    # This model creates 384-dimensional vectors. It will download once and cache locally.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Fetch candidates that don't have an embedding yet
        cursor.execute("SELECT id, raw_title, experience_years, skills FROM candidates WHERE embedding IS NULL")
        rows = cursor.fetchall()

        if not rows:
            print("[INFO] All candidates are already vectorized!")
            return

        print(f"[INFO] Found {len(rows)} candidates needing vectors. Processing...")

        for row in rows:
            record_id, raw_title, exp_years, skills = row
            
            # 1. Create a rich text summary of the candidate to feed to the AI
            skills_text = ", ".join(skills) if skills else "None"
            semantic_text = f"Title: {raw_title}. Experience: {exp_years} years. Skills: {skills_text}."
            
            # 2. Convert the text into a 384-dimensional vector array
            vector = model.encode(semantic_text)
            
            # PostgreSQL pgvector accepts vectors as standard string lists (e.g., "[0.1, 0.2, ...]")
            vector_string = str(vector.tolist())
            
            # 3. Update the candidate's row with their new vector
            cursor.execute(
                "UPDATE candidates SET embedding = %s WHERE id = %s",
                (vector_string, record_id)
            )
            print(f"   [SUCCESS] Generated & stored vector for Candidate ID {record_id}")

        conn.commit()
        print(f"\n🎉 Day 5 Complete! {len(rows)} vectors safely stored in Supabase.")

    except Exception as e:
        print(f"[ERROR] Vectorization failed: {e}")
        if 'conn' in locals(): conn.rollback()
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    generate_and_store_vectors()