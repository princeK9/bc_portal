import os
import json
import psycopg2
from psycopg2.extras import execute_values

# Database connection configuration
# Replace placeholder coordinates from supabase
import os
# Supabase Transaction Pooler Connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "6543"),
}

def migrate_json_to_db() -> None:
    json_path = "data/candidates.json"
    
    if not os.path.exists(json_path):
        print(f"[ERROR] Source artifact not found at: {json_path}")
        return

    with open(json_path, "r") as f:
        candidates = json.load(f)

    print(f"[INFO] Initializing migration sequence for {len(candidates)} records.")

    try:
        # Establish connection matrix
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Construction query utilizing positional parametrization
        insert_query = """
            INSERT INTO candidates (name, raw_title, experience_years, phone, email, skills, source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                name = EXCLUDED.name,
                raw_title = EXCLUDED.raw_title,
                experience_years = EXCLUDED.experience_years,
                phone = EXCLUDED.phone,
                skills = EXCLUDED.skills,
                source_file = EXCLUDED.source_file;
        """

        for record in candidates:
            cursor.execute(insert_query, (
                record.get("name"),
                record.get("raw_title"),
                record.get("experience_years"),
                record.get("phone"),
                record.get("email"),
                record.get("skills"),  # Psycopg2 maps Python lists to SQL arrays automatically
                record.get("source_file")
            ))

        conn.commit()
        print(f"[SUCCESS] Database state synchronized. {len(candidates)} entities committed.")

    except Exception as e:
        print(f"[ERROR] Transaction failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_json_to_db()