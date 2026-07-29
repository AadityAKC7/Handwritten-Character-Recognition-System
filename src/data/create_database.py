# create_database.py
import sqlite3
import hashlib

def create_database():
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )
    ''')
    
    # Create Recognition History table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recognition_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        predicted_character TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        actual_character TEXT,
        is_correct INTEGER,
        recognition_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processing_time_ms REAL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Optional: Create a table for model performance tracking
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        accuracy REAL,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 0
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database created successfully!")
    print("Tables: users, recognition_history, model_metadata")

def create_sample_user():
    """Optional: Create a sample user for testing"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create a simple hash for password "password123"
    # In production, use proper hashing like bcrypt
    sample_password = hashlib.sha256("password123".encode()).hexdigest()
    
    try:
        cursor.execute('''
        INSERT INTO users (username, email, password_hash, full_name)
        VALUES (?, ?, ?, ?)
        ''', ('demo_user', 'demo@example.com', sample_password, 'Demo User'))
        conn.commit()
        print("Sample user created: username='demo_user', password='password123'")
    except sqlite3.IntegrityError:
        print("Sample user already exists")
    
    conn.close()

def verify_database():
    """Verify database structure"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\n=== Database Verification ===")
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
        
        # Show schema for each table
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print(f"    Columns: {', '.join([col[1] for col in columns])}")
    
    conn.close()

if __name__ == "__main__":
    create_database()
    create_sample_user()
    verify_database()