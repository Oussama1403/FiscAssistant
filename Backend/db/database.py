import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("fiscassistant.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            language TEXT,
            timestamp DATETIME
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            declaration_type TEXT,
            deadline TEXT,
            created_at DATETIME
        )
    """)
    conn.commit()
    conn.close()

def save_conversation(user_id: str, role: str, content: str, language: str):
    conn = sqlite3.connect("fiscassistant.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (user_id, role, content, language, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, role, content, language, datetime.now())
    )
    conn.commit()
    conn.close()

def get_conversation_history(user_id: str, limit: int = 5):
    conn = sqlite3.connect("fiscassistant.db")
    c = conn.cursor()
    c.execute("SELECT role, content, language FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    history = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1], "language": row[2]} for row in history]

def save_reminder(user_id: str, declaration_type: str, deadline: str):
    conn = sqlite3.connect("fiscassistant.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (user_id, declaration_type, deadline, created_at) VALUES (?, ?, ?, ?)",
        (user_id, declaration_type, deadline, datetime.now())
    )
    conn.commit()
    conn.close()

def get_reminders(user_id: str):
    conn = sqlite3.connect("fiscassistant.db")
    c = conn.cursor()
    c.execute("SELECT declaration_type, deadline FROM reminders WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    reminders = c.fetchall()
    conn.close()
    return [{"type": row[0], "deadline": row[1]} for row in reminders]