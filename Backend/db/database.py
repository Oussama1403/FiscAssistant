import sqlite3
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "fisc_assistant.db"

def init_db():
    """
    Initialize the SQLite database and create necessary tables.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'user' or 'assistant'
                content TEXT NOT NULL,
                language TEXT NOT NULL,  -- 'en', 'fr', 'ar'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create reminders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                declaration_type TEXT NOT NULL,
                deadline TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info("SQLite database initialized successfully at %s", DB_PATH)
    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    finally:
        conn.close()

def save_conversation(user_id: str, role: str, content: str, language: str):
    """
    Save a conversation message to the database.
    
    Args:
        user_id (str): Unique identifier for the user
        role (str): 'user' or 'assistant'
        content (str): Message text
        language (str): Language code ('en', 'fr', 'ar')
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (user_id, role, content, language, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, role, content, language, datetime.now())
        )
        conn.commit()
        logger.info("Saved conversation for user_id: %s, role: %s", user_id, role)
    except sqlite3.Error as e:
        logger.error("Failed to save conversation: %s", e)
        raise
    finally:
        conn.close()

def get_conversation_history(user_id: str) -> list:
    """
    Retrieve conversation history for a user.
    
    Args:
        user_id (str): Unique identifier for the user
    
    Returns:
        list: List of dictionaries containing conversation entries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, language, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,)
        )
        rows = cursor.fetchall()
        history = [
            {"role": row[0], "content": row[1], "language": row[2], "timestamp": row[3]}
            for row in rows
        ]
        logger.info("Retrieved %d conversation entries for user_id: %s", len(history), user_id)
        return history
    except sqlite3.Error as e:
        logger.error("Failed to retrieve conversation history: %s", e)
        raise
    finally:
        conn.close()

def save_reminder(user_id: str, declaration_type: str, deadline: str):
    """
    Save a reminder to the database.
    
    Args:
        user_id (str): Unique identifier for the user
        declaration_type (str): Type of declaration (e.g., 'TVA', 'CNSS')
        deadline (str): Deadline date/time (e.g., '2025-08-28')
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (user_id, declaration_type, deadline, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, declaration_type, deadline, datetime.now())
        )
        conn.commit()
        logger.info("Saved reminder for user_id: %s, declaration_type: %s", user_id, declaration_type)
    except sqlite3.Error as e:
        logger.error("Failed to save reminder: %s", e)
        raise
    finally:
        conn.close()

def get_reminders(user_id: str) -> list:
    """
    Retrieve reminders for a user.
    
    Args:
        user_id (str): Unique identifier for the user
    
    Returns:
        list: List of dictionaries containing reminder entries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT declaration_type, deadline, timestamp FROM reminders WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,)
        )
        rows = cursor.fetchall()
        reminders = [
            {"declaration_type": row[0], "deadline": row[1], "timestamp": row[2]}
            for row in rows
        ]
        logger.info("Retrieved %d reminders for user_id: %s", len(reminders), user_id)
        return reminders
    except sqlite3.Error as e:
        logger.error("Failed to retrieve reminders: %s", e)
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    # Test database initialization
    init_db()
    # Test saving and retrieving a conversation
    test_user_id = "test_user"
    save_conversation(test_user_id, "user", "What’s the TVA rate?", "en")
    save_conversation(test_user_id, "assistant", "The VAT rate for restaurants is 7%.", "en")
    history = get_conversation_history(test_user_id)
    print("Conversation History:", history)
    # Test saving and retrieving a reminder
    save_reminder(test_user_id, "TVA", "2025-08-28")
    reminders = get_reminders(test_user_id)
    print("Reminders:", reminders)