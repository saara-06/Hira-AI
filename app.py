
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

FRONTEND_ORIGINS = [
    "https://saara-06.github.io",
    "https://edufamily-ai.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

CORS(app, resources={
    r"/chat": {"origins": FRONTEND_ORIGINS},
    r"/history/*": {"origins": FRONTEND_ORIGINS},
    r"/health": {"origins": FRONTEND_ORIGINS},
})

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DB_PATH = os.environ.get(
    "HIRA_DB_PATH",
    str(Path(__file__).resolve().parent / "hira_history.db")
)

client = genai.Client(api_key=API_KEY) if API_KEY else None
db_lock = Lock()

SYSTEM_INSTRUCTIONS = """
You are Hira AI, a friendly, caring, funny, motivating study and career assistant.

Known context:
- The user's name is Shaik Saara Tabassum.
- She is a second-year B.Tech Computer Science student.
- Hira AI is her AI assistant portfolio project.
- EduFamily AI is her education website project.
- She is learning programming and preparing for internships.

Instructions:
- Use the supplied conversation history to answer questions about earlier messages.
- Never say a message was not provided if it appears in the conversation history.
- Speak in simple English or Hinglish/Roman Urdu depending on the user.
- Explain things step by step for beginners.
- Help with Python, Java, C, C++, HTML, CSS, JavaScript, SQL, AI and debugging.
- Help with study notes, quizzes, exam answers, career roadmaps and interviews.
- When practical, provide complete working code instead of disconnected snippets.
- Be honest if information is missing. Never pretend to remember a different chat.
- Never reveal API keys, credentials or secrets.
"""

def database():
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    return con

def initialize_database():
    with db_lock:
        with database() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            con.execute("""
                CREATE INDEX IF NOT EXISTS messages_session_idx
                ON messages(session_id, id)
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    updated_at INTEGER NOT NULL
                )
            """)
            con.commit()

def get_history(session_id, limit=24):
    with db_lock:
        with database() as con:
            rows = con.execute(
                """
                SELECT role, text FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit)
            ).fetchall()

    return [
        {"role": row["role"], "text": row["text"]}
        for row in reversed(rows)
    ]

def save_message(session_id, role, text):
    now = int(time.time())

    with db_lock:
        with database() as con:
            con.execute(
                """
                INSERT INTO messages(session_id, role, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, text, now)
            )

            title = " ".join(text.split())[:48] or "New chat"

            con.execute(
                """
                INSERT INTO sessions(session_id, title, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, title, now)
            )
            con.commit()

initialize_database()

@app.get("/")
def home():
    return jsonify({
        "message": "Hira AI backend is running!",
        "chat_endpoint": "/chat",
        "health": "/health"
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(client),
        "model": MODEL,
        "history_storage": "SQLite"
    })

@app.get("/history/<session_id>")
def read_history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    return jsonify({
        "session_id": session_id,
        "messages": get_history(session_id, limit=100)
    })

@app.delete("/history/<session_id>")
def delete_history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    with db_lock:
        with database() as con:
            con.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            con.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            con.commit()

    return jsonify({"ok": True, "message": "Chat cleared."})

@app.post("/chat")
def chat():
    if client is None:
        return jsonify({
            "error": "GEMINI_API_KEY is missing in Render environment variables."
        }), 503

    data = request.get_json(silent=True) or {}
    message = data.get("message")
    session_id = data.get("session_id")

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 8000:
        return jsonify({"error": "Message is too long. Maximum 8000 characters."}), 400

    if not isinstance(session_id, str) or not session_id or len(session_id) > 100:
        return jsonify({
            "error": "A valid session_id is required."
        }), 400

    try:
        history = get_history(session_id, limit=24)

        contents = [
            {
                "role": item["role"],
                "parts": [{"text": item["text"]}]
            }
            for item in history
        ]

        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config={"system_instruction": SYSTEM_INSTRUCTIONS}
        )

        reply = (response.text or "").strip()

        if not reply:
            reply = "Yaar, I couldn't generate a reply. Please try again."

        # Save both messages only after the AI request succeeds.
        save_message(session_id, "user", message)
        save_message(session_id, "model", reply)

        return jsonify({
            "reply": reply,
            "session_id": session_id
        })

    except Exception:
        app.logger.exception("Hira AI request failed")
        return jsonify({
            "error": "Hira could not reply. Check Render logs and try again."
        }), 502

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
