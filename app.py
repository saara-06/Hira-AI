
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# ---------------- APP CONFIG ----------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

FRONTEND_ORIGINS = [
    "https://saara-06.github.io",
    "https://edufamily-ai.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

CORS(
    app,
    resources={r"/*": {"origins": FRONTEND_ORIGINS}},
    methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    supports_credentials=False,
)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

DB_PATH = os.environ.get(
    "HIRA_DB_PATH",
    str(Path(__file__).resolve().parent / "hira_history.db"),
)

# 90-second HTTP timeout for Gemini requests
client = (
    genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=90000),
    )
    if API_KEY
    else None
)

db_lock = Lock()

SYSTEM_INSTRUCTIONS = """
You are Hira AI, a friendly, caring, funny, motivating
and smart study and career assistant.

Known context:
- The user's name is Shaik Saara Tabassum.
- She is a second-year B.Tech Computer Science student.
- Hira AI is her AI assistant portfolio project.
- EduFamily AI is her education website project.
- She is learning programming and preparing for internships.

Instructions:
- Use the supplied conversation history to answer questions
  about earlier messages.
- Speak in simple English or Hinglish/Roman Urdu depending
  on the user's language.
- Explain things step by step for beginners.
- Help with Python, Java, C, C++, HTML, CSS, JavaScript,
  SQL, AI and debugging.
- Help with study notes, quizzes, exam answers, career
  roadmaps and interviews.
- Provide complete working code when practical.
- Be honest if information is missing.
- Never pretend to remember a different chat.
- Never reveal API keys, credentials or secrets.
"""

# ---------------- DATABASE ----------------

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
                SELECT role, text
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

    return [
        {"role": row["role"], "text": row["text"]}
        for row in reversed(rows)
    ]


def save_message(session_id, role, text):
    now = int(time.time())

    with db_lock:
        with database() as con:
            if role == "user":
                title = " ".join(text.split())[:48] or "New chat"

                con.execute(
                    """
                    INSERT INTO sessions
                    (session_id, title, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id)
                    DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (session_id, title, now),
                )
            else:
                con.execute(
                    """
                    INSERT INTO sessions
                    (session_id, title, updated_at)
                    VALUES (?, 'New chat', ?)
                    ON CONFLICT(session_id)
                    DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (session_id, now),
                )

            con.execute(
                """
                INSERT INTO messages
                (session_id, role, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, text, now),
            )

            con.commit()


initialize_database()

# ---------------- GEMINI RETRY ----------------

def generate_with_retry(contents):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config={
                    "system_instruction": SYSTEM_INSTRUCTIONS,
                },
            )

        except Exception as error:
            error_text = str(error).upper()

            temporary_error = any(
                code in error_text
                for code in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "502",
                    "504",
                    "INTERNAL",
                    "HIGH DEMAND",
                )
            )

            if not temporary_error or attempt == 2:
                raise

            delay = 2 ** (attempt + 1)

            app.logger.warning(
                "Temporary Gemini error; retry %s of 2 "
                "after %s seconds.",
                attempt + 1,
                delay,
            )

            time.sleep(delay)

    raise RuntimeError("Gemini request failed after retries.")

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return jsonify({
        "message": "Hira AI backend is running!",
        "chat_endpoint": "/chat",
        "health": "/health",
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(client),
        "model": MODEL,
        "history_storage": "SQLite",
    })


@app.get("/history/<session_id>")
def read_history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    return jsonify({
        "session_id": session_id,
        "messages": get_history(session_id, limit=100),
    })


@app.delete("/history/<session_id>")
def delete_history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    with db_lock:
        with database() as con:
            con.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            con.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            con.commit()

    return jsonify({
        "ok": True,
        "message": "Chat cleared.",
    })


@app.post("/chat")
def chat():
    if client is None:
        return jsonify({
            "error": (
                "GEMINI_API_KEY is missing in Render "
                "environment variables."
            )
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
        return jsonify({
            "error": "Message is too long. Maximum 8000 characters."
        }), 400

    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > 100
    ):
        return jsonify({
            "error": "A valid session_id is required."
        }), 400

    try:
        history = get_history(session_id, limit=24)

        contents = [
            {
                "role": item["role"],
                "parts": [{"text": item["text"]}],
            }
            for item in history
            if item["role"] in ("user", "model")
        ]

        contents.append({
            "role": "user",
            "parts": [{"text": message}],
        })

        response = generate_with_retry(contents)

        reply = (response.text or "").strip()

        if not reply:
            reply = (
                "Yaar, I couldn't generate a reply. "
                "Please try again."
            )

        save_message(session_id, "user", message)
        save_message(session_id, "model", reply)

        return jsonify({
            "reply": reply,
            "session_id": session_id,
        })

    except Exception as error:
        app.logger.exception("Hira AI request failed")

        return jsonify({
            "error": (
                "Gemini is temporarily unavailable or "
                "could not complete the request. Please retry."
            ),
            "type": type(error).__name__,
        }), 502


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": "Request is too large."
    }), 413


@app.errorhandler(500)
def internal_error(_error):
    app.logger.exception("Internal server error")

    return jsonify({
        "error": "Internal server error. Please try again."
    }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )
