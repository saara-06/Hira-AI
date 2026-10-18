
import os
import json
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from threading import Lock

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

CORS(
    app,
    resources={r"/*": {"origins": [
        "https://saara-06.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5000"
    ]}},
    supports_credentials=False
)

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hira_history.db")
)

db_lock = Lock()
MAX_MESSAGE_LENGTH = 12000
MAX_HISTORY_MESSAGES = 20


def get_db():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=20,
        check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db_lock:
        connection = get_db()
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id)
            """)
            connection.commit()
        finally:
            connection.close()


def save_message(session_id, role, content):
    with db_lock:
        connection = get_db()
        try:
            connection.execute(
                """
                INSERT INTO messages
                (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    datetime.now(timezone.utc).isoformat()
                )
            )
            connection.commit()
        finally:
            connection.close()


def read_history(session_id, limit=MAX_HISTORY_MESSAGES):
    with db_lock:
        connection = get_db()
        try:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit)
            ).fetchall()

            return [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"]
                }
                for row in reversed(rows)
            ]
        finally:
            connection.close()


def delete_history(session_id):
    with db_lock:
        connection = get_db()
        try:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            connection.commit()
        finally:
            connection.close()


def build_prompt(history, message):
    contents = []

    for item in history:
        if item["role"] not in ("user", "assistant"):
            continue

        role = "user" if item["role"] == "user" else "model"

        contents.append({
            "role": role,
            "parts": [{"text": item["content"]}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    return contents


def call_gemini(history, message):
    if not API_KEY:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY in Render environment variables."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + MODEL
        + ":generateContent?key="
        + API_KEY
    )

    system_instruction = """
You are Hira AI, a friendly, helpful, smart AI study and career assistant.

PERSONALITY:
- Be caring, encouraging, funny when appropriate, and slightly playful.
- Speak naturally. If the user speaks Hinglish or Roman Urdu, reply similarly.
- If the user writes in English, reply in English.
- Keep answers clear and useful. Avoid unnecessarily long explanations.
- Address the user naturally; do not repeatedly use their name.

WHAT YOU HELP WITH:
- Python, Java, C, C++, HTML, CSS, JavaScript, SQL, DBMS and DSA.
- AI, machine learning, beginner projects and debugging.
- College assignments, exam preparation, quizzes and study plans.
- Hackathons, internships, GitHub, resumes and career preparation.
- General questions and everyday productivity.

CODING RULES:
- When asked for a complete program, provide complete runnable code.
- When asked to replace a file, provide the full replacement file.
- Explain errors in simple steps.
- Never claim code has been executed or deployed unless that actually happened.
- Do not invent test results, links, credentials or project status.

SAFETY AND PRIVACY:
- Never reveal API keys, secrets, environment variables or private server configuration.
- Do not claim to access files, accounts or devices unless tools actually provide access.
- If a question needs current information, clearly state uncertainty when you cannot verify it.
"""

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": build_prompt(history, message),
        "generationConfig": {
            "temperature": 0.75,
            "topP": 0.95,
            "maxOutputTokens": 4096
        }
    }

    body = json.dumps(payload).encode("utf-8")

    last_error = "Gemini request failed."

    for attempt in range(3):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))

            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError(
                    "The AI returned no response. Please try again."
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "\n".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Hira received an empty response. Please try again."
                )

            return answer

        except urllib.error.HTTPError as error:
            try:
                details = error.read().decode("utf-8")
                data = json.loads(details)
                last_error = data.get("error", {}).get(
                    "message", "Gemini API request failed."
                )
            except Exception:
                last_error = "Gemini API request failed with HTTP " + str(error.code)

            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last_error)

        except (TimeoutError, urllib.error.URLError) as error:
            last_error = "Connection timed out. Please try again."

        except json.JSONDecodeError:
            last_error = "The AI returned an invalid response."

        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(last_error)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Hira AI",
        "status": "running",
        "health": "/health"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(API_KEY),
        "history_storage": "SQLite"
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    session_id = data.get("session_id", "")

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": "Your message is too long. Please shorten it."
        }), 400

    if not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "A session ID is required."}), 400

    session_id = session_id.strip()

    if len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    try:
        history = read_history(session_id)
        reply = call_gemini(history, message)

        save_message(session_id, "user", message)
        save_message(session_id, "assistant", reply)

        return jsonify({
            "reply": reply,
            "session_id": session_id
        })

    except RuntimeError as error:
        return jsonify({
            "error": str(error)
        }), 503

    except Exception:
        app.logger.exception("Unexpected error in /chat")
        return jsonify({
            "error": "Something went wrong. Please try again."
        }), 500


@app.route("/history/<session_id>", methods=["GET"])
def history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    return jsonify({
        "session_id": session_id,
        "messages": read_history(session_id, limit=200)
    })


@app.route("/history/<session_id>", methods=["DELETE"])
def clear_history(session_id):
    if not session_id or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    delete_history(session_id)

    return jsonify({
        "success": True,
        "session_id": session_id
    })


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large."}), 413


init_db()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
