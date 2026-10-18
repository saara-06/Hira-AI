
import os
import json
import sqlite3
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

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hira_history.db")
)

db_lock = Lock()
MAX_MESSAGE_LENGTH = 12000
MAX_HISTORY_MESSAGES = 20


def get_db():
    connection = sqlite3.connect(
        DATABASE_PATH, timeout=20, check_same_thread=False
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
                """INSERT INTO messages
                (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)""",
                (
                    session_id, role, content,
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
                """SELECT role, content, created_at FROM messages
                WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
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


def call_openrouter(history, message):
    if not API_KEY:
        raise RuntimeError(
            "OpenRouter API key is missing. Add OPENROUTER_API_KEY in Render."
        )

    system_prompt = """
You are Hira AI, a friendly, helpful, smart AI study and career assistant.

PERSONALITY:
- Be caring, encouraging, funny when appropriate, and slightly playful.
- If the user speaks Hinglish or Roman Urdu, reply similarly.
- If the user writes in English, reply in English.
- Keep answers clear and useful without unnecessary length.

WHAT YOU HELP WITH:
- Python, Java, C, C++, HTML, CSS, JavaScript, SQL, DBMS and DSA.
- AI, machine learning, beginner projects and debugging.
- College assignments, exams, study plans, hackathons and internships.

CODING RULES:
- Provide complete runnable programs when requested.
- Explain errors in simple steps.
- Never claim code was tested or deployed unless it actually was.
- Never reveal API keys or private configuration.
"""

    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1500
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://saara-06.github.io",
            "X-OpenRouter-Title": "Hira AI"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("The AI returned no response. Please retry.")

        answer = choices[0].get("message", {}).get("content", "")
        if isinstance(answer, list):
            answer = "\n".join(
                part.get("text", "")
                for part in answer
                if isinstance(part, dict) and part.get("text")
            )

        answer = answer.strip() if isinstance(answer, str) else ""
        if not answer:
            raise RuntimeError("Hira received an empty response. Please retry.")

        return answer

    except urllib.error.HTTPError as error:
        try:
            details = json.loads(error.read().decode("utf-8"))
            message = details.get("error", {}).get("message")
            if not message:
                message = str(details.get("error", "API request failed"))
        except Exception:
            message = "OpenRouter returned HTTP " + str(error.code)

        app.logger.error("OpenRouter HTTP %s: %s", error.code, message)
        raise RuntimeError(
            "OpenRouter request failed (HTTP " + str(error.code) +
            "). Check your API key, free-model availability and rate limits."
        )

    except urllib.error.URLError:
        raise RuntimeError("Could not connect to the AI service. Please retry.")

    except (TimeoutError, json.JSONDecodeError):
        raise RuntimeError("The AI service timed out or returned invalid data.")


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
        "provider": "OpenRouter",
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
        return jsonify({"error": "Your message is too long."}), 400

    if not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "A session ID is required."}), 400

    session_id = session_id.strip()
    if len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    try:
        history = read_history(session_id)
        reply = call_openrouter(history, message)

        save_message(session_id, "user", message)
        save_message(session_id, "assistant", reply)

        return jsonify({"reply": reply, "session_id": session_id})

    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    except Exception:
        app.logger.exception("Unexpected error in /chat")
        return jsonify({"error": "Something went wrong. Please retry."}), 500


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
    return jsonify({"success": True, "session_id": session_id})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large."}), 413


init_db()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
