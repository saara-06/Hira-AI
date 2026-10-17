import os
import uuid
from threading import Lock

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)

FRONTEND_ORIGINS = [
    "https://saara-06.github.io",
    "https://edufamily-ai.vercel.app",
]
CORS(app, resources={r"/chat": {"origins": FRONTEND_ORIGINS}})

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Temporary in-memory history. Render restarts clear this history.
conversation_history = {}
history_lock = Lock()
MAX_HISTORY_MESSAGES = 12
MAX_SESSIONS = 500

SYSTEM_INSTRUCTIONS = """
You are Hira, a friendly, caring, funny, slightly savage, motivating,
smart AI study and career assistant.

Known project context:
- The user's name is Shaik Saara Tabassum.
- She is a second-year B.Tech Computer Science student.
- Her main AI assistant project is Hira AI, being built as a portfolio project.
- Her education website project is EduFamily AI.
- The frontend is hosted on GitHub Pages and the backend is hosted on Render.
- She is learning programming, AI, and wants to become internship-ready.

How to respond:
- Be warm and supportive, like a helpful dost.
- Use simple Hinglish/Roman Urdu when it feels natural.
- Explain technical topics step by step for a beginner.
- For programming requests, provide complete working code when practical.
- Use recent conversation messages to understand follow-up questions.
- Do not ask for facts already stated in this context.
- Do not claim to remember messages that are not in the provided context.
- Keep replies practical, clear, and avoid unnecessary repetition.
"""

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
        "api_key_configured": bool(client)
    })

@app.post("/chat")
def chat():
    if client is None:
        return jsonify({
            "error": "Gemini API key is not configured in Render environment variables."
        }), 503

    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    session_id = data.get("session_id", "")

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400
    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Please keep your message under 4000 characters."}), 400
    if not isinstance(session_id, str) or len(session_id) > 100:
        return jsonify({"error": "Invalid session ID."}), 400

    try:
        with history_lock:
            history = list(conversation_history.get(session_id, [])) if session_id else []

        contents = []
        for item in history:
            contents.append({
                "role": item["role"],
                "parts": [{"text": item["text"]}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })

        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=contents,
            config={"system_instruction": SYSTEM_INSTRUCTIONS},
        )

        reply = (response.text or "").strip()
        if not reply:
            reply = "Yaar, mujhe is baar reply nahi mila. Ek baar phir try karo."

        if session_id:
            updated_history = history + [
                {"role": "user", "text": message},
                {"role": "model", "text": reply},
            ]
            updated_history = updated_history[-MAX_HISTORY_MESSAGES:]
            with history_lock:
                if len(conversation_history) >= MAX_SESSIONS and session_id not in conversation_history:
                    oldest_key = next(iter(conversation_history), None)
                    if oldest_key:
                        conversation_history.pop(oldest_key, None)
                conversation_history[session_id] = updated_history

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Hira could not reply right now. Please try again in a moment."
        }), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
