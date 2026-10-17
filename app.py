
import os
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Hira AI backend is running!"
    })


@app.route("/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "Gemini API key is not configured."
        }), 503

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "You are Hira, a friendly and caring study and career "
                "assistant. Explain things in simple Hinglish when helpful. "
                "Be supportive, practical, and concise.\n\n"
                f"User: {message}"
            )
        )
        return jsonify({"reply": response.text or "Please try again."})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Hira could not reply right now. Please try again."
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
