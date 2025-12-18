from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.inference import TypoInference

MODEL_PATH = BASE_DIR / "models" / "rut5-corrector"

app = Flask(__name__)
CORS(app)

print("Загружаем модель RuT5...")
corrector = TypoInference(str(MODEL_PATH))
print("Модель загружена!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/correct", methods=["POST"])
def correct_text():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Текст не предоставлен"}), 400

        result = corrector.analyze(text)

        corrected = result["text"]
        issues = result.get("issues", [])

        # corrections — список вариантов (пока один)
        corrections = [{
            "text": corrected,
            "confidence": issue.get("confidence", 0.85)
        } for issue in issues] or [{
            "text": corrected,
            "confidence": 0.95
        }]

        return jsonify({
            "original": text,
            "best_correction": corrected,
            "corrections": corrections,
            "errors": [
                {
                    "original": issue["original"],
                    "suggestions": issue["suggestions"],
                    "confidence": issue.get("confidence", 0.85),
                }
                for issue in issues
            ],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Сервер запущен!")
    print("http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True,
    )
