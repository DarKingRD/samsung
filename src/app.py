"""
app.py - Web application for Russian text error correction

This Flask application serves a web interface and REST API for the text correction model
implemented in `inference.py`. It supports:
    - Real-time text correction with multiple variants.
    - HTML highlighting of errors with tooltips.
    - API endpoints documented via Swagger (OpenAPI).
    - Fallback to base model if fine-tuned one is not found.

Frontend:
    - Located in `web/templates/` and `web/static/`.
    - Uses jQuery for AJAX calls to the backend.
    - Responsive design with error visualization.

API Endpoints:
    - POST /api/correct     → returns correction variants
    - POST /api/highlight   → returns HTML with error highlights
    - GET  /api/stats       → returns model status

Usage:
    $ python app.py
    → Open http://localhost:5000
    → API docs at http://localhost:5000/apidocs
"""

import logging
from pathlib import Path
import torch
from flask import Flask, jsonify, render_template, request
from flasgger import Swagger

# Local modules
from inference import ErrorCorrectionInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
    static_url_path="/static",
)
"""Flask app instance configured with custom template and static paths."""

app.config["JSON_SORT_KEYS"] = False
"""Disable JSON key sorting to preserve order in responses."""

app.config["SWAGGER"] = {
    "title": "Text Corrector API",
    "uiversion": 3,
    "openapi": "3.0.2",
}
"""Swagger/OpenAPI configuration."""

swagger = Swagger(
    app,
    template_file=str(WEB_DIR / "openapi.yaml"),
)
"""Swagger UI integration using external OpenAPI spec."""

# Load model
MODEL_PATH = "./models/correction_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model... (device: {DEVICE})")
try:
    if Path(MODEL_PATH).exists():
        model = ErrorCorrectionInference(model_path=MODEL_PATH, device=DEVICE)
    else:
        logger.warning(f"Model not found at {MODEL_PATH}, falling back to t5-small")
        model = ErrorCorrectionInference(device=DEVICE)
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

logger.info("✅ Application ready!")


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """
    Render the main page.

    Returns:
        Rendered HTML template 'index.html'.
    """
    return render_template("index.html")


@app.route("/api/correct", methods=["POST"])
def correct_text():
    """
    API endpoint to correct text and return multiple variants.

    Expects JSON with:
        {
            "text": str
        }

    Returns:
        {
            "status": "success" | "error",
            "original_text": str,
            "variants": [
                {
                    "rank": int,
                    "corrected_text": str,
                    "confidence": float,
                    "score": float,
                    "error_count": int,
                    "corrections": [
                        {
                            "position": int,
                            "original": str,
                            "corrected": str,
                            "confidence": float,
                            "error_type": str
                        }
                    ]
                }
            ]
        }

    Status Codes:
        200: Success
        400: Empty text
        500: Model not loaded or internal error
    """
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text is empty", "status": "error"}), 400
        if not model:
            return jsonify({"error": "Model not loaded", "status": "error"}), 500

        result = model.correct(text, n=3)

        response = {
            "status": "success",
            "original_text": result.original_text,
            "variants": [
                {
                    "rank": i + 1,
                    "corrected_text": v.corrected_text,
                    "confidence": round(v.confidence, 3),
                    "score": round(v.score, 4),
                    "error_count": v.error_count,
                    "corrections": [
                        {
                            "position": c.position,
                            "original": c.original,
                            "corrected": c.corrected,
                            "confidence": round(v.confidence * c.confidence, 3),
                            "error_type": c.error_type,
                        }
                        for c in v.corrections
                    ],
                }
                for i, v in enumerate(result.variants)
            ],
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /api/correct: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/highlight", methods=["POST"])
def highlight_errors():
    """
    API endpoint to get HTML with highlighted errors.

    Expects JSON with:
        {
            "text": str,
            "variant_index": int (optional, default=0)
        }

    Returns:
        {
            "status": "success",
            "highlighted_html": str,
            "error_count": int
        }

    Status Codes:
        200: Success
        400: Empty text or invalid variant
        500: Model error or internal exception
    """
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text or not model:
            return (
                jsonify(
                    {"error": "Text is empty or model not loaded", "status": "error"}
                ),
                400,
            )

        variant_index = int(data.get("variant_index", 0))
        highlighted = model.highlight_errors(text, variant_index=variant_index)

        result = model.correct(text, n=3)
        best = result.variants[variant_index]

        return jsonify(
            {
                "status": "success",
                "highlighted_html": highlighted,
                "error_count": best.error_count,
            }
        )

    except Exception as e:
        logger.error(f"Error in /api/highlight: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    API endpoint to get application and model status.

    Returns:
        {
            "status": "success",
            "model_loaded": bool,
            "app_version": str
        }

    Useful for health checks and frontend status indicators.
    """
    stats = {
        "status": "success",
        "model_loaded": model is not None,
        "app_version": "1.0.0",
    }
    return jsonify(stats)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Entry point for running the Flask development server.

    Launches the app with:
        - Host: 0.0.0.0 (accessible on network)
        - Port: 5000
        - Debug mode enabled
        - Reloader disabled (to save GPU memory)

    Prints startup message with access URLs.
    """
    logger.info("=" * 80)
    logger.info("STARTING APPLICATION")
    logger.info("=" * 80)
    logger.info("\nWeb interface available at: http://localhost:5000")
    logger.info("API documentation available at: http://localhost:5000/apidocs/")
    logger.info("\nPress Ctrl+C to exit\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,  # Disable reloader to avoid reloading model and save memory
    )
