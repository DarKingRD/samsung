"""
app.py - Веб-приложение для исправления ошибок (Flask + jQuery)
Аналог Grammarly
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import logging
import json
import torch
from inference import ErrorCorrectionInference, CorrectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]

WEB_DIR = BASE_DIR / "web"

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================================

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
    static_url_path="/static",
)
app.config['JSON_SORT_KEYS'] = False

# Загружаем модель
MODEL_PATH = "./models/correction_model_v2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger.info(f"🚀 Загружаю модель... (device: {DEVICE})")
try:
    if Path(MODEL_PATH).exists():
        model = ErrorCorrectionInference(model_path=MODEL_PATH, device=DEVICE)
    else:
        logger.warning(f"Модель не найдена в {MODEL_PATH}, используем базовую t5-small")
        model = ErrorCorrectionInference(device=DEVICE)
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

logger.info("✅ Приложение готово!")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/correct', methods=['POST'])
def correct_text():
    """API endpoint для исправления текста"""
    
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Текст пуст',
                'status': 'error'
            }), 400
        
        if not model:
            return jsonify({
                'error': 'Модель не загружена',
                'status': 'error'
            }), 500
        
        # Исправляем текст
        logger.info(f"Исправляю текст: {text[:50]}...")
        result = model.correct(text)
        
        # Конвертируем результат в JSON
        response = {
            'status': 'success',
            'original_text': result.original_text,
            'corrected_text': result.corrected_text,
            'error_count': result.error_count,
            'corrections': [
                {
                    'position': c.position,
                    'original': c.original,
                    'corrected': c.corrected,
                    'confidence': round(c.confidence, 2),
                    'error_type': c.error_type,
                }
                for c in result.corrections
            ]
        }
        
        logger.info(f"✅ Найдено {result.error_count} ошибок")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/highlight', methods=['POST'])
def highlight_errors():
    """API endpoint для выделения ошибок в HTML"""
    
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text or not model:
            return jsonify({
                'error': 'Текст пуст или модель не загружена',
                'status': 'error'
            }), 400
        
        # Исправляем и выделяем
        result = model.correct(text)
        highlighted = model.highlight_errors(text)
        
        return jsonify({
            'status': 'success',
            'highlighted_html': highlighted,
            'error_count': result.error_count,
        })
    
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint для статистики"""
    
    stats = {
        'status': 'success',
        'model_loaded': model is not None,
        'device': DEVICE,
        'app_version': '1.0.0',
    }
    
    return jsonify(stats)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🚀 ЗАПУСК ВЕЩЕСТВЛЕНИЯ ПРИЛОЖЕНИЯ")
    logger.info("="*80)
    logger.info("\n📱 Веб-интерфейс доступен на: http://localhost:5000")
    logger.info("📚 API документация:")
    logger.info("   POST /api/correct - исправление текста")
    logger.info("   GET /api/stats - статистика приложения")
    logger.info("\nНажмите Ctrl+C для выхода\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Отключаем reload-ер для экономии памяти
    )
