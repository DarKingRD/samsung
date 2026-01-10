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

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================================

app = Flask(__name__)
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
# HTML TEMPLATES
# ============================================================================

# Создаем папку templates если её нет
template_dir = Path('templates')
template_dir.mkdir(exist_ok=True)

# HTML шаблон
html_template = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔧 Корректор текста - Исправление ошибок</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 16px;
        }
        
        .content {
            padding: 40px 30px;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            font-family: 'Segoe UI', sans-serif;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            flex: 1;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .results {
            display: none;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .results.show {
            display: block;
        }
        
        .result-text {
            margin-bottom: 20px;
        }
        
        .result-text h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        .result-box {
            padding: 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            line-height: 1.6;
        }
        
        .error-count {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .corrections-list {
            list-style: none;
        }
        
        .correction-item {
            padding: 12px;
            margin-bottom: 8px;
            background: white;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .correction-type {
            display: inline-block;
            padding: 2px 8px;
            background: #ffc107;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
        }
        
        .correction-type.spelling { background: #ff6b6b; }
        .correction-type.punctuation { background: #4ecdc4; }
        .correction-type.grammar { background: #95e1d3; }
        .correction-type.semantics { background: #f38181; }
        
        .from-to {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 6px;
        }
        
        .from, .to {
            flex: 1;
        }
        
        .from {
            color: #d32f2f;
        }
        
        .to {
            color: #388e3c;
        }
        
        .confidence {
            font-size: 12px;
            color: #999;
            margin-left: auto;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            color: #d32f2f;
            font-weight: 500;
        }
        
        .success {
            color: #388e3c;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 Корректор текста</h1>
            <p>Исправление опечаток, грамматических и смысловых ошибок</p>
        </div>
        
        <div class="content">
            <!-- Ввод текста -->
            <div class="section">
                <div class="section-title">Введите текст с ошибками:</div>
                <textarea id="input-text" placeholder="Напишите текст с ошибками, и я его исправлю..."></textarea>
                
                <div class="button-group">
                    <button class="btn-primary" onclick="correctText()">✨ Исправить</button>
                    <button class="btn-secondary" onclick="clearAll()">Очистить</button>
                </div>
            </div>
            
            <!-- Загрузка -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Анализирую текст...</p>
            </div>
            
            <!-- Результаты -->
            <div class="results" id="results">
                <div class="section">
                    <div class="section-title">Исправленный текст:</div>
                    <div class="result-box result-text">
                        <p id="corrected-text"></p>
                    </div>
                </div>
                
                <div class="section">
                    <div class="error-count">
                        Найдено <span id="error-count">0</span> ошибок
                    </div>
                    
                    <ul class="corrections-list" id="corrections-list"></ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function correctText() {
            const text = document.getElementById('input-text').value.trim();
            
            if (!text) {
                alert('Пожалуйста, введите текст!');
                return;
            }
            
            // Показываем загрузку
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            
            // Отправляем запрос
            fetch('/api/correct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                
                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    alert('Ошибка: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                alert('Ошибка запроса: ' + error);
                console.error('Error:', error);
            });
        }
        
        function displayResults(data) {
            // Показываем исправленный текст
            document.getElementById('corrected-text').textContent = data.corrected_text;
            document.getElementById('error-count').textContent = data.error_count;
            
            // Показываем список ошибок
            const correctionsList = document.getElementById('corrections-list');
            correctionsList.innerHTML = '';
            
            data.corrections.forEach((corr, idx) => {
                const li = document.createElement('li');
                li.className = 'correction-item';
                
                li.innerHTML = `
                    <div>
                        <span class="correction-type ${corr.error_type}">${corr.error_type}</span>
                        <span class="confidence">${(corr.confidence * 100).toFixed(0)}% уверенность</span>
                    </div>
                    <div class="from-to">
                        <div class="from">❌ "${corr.original}"</div>
                        <div class="to">✅ "${corr.corrected}"</div>
                    </div>
                `;
                
                correctionsList.appendChild(li);
            });
            
            document.getElementById('results').classList.add('show');
        }
        
        function clearAll() {
            document.getElementById('input-text').value = '';
            document.getElementById('results').classList.remove('show');
        }
        
        // Позволяем исправлять по Ctrl+Enter
        document.getElementById('input-text').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                correctText();
            }
        });
    </script>
</body>
</html>
'''

# Создаем файл шаблона
with open(template_dir / 'index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

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
