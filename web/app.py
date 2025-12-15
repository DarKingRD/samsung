"""
Flask приложение для веб-интерфейса исправления опечаток.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import load_corrector

app = Flask(__name__)
CORS(app)

# Загружаем корректор при старте
print("Загружаем модель исправления опечаток...")
corrector = load_corrector()
print("Модель загружена!")


@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html')


@app.route('/api/correct', methods=['POST'])
def correct_text():
    """API endpoint для исправления текста."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Текст не предоставлен'}), 400
        
        # Получаем исправления
        corrections = corrector.correct_text(text, top_k=3)
        
        # Находим ошибки
        errors = corrector.find_errors(text)
        
        return jsonify({
            'original': text,
            'corrections': [
                {'text': corr[0], 'confidence': corr[1]} 
                for corr in corrections
            ],
            'errors': errors,
            'best_correction': corrections[0][0] if corrections else text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Проверка работоспособности API."""
    return jsonify({'status': 'ok', 'model_loaded': corrector.use_model or len(corrector.typo_dict) > 0})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Сервер запущен!")
    print("Откройте браузер по адресу: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
