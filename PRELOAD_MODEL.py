"""
Скрипт для предварительной загрузки модели.
Запустите этот скрипт один раз, чтобы модель загрузилась в кэш.
"""

import os
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("Предварительная загрузка модели ruT5-base")
print("=" * 60)
print("\nЭто займёт некоторое время при первом запуске...")
print("Модель будет загружена в кэш Hugging Face.\n")

# Устанавливаем переменные окружения
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("Загружаем токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ai-forever/ruT5-base", resume_download=True
    )
    print("✓ Токенизатор загружен")

    print("\nЗагружаем модель (это может занять несколько минут)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai-forever/ruT5-base", resume_download=True
    )
    print("✓ Модель загружена")

    print("\n" + "=" * 60)
    print("✓ Модель успешно загружена в кэш!")
    print("Теперь при запуске web/app.py модель загрузится быстрее.")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Ошибка при загрузке модели: {e}")
    print("\nПопробуйте:")
    print("1. Проверить интернет-соединение")
    print("2. Запустить скрипт снова (загрузка возобновится)")
    sys.exit(1)
