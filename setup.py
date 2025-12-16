"""
Скрипт для первоначальной настройки проекта.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Устанавливает зависимости."""
    print("Устанавливаем зависимости...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    print("✓ Зависимости установлены")


def download_nltk_data():
    """Скачивает данные NLTK."""
    print("Скачиваем данные NLTK...")
    import nltk

    try:
        nltk.download("punkt", quiet=True)
        print("✓ Данные NLTK скачаны")
    except Exception as e:
        print(f"⚠ Ошибка при скачивании NLTK данных: {e}")


def create_directories():
    """Создаёт необходимые директории."""
    dirs = ["data/raw", "data/processed", "models", "web/static", "web/templates"]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✓ Директории созданы")


def main():
    """Основная функция настройки."""
    print("=" * 50)
    print("Настройка проекта исправления опечаток")
    print("=" * 50)

    create_directories()
    install_requirements()
    download_nltk_data()

    print("\n" + "=" * 50)
    print("Настройка завершена!")
    print("\nСледующие шаги:")
    print("1. Запустите: python src/data_processing.py")
    print("2. Запустите: python src/train.py")
    print("3. Запустите: python web/app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
