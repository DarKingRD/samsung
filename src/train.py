"""
Скрипт для обучения модели исправления опечаток.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import json
import time
import os

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Автоматически определяем путь к кэшу
def get_cache_dir():
    """Определяет путь к кэшу Hugging Face."""
    # Стандартные пути кэша
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub",
        MODELS_DIR / "cache",
    ]
    
    for path in cache_paths:
        if path.exists():
            print(f"📁 Используем кэш: {path}")
            return str(path)
    
    # Если кэш не найден, создаем в папке models
    cache_path = MODELS_DIR / "cache"
    cache_path.mkdir(exist_ok=True)
    print(f"📁 Создаем кэш в: {cache_path}")
    return str(cache_path)

# Устанавливаем путь к кэшу
CACHE_DIR = get_cache_dir()
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = str(MODELS_DIR)

print(f"\n⚙️ Настройки:")
print(f"   Кэш моделей: {CACHE_DIR}")
print(f"   Папка моделей: {MODELS_DIR}")
print(f"   Данные: {DATA_DIR}")

class TypoDataset(Dataset):
    """Датасет для обучения модели исправления опечаток."""
    
    def __init__(self, csv_path: Path, tokenizer, max_length: int = 128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        original = str(self.data.iloc[idx]['original'])
        corrected = str(self.data.iloc[idx]['corrected'])
        
        # Для T5 добавляем префикс задачи
        original = "исправь опечатку: " + original
        
        inputs = self.tokenizer(
            original,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Токенизация целевого текста (без deprecated as_target_tokenizer)
        targets = self.tokenizer(
            text_target=[corrected],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Заменяем pad_token_id на -100 для игнорирования при loss
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
        }


def load_model_simple(model_name: str = "ai-forever/ruT5-base"):
    """Простая загрузка модели с указанием кэша."""
    print(f"\n{'='*60}")
    print(f"Загрузка модели: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Загружаем токенизатор
        print("📥 Загружаем токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            legacy=False  # Используем новый вариант токенизатора
        )
        print("✅ Токенизатор загружен")
        
        # Загружаем модель
        print("📥 Загружаем модель...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR
        )
        print("✅ Модель загружена")
        
        # Проверяем, что модель загружена
        print(f"\n📊 Информация о модели:")
        print(f"   Архитектура: {model.config.model_type}")
        print(f"   Размер словаря: {len(tokenizer)}")
        print(f"   Используется устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Ошибка загрузки: {e}")
        print("\n💡 Рекомендации:")
        print("1. Убедитесь что запустили preload_model.py")
        print("2. Проверьте интернет-соединение")
        print("3. Если модель уже в кэше, можно указать путь напрямую:")
        print(f"   model_name = '{CACHE_DIR}/models--ai-forever--ruT5-base'")
        raise


def train_seq2seq_model(csv_path: Path):
    """Обучает seq2seq модель (T5) для исправления опечаток."""
    print(f"\n📊 Загружаем данные из {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл {csv_path} не найден!")
    
    # Загружаем данные
    df = pd.read_csv(csv_path)
    print(f"📈 Загружено {len(df)} примеров")
    
    # Проверяем несколько примеров
    print("\n👀 Примеры данных:")
    for i in range(min(3, len(df))):
        print(f"   {i+1}. '{df.iloc[i]['original']}' -> '{df.iloc[i]['corrected']}'")
    
    # Загружаем модель и токенизатор
    model, tokenizer = load_model_simple()
    
    # Создаём датасет
    print("\n📚 Создаём датасет...")
    dataset = TypoDataset(csv_path, tokenizer, max_length=128)
    
    # Разделяем на train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Разделение: {train_size} тренировочных, {val_size} валидационных")
    
    # Параметры обучения (оптимизированы для T5)
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4 if torch.cuda.is_available() else 2,  # Уменьшил для стабильности
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        report_to="none",
        dataloader_num_workers=0,  # 0 для Windows, чтобы избежать проблем
        remove_unused_columns=False,  # Важно для T5
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Обучение
    print("\n" + "="*60)
    print("🚀 Начинаем обучение...")
    print(f"💻 Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"📈 Эпох: {training_args.num_train_epochs}")
    print(f"📦 Batch size: {training_args.per_device_train_batch_size}")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        print("\n✅ Обучение завершено успешно!")
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        return None
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        raise
    
    # Сохраняем модель
    model_path = MODELS_DIR / "typo_corrector_model"
    print(f"\n💾 Сохраняем модель в {model_path}...")
    
    # Сохраняем модель и токенизатор
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"✅ Модель сохранена в {model_path}")
    
    # Сохраняем конфигурацию
    config = {
        "model_type": "ruT5-base",
        "trained_on": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": len(df),
        "max_length": 128,
        "task_prefix": "исправь опечатку: "
    }
    
    with open(model_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return model_path


def train_simple_model(csv_path: Path):
    """Упрощённая модель на правилах."""
    print(f"📊 Загружаем данные из {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Создаём словарь опечаток
    typo_dict = {}
    for _, row in df.iterrows():
        original = str(row['original']).strip().lower()
        corrected = str(row['corrected']).strip().lower()
        if original != corrected:
            typo_dict[original] = corrected
    
    # Сохраняем словарь
    dict_path = MODELS_DIR / "typo_dict.json"
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(typo_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Создан словарь из {len(typo_dict)} пар опечаток")
    print(f"📁 Сохранён в {dict_path}")
    
    return dict_path


def main():
    """Основная функция обучения."""
    csv_path = DATA_DIR / "typo_corpus.csv"
    
    if not csv_path.exists():
        print(f"❌ Ошибка: файл {csv_path} не найден!")
        print("Сначала запустите src/data_processing.py")
        return
    
    print("\n" + "=" * 60)
    print("🤖 ВЫБОР МЕТОДА ОБУЧЕНИЯ")
    print("=" * 60)
    print("1. Seq2Seq модель (T5)")
    print("2. Упрощённая модель на правилах")
    print("=" * 60)
    
    # Интерактивный выбор
    try:
        choice = input("\n🎯 Введите номер (1 или 2): ").strip()
        if choice not in ["1", "2"]:
            print("Используем вариант 2 по умолчанию")
            choice = "2"
    except:
        choice = "2"
    
    print(f"\n✅ Выбран метод: {'Seq2Seq модель' if choice == '1' else 'Упрощённая модель'}")
    print("-" * 60)
    
    if choice == "1":
        try:
            train_seq2seq_model(csv_path)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("\n🔄 Пробуем упрощённую модель...")
            train_simple_model(csv_path)
    else:
        train_simple_model(csv_path)
    
    print("\n" + "=" * 60)
    print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()