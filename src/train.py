"""
train.py - Обучение модели коррекции текста
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ДАТАСЕТ
# ============================================================================


class TextCorrectionDataset(Dataset):
    """Датасет для исправления текстов"""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        """
        Args:
            csv_path: путь к CSV файлу
            tokenizer: tokenizer из transformers
            max_length: максимальная длина
        """
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Загружаем данные
        self.data = pd.read_csv(csv_path)

        # Убираем пустые
        self.data = self.data.dropna(subset=["input_text", "output_text"])
        self.data = self.data[
            (self.data["input_text"].str.len() > 0)
            & (self.data["output_text"].str.len() > 0)
        ]

        logger.info(f"Загружено {len(self.data)} примеров из {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        input_text = "fix: " + str(row['input_text'])

        # Кодируем input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Кодируем output (target)
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": output_encoding["input_ids"].squeeze(),
        }


# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================


class TextCorrectionTrainer:
    """Trainer для модели коррекции текстов"""

    def __init__(self, model_name: str = "t5-small", device: str = "cuda"):
        """
        Args:
            model_name: модель из HuggingFace
            device: cuda или cpu
        """
        self.device = device
        self.model_name = model_name

        # Загружаем модель и tokenizer
        logger.info(f"Загружаю {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)

        logger.info("✅ Модель загружена")

    def train(
        self,
        train_csv: str,
        output_dir: str = "./models/correction_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        validation_split: float = 0.1,
    ):
        """
        Обучает модель

        Args:
            train_csv: путь к CSV с данными
            output_dir: директория для сохранения модели
            num_epochs: количество эпох
            batch_size: размер батча
            learning_rate: learning rate
            validation_split: доля validation данных
        """

        logger.info("=" * 80)
        logger.info("🚀 ОБУЧЕНИЕ МОДЕЛИ КОРРЕКЦИИ ТЕКСТА")
        logger.info("=" * 80)

        # Создаем датасет
        logger.info("\n📋 Загрузка данных...")
        dataset = TextCorrectionDataset(train_csv, self.tokenizer)

        # Split на train/val
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        logger.info(f"   Train: {len(train_dataset)}")
        logger.info(f"   Val: {len(val_dataset)}")

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        logger.info("\n📖 ОБУЧЕНИЕ:")
        logger.info("-" * 80)

        self.model.train()
        total_steps = num_epochs * len(train_loader)
        current_step = 0

        for epoch in range(num_epochs):
            logger.info(f"\nЭпоха {epoch + 1}/{num_epochs}")

            epoch_loss = 0

            # Training
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                current_step += 1
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"   Train Loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"   Val Loss: {avg_val_loss:.4f}")

            self.model.train()

        # Сохраняем модель
        logger.info("\n" + "=" * 80)
        logger.info("💾 СОХРАНЕНИЕ МОДЕЛИ")
        logger.info("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем модель и tokenizer
        self.model.save_pretrained(output_path / "model")
        self.tokenizer.save_pretrained(output_path / "tokenizer")

        # Сохраняем параметры
        config = {
            "model_name": self.model_name,
            "max_length": 128,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }

        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"\n✅ Модель сохранена в {output_path}")
        logger.info(f"   📁 Model: {output_path / 'model'}")
        logger.info(f"   📁 Tokenizer: {output_path / 'tokenizer'}")
        logger.info(f"   📄 Config: {output_path / 'config.json'}")

        return self.model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение модели коррекции текста")
    parser.add_argument("--model", type=str, default="cointegrated/rut5-base", help="Базовая модель")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/all_train_enhanced.csv",
        help="CSV с данными",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/correction_model",
        help="Директория для сохранения",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")

    args = parser.parse_args()

    # Проверяем наличие данных
    if not Path(args.data).exists():
        # Используем картасловский файл если нет обработанных данных
        if Path("orfo_and_typos.L1_5.csv").exists():
            # Конвертируем картасловский файл
            logger.info("Конвертирую картасловский файл...")
            df = pd.read_csv("orfo_and_typos.L1_5.csv", sep=";", on_bad_lines="skip")

            # Переименовываем столбцы
            df.columns = ["input_text", "output_text", "weight"]

            # Сохраняем
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            df.to_csv("data/processed/all_train_enhanced.csv", index=False)
            args.data = "data/processed/all_train_enhanced.csv"
            logger.info(f"✅ Файл подготовлен: {args.data}")
        else:
            logger.error(f"❌ Не найден файл: {args.data}")
            exit(1)

    # Обучаем модель
    trainer = TextCorrectionTrainer(model_name=args.model, device=args.device)
    trainer.train(
        train_csv=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
    )

    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"\n🚀 Дальше:")
    print(f"   python inference.py --model {args.output}")
    print(f"   python app.py --model {args.output}")
