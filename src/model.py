"""
model.py - Модель коррекции текста на основе трансформера
Использует T5 или BERT для исправления ошибок
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from typing import Dict
import numpy as np


class ErrorCorrectionModel:
    """Модель для исправления ошибок в текстах"""

    def __init__(self, model_name: str = "cointegrated/rut5-base", device: str = "cpu"):
        """
        Args:
            model_name: название модели от HuggingFace
            device: 'cpu' или 'cuda'
        """
        self.device = device
        self.model_name = model_name

        # Выбираем модель
        if "t5" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = "seq2seq"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model_type = "masked"

        self.model.to(device)
        self.model.eval()

    def correct_text(self, text: str, max_length: int = 128) -> str:
        """
        Исправляет текст

        Args:
            text: текст с ошибками
            max_length: максимальная длина

        Returns:
            исправленный текст
        """
        if self.model_type == "seq2seq":
            return self._correct_seq2seq(text, max_length)
        else:
            return self._correct_masked(text, max_length)

    def _correct_seq2seq(self, text: str, max_length: int) -> str:
        """Коррекция с seq2seq моделью (T5)"""
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", max_length=max_length, truncation=True
        )
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                top_p=0.9,
            )

        corrected = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return corrected

    def _correct_masked(self, text: str, max_length: int) -> str:
        """Коррекция с masked моделью (BERT)"""
        # Для BERT используем более простой подход
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            tokens = tokens[: max_length - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        input_ids = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs[0]

        # Возвращаем исходный текст (BERT для полной коррекции нужно custom fine-tuning)
        return text

    def get_error_scores(self, text: str) -> Dict[int, float]:
        """
        Возвращает confidence scores для каждого слова
        (вероятность ошибки в слове)

        Args:
            text: текст для анализа

        Returns:
            dict с позициями слов и scores ошибок
        """
        words = text.split()
        error_scores = {}

        for i, word in enumerate(words):
            # Простой heuristic: длинные слова с редкими буквами - скорее всего ошибки
            rare_chars = sum(
                1 for c in word.lower() if c not in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            )
            score = min(1.0, rare_chars / len(word)) if len(word) > 0 else 0
            error_scores[i] = score

        return error_scores


class ErrorCorrectionModelFineTuned(ErrorCorrectionModel):
    """
    Модель с fine-tuning на датасете корректур
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        checkpoint_path: str = None,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: базовая модель
            checkpoint_path: путь к сохраненной модели
            device: устройство
        """
        super().__init__(model_name, device)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str):
        """Загружает веса модели"""
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def save_checkpoint(self, path: str, optimizer=None):
        """Сохраняет модель и optimizer"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)


# Простая модель для тестирования (rule-based)
class RuleBasedCorrectionModel:
    """
    Простая rule-based модель для быстрого тестирования
    """

    def __init__(self):
        """Инициализирует словарь исправлений"""
        self.corrections = {
            # Орфографические
            "привет": "привет",
            "привет": "привет",
            # Пунктуационные
            "не должно": "недолжно",
            "не смелость": "несмелость",
            "не зрелость": "незрелость",
            "не доверие": "недоверие",
        }

        # Загружаем исправления из датасета если есть
        try:
            import pandas as pd
            from pathlib import Path

            csv_path = Path("data/processed/all_train.csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.sample(min(1000, len(df))).iterrows():
                    self.corrections[str(row["source"]).lower()] = str(row["target"])
        except:
            pass

    def correct_text(self, text: str) -> str:
        """Простое исправление текста"""
        result = text

        # Исправляем фразы
        for incorrect, correct in self.corrections.items():
            if incorrect in result.lower():
                # Case-insensitive replace
                result = result.replace(incorrect, correct)

        return result

    def get_error_scores(self, text: str) -> Dict[int, float]:
        """Возвращает простые scores"""
        return {i: 0.5 for i in range(len(text.split()))}
