"""
Архитектура модели для исправления опечаток.
Используем BERT-based модель с fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
)
from typing import List, Tuple, Optional
from pathlib import Path


class TypoCorrectionModel(nn.Module):
    """
    Модель для исправления опечаток на основе BERT.
    Использует masked language modeling для генерации исправлений.
    """

    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        use_seq2seq: bool = False,
    ):
        super().__init__()
        self.use_seq2seq = use_seq2seq

        if use_seq2seq:
            # Seq2Seq подход (например, mT5 или T5)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            # Masked LM подход (BERT)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model_name = model_name

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass модели."""
        if self.use_seq2seq:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        else:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        return outputs

    def predict_corrections(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Предсказывает исправления для текста.
        Возвращает список кортежей (исправленный_текст, уверенность).
        """
        # Токенизация
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        if self.use_seq2seq:
            # Для seq2seq моделей
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    num_return_sequences=top_k,
                    early_stopping=True,
                )

            corrections = []
            for output in outputs:
                corrected_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Упрощённая оценка уверенности
                confidence = 0.8  # Можно улучшить, используя logits
                corrections.append((corrected_text, confidence))

            return corrections
        else:
            # Для BERT MLM - используем более простой подход
            # Находим потенциально проблемные токены и предлагаем исправления
            return self._bert_mlm_corrections(text, inputs, top_k)

    def _bert_mlm_corrections(
        self, text: str, inputs: dict, top_k: int
    ) -> List[Tuple[str, float]]:
        """Исправления через BERT MLM."""
        # Простой подход: находим слова с низкой вероятностью и предлагаем альтернативы
        # Для полноценной реализации нужен более сложный алгоритм

        # Пока возвращаем исходный текст с небольшой модификацией
        # В реальной реализации здесь будет логика поиска ошибок через MLM
        corrections = [(text, 1.0)]

        # Можно добавить логику поиска опечаток через сравнение с словарём
        # или через анализ вероятностей токенов

        return corrections


class SimpleTypoCorrector:
    """
    Упрощённый корректор опечаток, использующий правила и эвристики.
    Используется как fallback или для быстрой демонстрации.
    """

    def __init__(self):
        self.common_typos = {
            "привет как дела": "привет, как дела",
            "он сказал что": "он сказал, что",
            "я думаю что": "я думаю, что",
            "придет": "придёт",
            "пришел": "пришёл",
            "мама сказала что": "мама сказала, что",
            "он знал что": "он знал, что",
            "я вижу что": "я вижу, что",
            "она сказала что": "она сказала, что",
            "когда я пришел": "когда я пришёл",
            "он не знал что": "он не знал, что",
            "я видел что": "я видел, что",
            "она поняла что": "она поняла, что",
        }

    def correct(self, text: str) -> List[Tuple[str, float]]:
        """Исправляет опечатки в тексте."""
        corrected = text

        # Применяем правила
        for typo, correction in self.common_typos.items():
            if typo in corrected.lower():
                corrected = corrected.replace(typo, correction)

        # Добавляем запятые перед "что", "который" и т.д. (упрощённо)
        import re

        corrected = re.sub(
            r"(\w+)\s+(что|который|когда|где|куда)\s+", r"\1, \2 ", corrected
        )

        return [(corrected, 0.9) if corrected != text else (text, 1.0)]


def load_model(model_path: Optional[str] = None, use_seq2seq: bool = False):
    """Загружает модель из файла или создаёт новую."""
    if model_path and Path(model_path).exists():
        print(f"Загружаем модель из {model_path}")
        model = TypoCorrectionModel(use_seq2seq=use_seq2seq)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        print("Создаём новую модель")
        return TypoCorrectionModel(use_seq2seq=use_seq2seq)


if __name__ == "__main__":
    # Тестирование модели
    model = TypoCorrectionModel()
    test_text = "привет как дела"
    corrections = model.predict_corrections(test_text)
    print(f"Исходный текст: {test_text}")
    print(f"Исправления: {corrections}")
