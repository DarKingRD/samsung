"""
Модуль для инференса модели исправления опечаток.
"""

import torch
from pathlib import Path
import json
import re
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model import SimpleTypoCorrector

# Пытаемся импортировать pymorphy2, но не критично если не получится
try:
    import pymorphy2

    PYMORPHY2_AVAILABLE = True
except (ImportError, AttributeError):
    PYMORPHY2_AVAILABLE = False
    pymorphy2 = None

# Пытаемся импортировать Levenshtein для поиска похожих слов
try:
    from Levenshtein import distance as levenshtein_distance_func

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    try:
        import Levenshtein

        levenshtein_distance_func = Levenshtein.distance
        LEVENSHTEIN_AVAILABLE = True
    except ImportError:
        LEVENSHTEIN_AVAILABLE = False

        # Простая реализация расстояния Левенштейна
        def levenshtein_distance_func(s1, s2):
            """Простая реализация расстояния Левенштейна."""
            if len(s1) < len(s2):
                return levenshtein_distance_func(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]


MODELS_DIR = Path("models")


class TypoCorrectionInference:
    """Класс для исправления опечаток в тексте."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or (MODELS_DIR / "typo_corrector_model")
        self.dict_path = MODELS_DIR / "typo_dict.json"

        # Пытаемся загрузить обученную модель
        self.model = None
        self.tokenizer = None
        self.use_model = False

        if self.model_path.exists() and (self.model_path / "config.json").exists():
            try:
                print(f"Загружаем модель из {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
                self.model.eval()
                self.use_model = True
                print("Модель загружена успешно")
            except Exception as e:
                print(f"Не удалось загрузить модель: {e}")
                print("Используем упрощённый корректор")

        
        # Загружаем словарь опечаток
        self.typo_dict = {}
        self.correct_words = set()  # Множество правильных слов
        if self.dict_path.exists():
            with open(self.dict_path, "r", encoding="utf-8") as f:
                self.typo_dict = json.load(f)
                # Создаём множество правильных слов из словаря
                self.correct_words = set(self.typo_dict.values())
            print(f"Загружен словарь из {len(self.typo_dict)} пар")

        # Инициализируем морфологический анализатор для проверки слов
        self.morph = None
        try:
            # Пытаемся загрузить pymorphy2
            self.morph = pymorphy2.MorphAnalyzer()
            print("Морфологический анализатор загружен")
        except (AttributeError, ImportError, Exception) as e:
            # Игнорируем ошибки загрузки (Python 3.13+ несовместимость)
            # Система будет работать без морфологического анализатора
            if "getargspec" not in str(e):
                print(f"Не удалось загрузить морфологический анализатор: {e}")
            self.morph = None

        # Инициализируем простой корректор
        self.simple_corrector = SimpleTypoCorrector()

    def correct_text(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Исправляет опечатки в тексте.
        Возвращает список кортежей (исправленный_текст, уверенность).
        """
        if not text.strip():
            return [(text, 1.0)]

        # Используем обученную модель если доступна
        if self.use_model and self.model:
            return self._correct_with_model(text, top_k)

        # Иначе используем комбинацию словаря и правил
        return self._correct_with_dict_and_rules(text, top_k)

    def _correct_with_model(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Исправление с помощью обученной модели."""
        # Добавил префикс для задачи исправления
        # Это помогает понять моделе, что нужно исправить опечатки
        prompts = [
            f"Correct typos: {text}",
        ]
        
        all_corrections = []
        
        
        for prompt in prompts[:1]:  # Используем только первый промпт для начала
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,  # Уменьшим для скорости
                )

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=3,  # Уменьшим для скорости
                        num_return_sequences=min(top_k, 3),
                        early_stopping=True,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0,
                    )

                for i, output in enumerate(outputs):
                    corrected_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    
                    # Удаляем возможные префиксы
                    for prefix in ["Correct typos:"]:
                        if corrected_text.startswith(prefix):
                            corrected_text = corrected_text[len(prefix):].strip()
                    
                    # Удаляем кавычки если есть
                    corrected_text = corrected_text.strip('"').strip("'").strip()
                    
                    # Пропускаем пустые или слишком короткие результаты
                    if not corrected_text or len(corrected_text) < len(text) * 0.3:
                        continue
                    
                    # Рассчитываем уверенность
                    confidence = self._calculate_confidence(text, corrected_text, i)
                    
                    all_corrections.append((corrected_text, confidence))
                    
            except Exception as e:
                print(f"Ошибка при генерации: {e}")
                continue
        
        # Убираем дубликаты
        unique_corrections = []
        seen_texts = set()
        for text_corr, conf in all_corrections:
            if text_corr not in seen_texts:
                seen_texts.add(text_corr)
                unique_corrections.append((text_corr, conf))
        
        # Сортируем по уверенности
        unique_corrections.sort(key=lambda x: x[1], reverse=True)
        
        # Если нет хороших исправлений, используем словарь
        if not unique_corrections or all(conf < 0.5 for _, conf in unique_corrections):
            dict_result = self._correct_with_dict_and_rules(text, top_k)
            if dict_result and dict_result[0][0] != text:
                return dict_result
        
        return unique_corrections[:top_k] if unique_corrections else [(text, 1.0)]

    def _calculate_confidence(self, original: str, corrected: str, rank: int) -> float:
        """Рассчитывает уверенность в исправлении."""
        # Базовая уверенность
        base_confidence = 0.8 - (rank * 0.1)
        
        # Штраф за слишком большие изменения
        len_ratio = len(corrected) / max(len(original), 1)
        if len_ratio < 0.5 or len_ratio > 2.0:
            base_confidence *= 0.5
        
        # Поощрение за сохранение структуры
        original_words = original.split()
        corrected_words = corrected.split()
        
        if len(original_words) > 0 and len(corrected_words) > 0:
            # Процент общих слов
            common_words = set(w.lower() for w in original_words) & set(w.lower() for w in corrected_words)
            word_similarity = len(common_words) / max(len(original_words), 1)
            base_confidence *= (0.3 + 0.7 * word_similarity)
        
        return max(0.1, min(0.95, base_confidence))

    def _find_similar_word(self, word: str, max_distance: int = 2) -> Tuple[str, float]:
        """
        Находит похожее слово в словаре опечаток.
        Возвращает (исправление, уверенность) или (word, 0.0) если не найдено.
        """
        word_lower = word.lower()

        # Сначала проверяем точное совпадение в словаре опечаток
        if word_lower in self.typo_dict:
            return (self.typo_dict[word_lower], 0.9)

        # Ищем похожие слова через расстояние Левенштейна
        # Ищем как среди опечаток (ключи), так и среди правильных слов (значения)
        best_match = None
        best_distance = max_distance + 1
        best_correction = None

        # Поиск среди опечаток (ключи словаря)
        for typo, correction in self.typo_dict.items():
            # Вычисляем расстояние до опечатки
            distance = levenshtein_distance_func(word_lower, typo)

            # Если расстояние меньше текущего лучшего
            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_match = typo
                best_correction = correction

        # Также ищем среди правильных слов (значения словаря)
        # Это поможет найти "скачать" для "скачть"
        for correction in self.correct_words:
            distance = levenshtein_distance_func(word_lower, correction.lower())

            # Если нашли более близкое слово среди правильных
            if distance < best_distance and distance <= max_distance:
                best_distance = distance
                best_match = correction
                best_correction = correction

        if best_match:
            # Уверенность зависит от расстояния
            confidence = max(0.5, 1.0 - (best_distance / max_distance))
            return (best_correction, confidence)

        return (word, 0.0)

    def _correct_with_dict_and_rules(
        self, text: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """Исправление с помощью словаря и правил."""
        corrected = text
        changes_made = False

        # Разбиваем текст на слова и проверяем каждое
        words = re.findall(r"\b\w+\b", text)

        for word in words:
            word_lower = word.lower()

            # Пропускаем короткие слова
            if len(word_lower) <= 2:
                continue

            # Сначала проверяем точное совпадение в словаре опечаток
            if word_lower in self.typo_dict:
                # Если слово есть в словаре опечаток - это точно опечатка
                correction = self.typo_dict[word_lower]
                pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                if pattern.search(corrected):
                    corrected = pattern.sub(correction, corrected)
                    changes_made = True
            else:
                # Если слова нет в словаре опечаток, ищем похожие слова
                # Ищем среди правильных слов (значения словаря)
                similar_correction, confidence = self._find_similar_word(
                    word, max_distance=2
                )
                if confidence > 0.5 and similar_correction != word_lower:
                    # Нашли похожее слово - заменяем
                    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                    if pattern.search(corrected):
                        corrected = pattern.sub(similar_correction, corrected)
                        changes_made = True

        # Применяем правила из простого корректора только если есть изменения
        if changes_made:
            simple_result = self.simple_corrector.correct(corrected)
            if simple_result:
                corrected = simple_result[0][0]

        # Грамматические правила (только для пунктуации, не для слов)
        corrected = self._apply_grammar_rules(corrected)

        # Если текст не изменился, значит ошибок не было
        if corrected == text:
            return [(text, 1.0)]

        confidence = 0.8 if changes_made else 1.0
        return [(corrected, confidence)]

    def _apply_grammar_rules(self, text: str) -> str:
        """Применяет грамматические правила."""
        # Добавление запятых перед союзами
        text = re.sub(
            r"(\w+)\s+(что|который|которая|которое|которые|когда|где|куда|откуда)\s+",
            r"\1, \2 ",
            text,
        )

        # Исправление "придет" -> "придёт" и подобных
        text = re.sub(r"\bпридет\b", "придёт", text, flags=re.IGNORECASE)
        text = re.sub(r"\bпришел\b", "пришёл", text, flags=re.IGNORECASE)
        text = re.sub(r"\bпришла\b", "пришла", text, flags=re.IGNORECASE)
        text = re.sub(r"\bпришли\b", "пришли", text, flags=re.IGNORECASE)

        # Исправление "как дела" -> "как дела" (добавление запятой)
        text = re.sub(r"(\w+),?\s+как дела", r"\1, как дела", text, flags=re.IGNORECASE)

        # Дополнительные грамматические правила
        # Запятая перед "который" и его формами
        text = re.sub(
            r"(\w+)\s+(который|которая|которое|которые)\s+",
            r"\1, \2 ",
            text,
            flags=re.IGNORECASE,
        )

        return text

    def find_errors(self, text: str) -> List[Dict]:
        """
        Находит ошибки в тексте и возвращает их позиции.
        Возвращает список словарей с информацией об ошибках.
        """
        errors = []

        if not text.strip():
            return errors

        # Разбиваем текст на слова с позициями
        words_with_pos = []
        for match in re.finditer(r"\b\w+\b", text):
            word = match.group()
            words_with_pos.append(
                {"word": word, "start": match.start(), "end": match.end()}
            )

        # Проверяем каждое слово
        for word_info in words_with_pos:
            word = word_info["word"]
            word_lower = word.lower()

            # Пропускаем короткие слова
            if len(word) <= 2:
                continue

            # Проверяем, является ли слово опечаткой
            is_typo = False
            suggestions = []

            # Проверяем в словаре опечаток
            if word_lower in self.typo_dict:
                correction = self.typo_dict[word_lower]
                is_typo = True
                suggestions.append(correction)
            else:
                # Ищем похожие слова
                similar_correction, confidence = self._find_similar_word(
                    word, max_distance=2
                )
                if confidence > 0.5 and similar_correction != word_lower:
                    is_typo = True
                    suggestions.append(similar_correction)

            if is_typo and suggestions:
                errors.append(
                    {
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "original": word,
                        "suggestions": suggestions[:3],
                        "confidence": 0.7,
                    }
                )

        # Если не нашли ошибок по словам, проверяем общую коррекцию
        if not errors:
            corrections = self.correct_text(text)
            if corrections and corrections[0][0] != text:
                # Проверяем, что изменения действительно нужны
                # (например, только пунктуация)
                original_words = set(re.findall(r"\b\w+\b", text.lower()))
                corrected_words = set(re.findall(r"\b\w+\b", corrections[0][0].lower()))

                # Если изменились слова (не только пунктуация), добавляем ошибку
                if original_words != corrected_words:
                    errors.append(
                        {
                            "start": 0,
                            "end": len(text),
                            "original": text,
                            "suggestions": [corr[0] for corr in corrections[:3]],
                            "confidence": corrections[0][1] if corrections else 0.8,
                        }
                    )

        return errors


def load_corrector(model_path: str = None) -> TypoCorrectionInference:
    """Загружает корректор опечаток."""
    return TypoCorrectionInference(model_path)


if __name__ == "__main__":
    # Тестирование
    corrector = load_corrector()

    test_texts = [
        "привет как дела",
        "он сказал что придет",
        "я думаю что это правильно",
    ]

    for text in test_texts:
        print(f"\nИсходный текст: {text}")
        corrections = corrector.correct_text(text)
        print(f"Исправления: {corrections}")
        errors = corrector.find_errors(text)
        print(f"Найдено ошибок: {len(errors)}")
