import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from typing import List
import json
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# КЛАССЫ
# ============================================================================


@dataclass
class Correction:
    """Один вариант исправления"""
    position: int           # позиция в тексте
    original: str           # оригинальный текст
    corrected: str          # исправленный текст
    confidence: float       # уверенность (0-1)
    error_type: str         # тип ошибки
    alternatives: List[str] = None  # альтернативные варианты


@dataclass
class CorrectionResult:
    """Результат исправления текста"""
    original_text: str              # оригинальный текст
    corrected_text: str             # исправленный текст
    corrections: List[Correction]   # список исправлений
    error_count: int                # количество ошибок


class ErrorCorrectionInference:
    """Инференс модели коррекции ошибок"""

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        Args:
            model_path: путь к сохраненной модели
            device: cuda или cpu
        """
        self.device = device

        if model_path and Path(model_path).exists():
            logger.info(f"Загружаю модель из {model_path}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_path}/model")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

            # Загружаем конфиг
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {'max_length': 128}
        else:
            # Используем базовую модель
            logger.info("Загружаю базовую модель t5-small...")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.config = {'max_length': 128}

        # === НОВОЕ: префикс задачи + текстовая инструкция ===
        self.task_prefix = self.config.get("task_prefix", "grammar: ")
        self.lang_instruction = self.config.get(
            "lang_instruction",
            "Исправь ошибки, сохрани смысл и стиль. Верни только исправленный текст."
        )

        self.model.to(device)
        self.model.eval()
        logger.info("✅ Модель готова")

    def _build_input(self, text: str) -> str:
        """
        Формируем вход для модели.
        Важно: instruction + сам текст, но без длинного промпта внутри выхода.
        """
        return f"{self.task_prefix}{text}"

    def correct(self, text: str, return_alternatives: bool = False) -> CorrectionResult:
        """
        Исправляет текст

        Args:
            text: текст с ошибками
            return_alternatives: возвращать ли альтернативные варианты

        Returns:
            CorrectionResult с исправлениями
        """
        max_length = self.config.get('max_length', 128)

        # ВАЖНО: в модель подаём ровно то, на чём обучали — сам текст с ошибками
        input_text = text

        # Кодируем текст
        inputs = self.tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",    # можно оставить padding='max_length', если так тренировал
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Генерируем исправленный текст
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                # ограничиваем "сколько нового" модель может наговорить
                max_new_tokens=64,

                # усиливаем поиск вариантов
                num_beams=8,

                # без рандома
                do_sample=False,

                # уменьшаем повторы
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,

                early_stopping=True,
            )

        # Декодируем результат
        if isinstance(outputs, dict):
            corrected_text = self.tokenizer.decode(
                outputs['sequences'][0],
                skip_special_tokens=True
            )
        else:
            corrected_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        # Находим различия между оригиналом и исправленным текстом
        corrections = self._find_corrections(text, corrected_text)

        result = CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=corrections,
            error_count=len(corrections)
        )

        return result

    def _classify_error(self, original: str, corrected: str) -> str:
        # Пунктуация: если буквенно-цифровая часть одинаковая
        def strip_punct(s: str) -> str:
            return "".join(c for c in s if c.isalnum()).lower()

        if strip_punct(original) == strip_punct(corrected) and original != corrected:
            return "punctuation"

        similarity = self._word_similarity(original.lower(), corrected.lower())

        # Грамматика: очень похоже (окончание/форма)
        if similarity >= 0.8:
            return "grammar"

        # Орфография: похоже, но не настолько (опечатка)
        if similarity >= 0.55:
            return "spelling"

        return "semantics"

    def _find_corrections(self, original: str, corrected: str) -> List[Correction]:
        corrections = []

        original_words = original.split()
        corrected_words = corrected.split()

        matcher = SequenceMatcher(None, original_words, corrected_words)

        original_pos = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            # Берём диапазоны слов, а не один элемент
            orig_chunk = " ".join(original_words[i1:i2]).strip()
            corr_chunk = " ".join(corrected_words[j1:j2]).strip()

            if not orig_chunk and not corr_chunk:
                continue

            # позиция: ищем начало куска в оригинальном тексте
            if orig_chunk:
                pos = original.find(orig_chunk.split()[0], original_pos)
                if pos == -1:
                    pos = original_pos
                original_pos = pos + len(orig_chunk)
            else:
                pos = original_pos

            error_type = self._classify_error(orig_chunk, corr_chunk)
            confidence = self._calculate_confidence(orig_chunk, corr_chunk, error_type)

            corrections.append(Correction(
                position=pos,
                original=orig_chunk,
                corrected=corr_chunk,
                confidence=confidence,
                error_type=error_type,
            ))

        return corrections
    

    def _word_similarity(self, word1: str, word2: str) -> float:
        """Считает сходство двух слов (0-1)"""
        if not word1 or not word2:
            return 0.0

        matcher = SequenceMatcher(None, word1, word2)
        return matcher.ratio()

    def _calculate_confidence(self, original: str, corrected: str, error_type: str) -> float:
        """Считает уверенность в исправлении"""

        if error_type == 'punctuation':
            return 0.95
        elif error_type == 'spelling':
            similarity = self._word_similarity(original, corrected)
            return min(0.95, similarity + 0.2)
        elif error_type == 'grammar':
            return 0.7
        else:
            return 0.6

    def correct_batch(self, texts: List[str]) -> List[CorrectionResult]:
        """Исправляет несколько текстов"""
        return [self.correct(text) for text in texts]

    def highlight_errors(self, text: str) -> str:
        """
        Возвращает текст с выделенными ошибками (HTML для веб-приложения)
        """
        result = self.correct(text)

        highlighted = text
        offset = 0

        for correction in sorted(result.corrections, key=lambda x: x.position):
            pos = correction.position + offset

            before = highlighted[:pos]
            match = highlighted[pos:pos + len(correction.original)]
            after = highlighted[pos + len(correction.original):]

            html = f'<span class="error error-{correction.error_type}" title="{correction.corrected}">{match}</span>'
            highlighted = before + html + after

            offset += len(html) - len(match)

        return highlighted
