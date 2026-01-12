import math
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """
    Информация об одном исправлении в тексте.

    Attributes:
        position (int): Позиция начала ошибки в оригинальном тексте
        original (str): Оригинальный фрагмент с ошибкой
        corrected (str): Исправленный вариант
        confidence (float): Уверенность в исправлении (0-1)
        error_type (str): Тип ошибки: spelling, grammar, punctuation, semantics
    """

    position: int
    original: str
    corrected: str
    confidence: float
    error_type: str


@dataclass
class CorrectionVariant:
    """
    Один вариант исправления всего текста.

    Attributes:
        corrected_text (str): Полный исправленный текст
        score (float): Оценка модели для этого варианта
        confidence (float): Общая уверенность в исправлении
        corrections (List[Correction]): Список отдельных исправлений
        error_count (int): Количество найденных ошибок
    """

    corrected_text: str
    score: float
    confidence: float
    corrections: List[Correction]
    error_count: int


@dataclass
class CorrectionResult:
    """
    Результат исправления текста.

    Attributes:
        original_text (str): Оригинальный текст
        variants (List[CorrectionVariant]): Варианты исправления, отсортированные по качеству
    """

    original_text: str
    variants: List[CorrectionVariant]


class ErrorCorrectionInference:
    """
    Модель для исправления ошибок в тексте.

    Поддерживает загрузку предобученных моделей или использование базовой T5.
    Обрабатывает орфографические, грамматические, пунктуационные и семантические ошибки.

    Args:
        model_path (Optional[str]): Путь к сохраненной модели. Если None, используется t5-small
        device (str): Устройство для вычислений (cuda/cpu)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Инициализация модели исправления ошибок.

        Args:
            model_path: Путь к директории с сохраненной моделью (опционально)
            device: Устройство для загрузки модели ('cuda' или 'cpu')
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        logger.info(f"Использую устройство: {self.device}")

        # Загрузка токенизатора и модели
        self._load_model_and_tokenizer(model_path)

        # Настройка параметров генерации
        self._setup_generation_parameters()

        self.model.eval()
        logger.info("✅ Модель успешно загружена и готова к работе")

    def _load_model_and_tokenizer(self, model_path: Optional[str]) -> None:
        """
        Загружает модель и токенизатор из указанного пути или использует базовую модель.

        Args:
            model_path: Путь к директории с моделью или None для базовой модели
        """
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"Загружаю модель из локальной директории: {model_path}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    f"{model_path}/model"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"{model_path}/tokenizer"
                )
                self.config = self._load_config(model_path)
            else:
                logger.info(
                    "Локальная модель не найдена. Загружаю базовую модель t5-small..."
                )
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
                self.config = {}

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise RuntimeError(f"Не удалось загрузить модель: {e}")

    def _load_config(self, model_path: str) -> Dict[str, Any]:
        """
        Загружает конфигурацию модели из JSON файла.

        Args:
            model_path: Путь к директории с моделью

        Returns:
            Словарь с параметрами конфигурации или пустой словарь при ошибке
        """
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Не удалось прочитать конфигурацию: {e}")
        return {}

    def _setup_generation_parameters(self) -> None:
        """
        Настраивает параметры генерации текста на основе конфигурации или значений по умолчанию.
        """
        self.max_length = int(self.config.get("max_length", 128))

        self.gen_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 64)),
            "num_beams": int(self.config.get("num_beams", 8)),
            "do_sample": False,  # Используем beam search для стабильности
            "no_repeat_ngram_size": int(self.config.get("no_repeat_ngram_size", 3)),
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.05)),
            "early_stopping": True,
        }

        # Перемещаем модель на выбранное устройство
        self.model.to(self.device)

    def _scores_to_confidences(self, scores: List[float]) -> List[float]:
        """
        Преобразует raw scores модели в вероятностное распределение.

        Args:
            scores: Список оценок от модели

        Returns:
            Нормализованные уверенности (сумма = 1.0)
        """
        if not scores:
            return []

        # Для численной стабильности вычитаем максимум
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)

        # Нормализация
        return [exp_score / total for exp_score in exps]

    def correct(self, text: str, n: int = 3) -> CorrectionResult:
        """
        Исправляет ошибки в тексте и возвращает несколько вариантов исправления.

        Args:
            text: Текст для исправления
            n: Количество возвращаемых вариантов (по умолчанию 3)

        Returns:
            CorrectionResult с оригинальным текстом и вариантами исправления

        Raises:
            ValueError: Если текст слишком длинный или пустой
        """
        if not text or not text.strip():
            raise ValueError("Текст не может быть пустым")

        if len(text) > 1000:
            logger.warning(
                f"Текст довольно длинный ({len(text)} символов), возможны потери при токенизации"
            )

        logger.info(f"Исправляю текст длиной {len(text)} символов...")

        # Токенизация входного текста
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Генерация исправленных вариантов
        num_beams = max(self.gen_kwargs.get("num_beams", 8), n)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **{**self.gen_kwargs, "num_beams": num_beams},
                num_return_sequences=n,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Декодирование сгенерированного текста
        corrected_texts = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        scores = outputs.sequences_scores.tolist()
        confidences = self._scores_to_confidences(scores)

        # Создание вариантов исправления
        variants: List[CorrectionVariant] = []
        for corrected_text, score, confidence in zip(
            corrected_texts, scores, confidences
        ):
            corrections = self._find_corrections(text, corrected_text)

            variant = CorrectionVariant(
                corrected_text=corrected_text,
                score=float(score),
                confidence=float(confidence),
                corrections=corrections,
                error_count=len(corrections),
            )
            variants.append(variant)

        # Сортировка по качеству (лучшие варианты первыми)
        variants.sort(key=lambda v: v.score, reverse=True)

        logger.info(f"Сгенерировано {len(variants)} вариантов исправления")
        return CorrectionResult(original_text=text, variants=variants)

    def _classify_error(self, original: str, corrected: str) -> str:
        """
        Классифицирует тип ошибки на основе сравнения оригинального и исправленного текста.

        Args:
            original: Оригинальный текст
            corrected: Исправленный текст

        Returns:
            Тип ошибки: 'punctuation', 'spelling', 'grammar' или 'semantics'
        """

        # Удаляем пунктуацию и приводим к нижнему регистру для сравнения
        def normalize_text(s: str) -> str:
            return "".join(char for char in s if char.isalnum()).lower()

        # Проверка на пунктуационные ошибки
        if (
            normalize_text(original) == normalize_text(corrected)
            and original != corrected
        ):
            return "punctuation"

        # Вычисление схожести слов
        similarity = self._calculate_word_similarity(
            original.lower(), corrected.lower()
        )

        # Классификация на основе степени схожести
        if similarity >= 0.8:
            return "grammar"
        elif similarity >= 0.55:
            return "spelling"
        else:
            return "semantics"

    def _find_corrections(self, original: str, corrected: str) -> List[Correction]:
        """
        Находит различия между оригинальным и исправленным текстом.

        Args:
            original: Оригинальный текст
            corrected: Исправленный текст

        Returns:
            Список объектов Correction с информацией об исправлениях
        """
        corrections: List[Correction] = []

        original_words = original.split()
        corrected_words = corrected.split()

        # Используем SequenceMatcher для поиска различий
        matcher = SequenceMatcher(None, original_words, corrected_words, autojunk=False)
        current_position = 0

        for (
            operation_type,
            orig_start,
            orig_end,
            corr_start,
            corr_end,
        ) in matcher.get_opcodes():
            if operation_type == "equal":
                # Пропускаем идентичные части
                continue

            # Извлекаем измененные фрагменты
            original_fragment = " ".join(original_words[orig_start:orig_end]).strip()
            corrected_fragment = " ".join(corrected_words[corr_start:corr_end]).strip()

            # Пропускаем пустые изменения
            if not original_fragment and not corrected_fragment:
                continue

            # Находим позицию изменения в оригинальном тексте
            if original_fragment:
                # Ищем начало первого слова фрагмента
                position = original.find(original_fragment.split()[0], current_position)
                if position == -1:
                    position = current_position
                current_position = position + len(original_fragment)
            else:
                # Вставка нового текста
                position = current_position

            # Определяем тип ошибки и уверенность
            error_type = self._classify_error(original_fragment, corrected_fragment)
            confidence = self._calculate_confidence(
                original_fragment, corrected_fragment, error_type
            )

            # Создаем объект исправления
            correction = Correction(
                position=position,
                original=original_fragment,
                corrected=corrected_fragment,
                confidence=confidence,
                error_type=error_type,
            )
            corrections.append(correction)

        return corrections

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Вычисляет схожесть между двумя словами/фразами.

        Args:
            word1: Первое слово/фраза
            word2: Второе слово/фраза

        Returns:
            Коэффициент схожести от 0.0 до 1.0
        """
        if not word1 or not word2:
            return 0.0

        # Используем SequenceMatcher для вычисления схожести
        return SequenceMatcher(None, word1, word2).ratio()

    def _calculate_confidence(
        self, original: str, corrected: str, error_type: str
    ) -> float:
        """
        Вычисляет уверенность в исправлении на основе типа ошибки и схожести текстов.

        Args:
            original: Оригинальный текст
            corrected: Исправленный текст
            error_type: Тип ошибки

        Returns:
            Уверенность в исправлении от 0.0 до 1.0
        """
        if error_type == "punctuation":
            return 0.95  # Высокая уверенность для пунктуации

        elif error_type == "spelling":
            # Для орфографии уверенность зависит от схожести
            similarity = self._calculate_word_similarity(original, corrected)
            return min(0.95, similarity + 0.2)

        elif error_type == "grammar":
            return 0.7  # Средняя уверенность для грамматики

        else:  # semantics
            return 0.6  # Ниже уверенность для семантических изменений

    def highlight_errors(self, text: str, variant_index: int = 0) -> str:
        """
        Возвращает текст с HTML-разметкой для подсветки ошибок.

        Args:
            text: Текст для исправления и подсветки
            variant_index: Индекс варианта исправления (по умолчанию первый)

        Returns:
            Текст с HTML-разметкой для подсветки ошибок

        Raises:
            IndexError: Если запрошенный вариант не существует
        """
        # Получаем исправления
        result = self.correct(text, n=max(3, variant_index + 1))

        # Проверяем доступность запрошенного варианта
        if variant_index >= len(result.variants):
            raise IndexError(
                f"Вариант {variant_index} не существует. Доступно {len(result.variants)} вариантов."
            )

        variant = result.variants[variant_index]

        # Создаем HTML с подсветкой ошибок
        highlighted_text = text
        offset = 0  # Смещение из-за добавления HTML-тегов

        # Сортируем исправления по позиции для последовательной обработки
        for correction in sorted(variant.corrections, key=lambda c: c.position):
            position = correction.position + offset

            # Разделяем текст на части
            before = highlighted_text[:position]
            error_part = highlighted_text[
                position : position + len(correction.original)
            ]
            after = highlighted_text[position + len(correction.original) :]

            # Создаем HTML-разметку с подсказкой
            html_tag = (
                f'<span class="error error--{correction.error_type}" '
                f'title="Исправление: {correction.corrected} '
                f'(уверенность: {correction.confidence:.2f})">'
                f"{error_part}</span>"
            )

            # Собираем текст с подсветкой
            highlighted_text = before + html_tag + after

            # Обновляем смещение для следующих исправлений
            offset += len(html_tag) - len(error_part)

        return highlighted_text

    def get_best_correction(self, text: str) -> str:
        """
        Возвращает лучший вариант исправления текста.

        Args:
            text: Текст для исправления

        Returns:
            Лучший исправленный вариант текста
        """
        result = self.correct(text, n=1)
        return result.variants[0].corrected_text

    def batch_correct(self, texts: List[str], n: int = 1) -> List[CorrectionResult]:
        """
        Исправляет несколько текстов за один вызов.

        Args:
            texts: Список текстов для исправления
            n: Количество вариантов для каждого текста

        Returns:
            Список результатов исправления
        """
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Обрабатываю текст {i}/{len(texts)}...")
            try:
                result = self.correct(text, n=n)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при обработке текста {i}: {e}")
                # Возвращаем пустой результат при ошибке
                results.append(CorrectionResult(original_text=text, variants=[]))

        return results


# Пример использования
if __name__ == "__main__":
    # Инициализация модели
    corrector = ErrorCorrectionInference(device="cuda")

    # Пример текста с ошибками
    test_text = (
        "Привет как дела? Меня зовут Андрей, я живу в Москве. "
        "Я люблю читат книги и сматреть фильмы. Завтра я иду в кина."
    )

    # Исправление текста
    result = corrector.correct(test_text, n=2)

    print("Оригинальный текст:", result.original_text)
    print("\nВарианты исправления:")

    for i, variant in enumerate(result.variants, 1):
        print(f"\nВариант {i} (уверенность: {variant.confidence:.3f}):")
        print(f"Исправленный текст: {variant.corrected_text}")
        print(f"Найдено ошибок: {variant.error_count}")

        if variant.corrections:
            print("Детали исправлений:")
            for corr in variant.corrections:
                print(
                    f"  - '{corr.original}' -> '{corr.corrected}' "
                    f"({corr.error_type}, уверенность: {corr.confidence:.2f})"
                )

    # HTML-подсветка ошибок
    highlighted = corrector.highlight_errors(test_text)
    print(f"\nHTML с подсветкой ошибок:\n{highlighted}")
