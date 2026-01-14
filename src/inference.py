"""
Inference module for text error correction using a sequence-to-sequence model.

This module provides a class `ErrorCorrectionInference` for correcting spelling,
grammar, punctuation, and semantic errors in Russian text using a fine-tuned T5-based model.
It supports loading from a local checkpoint or falling back to a base Hugging Face model.

Key features:
    - Load fine-tuned models with tokenizer and generation config.
    - Generate multiple correction variants with confidence scores.
    - Extract detailed corrections (position, type, confidence).
    - Highlight errors in HTML format for visualization.
    - Batch processing of multiple texts.

Usage:
    >>> corrector = ErrorCorrectionInference(model_path="models/correction_model", device="cuda")
    >>> result = corrector.correct("привет как длеа?")
    >>> print(result.variants[0].corrected_text)
    >>> html_output = corrector.highlight_errors("текст с ошбками")
"""

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
    Represents a single correction in the text.

    Attributes:
        position (int): Start position of the error in the original text.
        original (str): Original erroneous text fragment.
        corrected (str): Corrected version of the fragment.
        confidence (float): Model confidence in the correction (0.0 to 1.0).
        error_type (str): Type of error: 'spelling', 'grammar', 'punctuation', or 'semantics'.
    """
    position: int
    original: str
    corrected: str
    confidence: float
    error_type: str


@dataclass
class CorrectionVariant:
    """
    A single correction variant for the entire input text.

    Attributes:
        corrected_text (str): Full corrected version of the input text.
        score (float): Raw model score (log-probability) for this sequence.
        confidence (float): Normalized confidence score (0.0 to 1.0).
        corrections (List[Correction]): List of individual corrections made.
        error_count (int): Number of corrections applied.
    """
    corrected_text: str
    score: float
    confidence: float
    corrections: List[Correction]
    error_count: int


@dataclass
class CorrectionResult:
    """
    Complete result of the text correction process.

    Attributes:
        original_text (str): The input text provided for correction.
        variants (List[CorrectionVariant]): List of correction variants, sorted by quality (best first).
    """
    original_text: str
    variants: List[CorrectionVariant]


class ErrorCorrectionInference:
    """
    A model wrapper for inference in text error correction tasks.

    Supports loading a fine-tuned seq2seq model (e.g., T5) and generating corrected versions
    of input text with detailed analysis of each correction.

    The model can classify error types and compute confidence scores for corrections.
    It also supports HTML highlighting of detected errors.

    Args:
        model_path (Optional[str]): Path to the saved model directory. If None, uses 't5-small' as fallback.
        device (str): Device to run inference on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initializes the error correction model and tokenizer.

        Args:
            model_path (Optional[str]): Path to directory containing 'model/' and 'tokenizer/' subdirs.
            device (str): Target device for inference. Falls back to 'cpu' if CUDA is not available.
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self._load_model_and_tokenizer(model_path)
        self._setup_generation_parameters()

        self.model.eval()
        logger.info("✅ Model loaded successfully and set to evaluation mode")

    def _load_model_and_tokenizer(self, model_path: Optional[str]) -> None:
        """
        Loads the model and tokenizer from a local path or defaults to 't5-small'.

        Args:
            model_path (Optional[str]): Path to saved model. If None or invalid, uses 't5-small'.

        Raises:
            RuntimeError: If model loading fails for both local and default options.
        """
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"Loading model from local path: {model_path}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_path}/model")
                self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
                self.config = self._load_config(model_path)
            else:
                logger.info("No local model found. Falling back to 't5-small'...")
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
                self.config = {}

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model: {e}")

    def _load_config(self, model_path: str) -> Dict[str, Any]:
        """
        Loads model configuration from config.json.

        Args:
            model_path (str): Directory containing 'config.json'.

        Returns:
            Dict with configuration or empty dict if not found/parsable.
        """
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse config: {e}")
        return {}

    def _setup_generation_parameters(self) -> None:
        """
        Sets up text generation parameters from config or defaults.
        Moves model to the specified device.
        """
        self.max_length = int(self.config.get("max_length", 128))

        self.gen_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 64)),
            "num_beams": int(self.config.get("num_beams", 8)),
            "do_sample": False,
            "no_repeat_ngram_size": int(self.config.get("no_repeat_ngram_size", 3)),
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.05)),
            "early_stopping": True,
        }

        self.model.to(self.device)

    def _scores_to_confidences(self, scores: List[float]) -> List[float]:
        """
        Converts raw model scores to normalized confidence probabilities.

        Applies softmax with numerical stability (log-sum-exp trick).

        Args:
            scores (List[float]): Raw sequence scores from model.generate().

        Returns:
            List[float]: Normalized confidence scores (sum to 1.0).
        """
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        return [exp / total for exp in exps]

    def correct(self, text: str, n: int = 3) -> CorrectionResult:
        """
        Corrects errors in the input text and returns multiple correction variants.

        Args:
            text (str): Input text with potential errors.
            n (int): Number of correction variants to generate. Defaults to 3.

        Returns:
            CorrectionResult: Contains original text and list of ranked variants.

        Raises:
            ValueError: If the input text is empty or too long.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 1000:
            logger.warning(f"Text is long ({len(text)} chars), truncation may occur")

        logger.info(f"Correcting text of length {len(text)}...")

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

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

        corrected_texts = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        scores = outputs.sequences_scores.tolist()
        confidences = self._scores_to_confidences(scores)

        variants: List[CorrectionVariant] = []
        for corrected_text, score, confidence in zip(corrected_texts, scores, confidences):
            corrections = self._find_corrections(text, corrected_text)
            variant = CorrectionVariant(
                corrected_text=corrected_text,
                score=float(score),
                confidence=float(confidence),
                corrections=corrections,
                error_count=len(corrections),
            )
            variants.append(variant)

        variants.sort(key=lambda v: v.score, reverse=True)

        logger.info(f"Generated {len(variants)} correction variants")
        return CorrectionResult(original_text=text, variants=variants)

    def _classify_error(self, original: str, corrected: str) -> str:
        """
        Classifies the type of error based on original and corrected text fragments.

        Args:
            original (str): Original erroneous fragment.
            corrected (str): Corrected fragment.

        Returns:
            str: One of 'punctuation', 'spelling', 'grammar', 'semantics'.
        """
        def normalize(s: str) -> str:
            return "".join(c for c in s if c.isalnum()).lower()

        if normalize(original) == normalize(corrected) and original != corrected:
            return "punctuation"

        similarity = self._calculate_word_similarity(original.lower(), corrected.lower())
        if similarity >= 0.8:
            return "grammar"
        elif similarity >= 0.55:
            return "spelling"
        else:
            return "semantics"

    def _find_corrections(self, original: str, corrected: str) -> List[Correction]:
        """
        Detects differences between original and corrected text and creates correction entries.

        Args:
            original (str): Original input text.
            corrected (str): Model-generated corrected text.

        Returns:
            List[Correction]: List of detected corrections with metadata.
        """
        corrections: List[Correction] = []
        original_words = original.split()
        corrected_words = corrected.split()
        matcher = SequenceMatcher(None, original_words, corrected_words, autojunk=False)
        current_position = 0

        for op, o_start, o_end, c_start, c_end in matcher.get_opcodes():
            if op == "equal":
                continue

            orig_frag = " ".join(original_words[o_start:o_end]).strip()
            corr_frag = " ".join(corrected_words[c_start:c_end]).strip()

            if not orig_frag and not corr_frag:
                continue

            position = original.find(orig_frag.split()[0], current_position) if orig_frag else current_position
            if position == -1:
                position = current_position
            current_position = position + len(orig_frag)

            error_type = self._classify_error(orig_frag, corr_frag)
            confidence = self._calculate_confidence(orig_frag, corr_frag, error_type)

            correction = Correction(
                position=position,
                original=orig_frag,
                corrected=corr_frag,
                confidence=confidence,
                error_type=error_type,
            )
            corrections.append(correction)

        return corrections

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Computes similarity ratio between two strings using SequenceMatcher.

        Args:
            word1 (str): First string.
            word2 (str): Second string.

        Returns:
            float: Similarity ratio in [0.0, 1.0].
        """
        if not word1 or not word2:
            return 0.0
        return SequenceMatcher(None, word1, word2).ratio()

    def _calculate_confidence(self, original: str, corrected: str, error_type: str) -> float:
        """
        Estimates confidence in a correction based on error type and similarity.

        Args:
            original (str): Original fragment.
            corrected (str): Corrected fragment.
            error_type (str): Type of error.

        Returns:
            float: Confidence score in [0.0, 1.0].
        """
        if error_type == "punctuation":
            return 0.95
        elif error_type == "spelling":
            similarity = self._calculate_word_similarity(original, corrected)
            return min(0.95, similarity + 0.2)
        elif error_type == "grammar":
            return 0.7
        else:  # semantics
            return 0.6

    def highlight_errors(self, text: str, variant_index: int = 0) -> str:
        """
        Returns HTML-marked text with errors highlighted.

        Useful for visualization in web interfaces.

        Args:
            text (str): Input text to correct and highlight.
            variant_index (int): Index of the correction variant to use. Defaults to 0 (best).

        Returns:
            str: HTML string with <span class="error error--{type}"> tags.

        Raises:
            IndexError: If the requested variant index is out of range.
        """
        result = self.correct(text, n=max(3, variant_index + 1))
        if variant_index >= len(result.variants):
            raise IndexError(
                f"Variant {variant_index} does not exist. Available: {len(result.variants)}"
            )

        variant = result.variants[variant_index]
        highlighted_text = text
        offset = 0

        for correction in sorted(variant.corrections, key=lambda c: c.position):
            pos = correction.position + offset
            orig_len = len(correction.original)
            before = highlighted_text[:pos]
            error_part = highlighted_text[pos:pos + orig_len]
            after = highlighted_text[pos + orig_len:]

            html_tag = (
                f'<span class="error error--{correction.error_type}" '
                f'title="Correction: {correction.corrected} '
                f'(confidence: {correction.confidence:.2f})">'
                f"{error_part}</span>"
            )

            highlighted_text = before + html_tag + after
            offset += len(html_tag) - len(error_part)

        return highlighted_text

    def get_best_correction(self, text: str) -> str:
        """
        Returns the top correction variant as plain text.

        Args:
            text (str): Input text.

        Returns:
            str: Corrected text with highest model score.
        """
        result = self.correct(text, n=1)
        return result.variants[0].corrected_text

    def batch_correct(self, texts: List[str], n: int = 1) -> List[CorrectionResult]:
        """
        Processes multiple texts in sequence.

        Args:
            texts (List[str]): List of input texts.
            n (int): Number of variants per text.

        Returns:
            List[CorrectionResult]: List of correction results for each text.
        """
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}...")
            try:
                result = self.correct(text, n=n)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                results.append(CorrectionResult(original_text=text, variants=[]))
        return results


# Example usage
if __name__ == "__main__":
    corrector = ErrorCorrectionInference(device="cuda")

    test_text = (
        "Привет как дела? Меня зовут Андрей, я живу в Москве. "
        "Я люблю читат книги и сматреть фильмы. Завтра я иду в кина."
    )

    result = corrector.correct(test_text, n=2)

    print("Original text:", result.original_text)
    print("\nCorrection variants:")

    for i, variant in enumerate(result.variants, 1):
        print(f"\nVariant {i} (confidence: {variant.confidence:.3f}):")
        print(f"Corrected text: {variant.corrected_text}")
        print(f"Errors found: {variant.error_count}")

        if variant.corrections:
            print("Corrections:")
            for corr in variant.corrections:
                print(
                    f"  - '{corr.original}' → '{corr.corrected}' "
                    f"({corr.error_type}, confidence: {corr.confidence:.2f})"
                )

    highlighted = corrector.highlight_errors(test_text)
    print(f"\nHTML with error highlights:\n{highlighted}")
