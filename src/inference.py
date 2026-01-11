import math
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Correction:
    position: int
    original: str
    corrected: str
    confidence: float
    error_type: str


@dataclass
class CorrectionVariant:
    corrected_text: str
    score: float
    confidence: float
    corrections: List[Correction]
    error_count: int


@dataclass
class CorrectionResult:
    original_text: str
    variants: List[CorrectionVariant]


class ErrorCorrectionInference:
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device

        if model_path and Path(model_path).exists():
            logger.info(f"Загружаю модель из {model_path}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_path}/model")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

            config_path = Path(model_path) / "config.json"
            self.config = {}
            if config_path.exists():
                self.config = config_path.read_text(encoding="utf-8")
                import json

                self.config = json.loads(self.config)
        else:
            logger.info("Загружаю базовую модель t5-small...")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.config = {}

        self.max_length = int(self.config.get("max_length", 128))
        self.gen_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 64)),
            "num_beams": int(self.config.get("num_beams", 8)),
            "do_sample": False,  # beam
            "no_repeat_ngram_size": int(self.config.get("no_repeat_ngram_size", 3)),
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.05)),
            "early_stopping": True,
        }

        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ Модель готова")

    def _scores_to_confidences(self, scores: List[float]) -> List[float]:
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps) or 1.0
        return [e / z for e in exps]

    def correct(self, text: str, n: int = 3) -> CorrectionResult:
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        num_beams = max(int(self.gen_kwargs.get("num_beams", 8)), n)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **{**self.gen_kwargs, "num_beams": num_beams},
                num_return_sequences=n,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.tolist()
        confs = self._scores_to_confidences(scores)

        variants: List[CorrectionVariant] = []
        for t, s, c in zip(texts, scores, confs):
            corrs = self._find_corrections(text, t)
            variants.append(
                CorrectionVariant(
                    corrected_text=t,
                    score=float(s),
                    confidence=float(c),
                    corrections=corrs,
                    error_count=len(corrs),
                )
            )

        variants.sort(key=lambda v: v.score, reverse=True)
        return CorrectionResult(original_text=text, variants=variants)

    def _classify_error(self, original: str, corrected: str) -> str:
        def strip_punct(s: str) -> str:
            return "".join(c for c in s if c.isalnum()).lower()

        if strip_punct(original) == strip_punct(corrected) and original != corrected:
            return "punctuation"

        similarity = self._word_similarity(original.lower(), corrected.lower())
        if similarity >= 0.8:
            return "grammar"
        if similarity >= 0.55:
            return "spelling"
        return "semantics"

    def _find_corrections(self, original: str, corrected: str) -> List[Correction]:
        corrections: List[Correction] = []

        original_words = original.split()
        corrected_words = corrected.split()
        matcher = SequenceMatcher(None, original_words, corrected_words)

        original_pos = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            orig_chunk = " ".join(original_words[i1:i2]).strip()
            corr_chunk = " ".join(corrected_words[j1:j2]).strip()
            if not orig_chunk and not corr_chunk:
                continue

            if orig_chunk:
                pos = original.find(orig_chunk.split()[0], original_pos)
                if pos == -1:
                    pos = original_pos
                original_pos = pos + len(orig_chunk)
            else:
                pos = original_pos

            error_type = self._classify_error(orig_chunk, corr_chunk)
            confidence = self._calculate_confidence(orig_chunk, corr_chunk, error_type)

            corrections.append(
                Correction(
                    position=pos,
                    original=orig_chunk,
                    corrected=corr_chunk,
                    confidence=confidence,
                    error_type=error_type,
                )
            )

        return corrections

    def _word_similarity(self, word1: str, word2: str) -> float:
        if not word1 or not word2:
            return 0.0
        return SequenceMatcher(None, word1, word2).ratio()

    def _calculate_confidence(
        self, original: str, corrected: str, error_type: str
    ) -> float:
        if error_type == "punctuation":
            return 0.95
        if error_type == "spelling":
            similarity = self._word_similarity(original, corrected)
            return min(0.95, similarity + 0.2)
        if error_type == "grammar":
            return 0.7
        return 0.6

    def highlight_errors(self, text: str, variant_index: int = 0) -> str:
        result = self.correct(text, n=max(3, variant_index + 1))
        v = result.variants[variant_index]

        highlighted = text
        offset = 0
        for c in sorted(v.corrections, key=lambda x: x.position):
            pos = c.position + offset
            before = highlighted[:pos]
            match = highlighted[pos : pos + len(c.original)]
            after = highlighted[pos + len(c.original) :]

            html = f'<span class="error error--{c.error_type}" title="{c.corrected}">{match}</span>'
            highlighted = before + html + after
            offset += len(html) - len(match)

        return highlighted
