import re
from typing import List, Dict
from .model import RuT5CorrectorModel


class TypoInference:
    def __init__(self, model_path: str):
        self.model = RuT5CorrectorModel(model_path)

    def correct(self, text: str) -> str:
        task = self._detect_task(text)
        prompt = f"{task}: {text}"

        outputs = self.model.generate(prompt)

        best = self._select_best(text, outputs)
        return best

    def analyze(self, text: str) -> Dict:
        corrected = self.correct(text)

        if corrected == text:
            return {
                "text": text,
                "issues": [],
            }

        return {
            "text": corrected,
            "issues": self._diff(text, corrected),
        }

    # -------------------- helpers --------------------

    def _detect_task(self, text: str) -> str:
        if re.search(r"[,.!?]", text):
            return "grammar"
        return "spell"

    def _select_best(self, original: str, candidates: List[str]) -> str:
        # Пока простой выбор
        for c in candidates:
            if c != original:
                return c
        return original

    def _diff(self, original: str, corrected: str) -> List[Dict]:
        issues = []

        if original != corrected:
            issues.append({
                "start": 0,
                "end": len(original),
                "original": original,
                "suggestions": [corrected],
                "type": "auto",
                "confidence": 0.85,
            })

        return issues
