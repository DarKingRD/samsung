import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path


class RuT5CorrectorModel:
    def __init__(self, model_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(model_dir)

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_length: int = 64,
        num_beams: int = 4,
        temperature: float = 0.7,
    ) -> list[str]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]
