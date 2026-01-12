from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextCorrectionDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["input_text", "output_text"])
        df = df[(df["input_text"].str.len() > 0) & (df["output_text"].str.len() > 0)]
        self.data = df.reset_index(drop=True)
        print("Примеров:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src = str(row["input_text"])
        tgt = str(row["output_text"])

        enc = self.tokenizer(
            src,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Оставляю как у тебя (Kaggle-стиль)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]

        # маскируем паддинги в labels → -100
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="cointegrated/rut5-base")
    p.add_argument("--data_path", type=str, default="data/processed/all_train_enhanced.csv")
    p.add_argument("--output_dir", type=str, default="models/correction_model")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден датасет: {data_path.resolve()}")

    # ====== Данные ======
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    dataset = TextCorrectionDataset(str(data_path), tokenizer, max_length=args.max_length)

    val_size = int(len(dataset) * args.val_frac)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ====== Обучение ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print(f"\nЭпоха {epoch+1}/{args.num_epochs}")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        print(f"Train loss: {train_loss:.4f}")

        # валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        val_loss /= max(len(val_loader), 1)
        print(f"Val loss:   {val_loss:.4f}")

    # ====== Сохранение (как в Kaggle) ======
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Готово, модель сохранена в", out_dir.resolve())


if __name__ == "__main__":
    main()
