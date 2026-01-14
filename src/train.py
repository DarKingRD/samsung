"""
Training module for a text correction model using T5-based architectures.

This module implements a full training pipeline for a sequence-to-sequence model
designed to correct text errors (e.g., spelling, grammar, style) in Russian.
It uses Hugging Face Transformers (e.g., `cointegrated/rut5-base`) and PyTorch.

Key features:
    - TextCorrectionDataset: loads and tokenizes input-output text pairs from CSV.
    - Train/validation split with configurable fraction.
    - Training loop with AdamW optimizer and gradient clipping.
    - Model and tokenizer saving with configuration.

Usage:
    Run from command line:
    ```bash
    python train.py --model_name "cointegrated/rut5-small" \\
                    --data_path "data/train.csv" \\
                    --output_dir "models/my_model" \\
                    --batch_size 16 --num_epochs 5 --lr 3e-5
    ```

Dependencies:
    - torch
    - transformers
    - pandas
    - argparse
    - json
    - pathlib

CSV format example:
    input_text,output_text
    "привет как длеа?", "привет как дела?"
    "этот текст с ошбками", "этот текст с ошибками"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextCorrectionDataset(Dataset):
    """
    A PyTorch Dataset for text correction tasks in a seq2seq setup.

    Loads a CSV file with input-output text pairs and tokenizes them
    using a Hugging Face tokenizer.

    The dataset filters out rows with missing or empty text fields.

    Args:
        csv_path (str): Path to the CSV file containing 'input_text' and 'output_text' columns.
        tokenizer (AutoTokenizer): Tokenizer from Hugging Face Transformers.
        max_length (int): Maximum sequence length for tokenization. Default is 128.

    Attributes:
        data (pd.DataFrame): Cleaned DataFrame with input-output pairs.
        max_length (int): Maximum token sequence length.
        tokenizer (AutoTokenizer): Tokenizer used for encoding.

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base")
        >>> dataset = TextCorrectionDataset("data/train.csv", tokenizer, max_length=128)
        >>> sample = dataset[0]
        >>> print(sample["input_ids"].shape)  # torch.Size([128])
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = tokenizer

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["input_text", "output_text"])
        df = df[(df["input_text"].str.len() > 0) & (df["output_text"].str.len() > 0)]
        self.data = df.reset_index(drop=True)
        print("Number of samples:", len(self.data))

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of text pairs.
        """
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """
        Returns a tokenized training example.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with:
                - 'input_ids': Encoded input sequence (source).
                - 'attention_mask': Attention mask for input.
                - 'labels': Encoded target sequence with padding masked (-100).
        """
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

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]

        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            model_name (str): Pretrained model name or path. Default: "cointegrated/rut5-base".
            data_path (str): Path to the training CSV file. Default: "data/processed/all_train_enhanced.csv".
            output_dir (str): Directory to save the trained model. Created if not exists. Default: "models/correction_model".
            max_length (int): Maximum sequence length. Default: 128.
            batch_size (int): Training batch size. Default: 8.
            lr (float): Learning rate for AdamW. Default: 5e-5.
            num_epochs (int): Number of training epochs. Default: 4.
            val_frac (float): Fraction of data to use for validation. Default: 0.1.
            seed (int): Random seed for reproducibility. Default: 42.
    """
    p = argparse.ArgumentParser(description="Train a text correction model based on T5.")
    p.add_argument("--model_name", type=str, default="cointegrated/rut5-base")
    p.add_argument(
        "--data_path", type=str, default="data/processed/all_train_enhanced.csv"
    )
    p.add_argument("--output_dir", type=str, default="models/correction_model")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """
    Main training function.

    Workflow:
        1. Parses command-line arguments.
        2. Sets random seed.
        3. Loads dataset and splits into train/validation.
        4. Initializes model and tokenizer.
        5. Runs training loop with validation.
        6. Saves model, tokenizer, and config.

    The model is saved in two subdirectories:
        - `model/`: trained weights.
        - `tokenizer/`: tokenizer files.
    A `config.json` file is also saved with key training parameters.

    Note:
        The model is trained in `train()` mode and evaluated in `eval()` mode.
        Gradient clipping (`max_norm=1.0`) is applied to avoid explosion.
    """
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    # ====== Load data ======
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    dataset = TextCorrectionDataset(
        str(data_path), tokenizer, max_length=args.max_length
    )

    val_size = int(len(dataset) * args.val_frac)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ====== Training ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
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

        # Validation
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

    # ====== Save model ======
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

    print("Model saved to", out_dir.resolve())


if __name__ == "__main__":
    main()
