from pathlib import Path
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Пути
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "all_train.csv"
MODEL_OUT = BASE_DIR / "models" / "rut5-corrector"

MODEL_NAME = "ai-forever/ruT5-base"

print("Загружаем данные:", DATA_PATH)

# ---------------------------
# Загрузка датасета
# ---------------------------

df = pd.read_csv(DATA_PATH)

# минимальная валидация
df = df.dropna(subset=["input", "target"])
df = df[df["input"].str.len() > 1]
df = df[df["target"].str.len() > 1]

dataset = Dataset.from_pandas(df[["input", "target"]])

print("Размер датасета:", len(dataset))

# ---------------------------
# Модель и токенизатор
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ---------------------------
# Препроцессинг
# ---------------------------

def preprocess(batch):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["input", "target"],
)

# ---------------------------
# Аргументы обучения
# ---------------------------

args = TrainingArguments(
    output_dir=str(MODEL_OUT),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=200,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    optim="adamw_torch",
)

# ---------------------------
# Trainer
# ---------------------------

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
)

# ---------------------------
# Обучение
# ---------------------------

trainer.train()

# ---------------------------
# Сохранение
# ---------------------------

trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

print("✅ Модель сохранена в:", MODEL_OUT)
