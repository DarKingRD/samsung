"""
Подготовка датасета для обучения модели исправления ошибок (ruT5).

Источники:
- Kartaslov: орфография (опечатки)
- LORuGEC: grammar / punctuation / semantics (+ иногда spelling)

Синтетика:
- Генерируем реалистичные ошибки по типам: spelling / punctuation / grammar / semantics

Результат:
- data/processed/all_train.csv (полный, с source/target/type/error_category)
- data/processed/all_train_enhanced.csv (для обучения: input_text/output_text/error_type)
"""

from __future__ import annotations

import random
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# Конфиг
# ============================================================================

BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ERROR_TYPES = ["spelling", "punctuation", "grammar", "semantics"]
TARGET_PER_TYPE = 30000


# ============================================================================
# Валидация текста
# ============================================================================


def is_russian_text(text: str) -> bool:
    return bool(re.search(r"[а-яёА-ЯЁ]", str(text)))


def looks_like_code(text: str) -> bool:
    if not isinstance(text, str):
        return False
    code_patterns = [
        r"<[^>]+>",
        r"\$\{[^}]+\}",
        r"function|const|let|var|=>",
        r"def |class |import ",
        r"[(){}\[\]]",
        r"^\s*//",
        r"^\s*\*",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in code_patterns)


def looks_like_markup(text: str) -> bool:
    if not isinstance(text, str):
        return False
    markup_patterns = [
        r"<[^>]+>",
        r"^\s*#+\s+",
        r"^\s*[-*]\s+",
        r"\[.+\]\(.+\)",
        r"```|~~~",
        r"``.+``",
    ]
    return any(
        re.search(p, text, re.MULTILINE | re.IGNORECASE) for p in markup_patterns
    )


def is_valid_text(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 3:
        return False
    if not is_russian_text(text):
        return False
    if looks_like_code(text) or looks_like_markup(text):
        return False
    if re.search(r"https?://|www\.|\.(com|ru|org|net)", text, re.IGNORECASE):
        return False
    return True


def is_sentence_like(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 25:
        return False
    if len(text.split()) < 4:
        return False
    return True


def is_punct_candidate(text: str) -> bool:
    text = str(text)
    if not is_sentence_like(text):
        return False
    if "," in text:
        return True
    return any(
        w in text.lower() for w in [" но ", " а ", " и ", " что ", " если ", " когда "]
    )


# ============================================================================
# Классификация типа ошибки (простая эвристика)
# ============================================================================


def classify_error_type(original: str, corrected: str) -> str:
    if not original or not corrected or original == corrected:
        return "unknown"

    original = str(original).strip()
    corrected = str(corrected).strip()

    orig_no_punct = re.sub(r"[^\w\s]", "", original).lower()
    corr_no_punct = re.sub(r"[^\w\s]", "", corrected).lower()

    if orig_no_punct == corr_no_punct and original != corrected:
        return "punctuation"

    spelling_patterns = [
        ("что бы", "чтобы"),
        ("так же", "также"),
        ("то же", "тоже"),
        ("в течении", "в течение"),
        ("в следствии", "вследствие"),
        ("в виду", "ввиду"),
        ("на счет", "насчет"),
    ]

    for wrong, right in spelling_patterns:
        if wrong in original.lower() and right in corrected.lower():
            return "spelling"

    if (
        abs(len(original) - len(corrected)) <= 3
        and len(original) >= 2
        and len(original) == len(corrected)
    ):
        diff = sum(1 for a, b in zip(original.lower(), corrected.lower()) if a != b)
        if diff <= 2:
            return "spelling"

    orig_words = orig_no_punct.split()
    corr_words = corr_no_punct.split()

    if len(orig_words) == len(corr_words):
        similar = 0
        for ow, cw in zip(orig_words, corr_words):
            if ow == cw or (len(ow) > 3 and len(cw) > 3 and ow[:3] == cw[:3]):
                similar += 1
        if similar >= max(1, int(len(orig_words) * 0.7)):
            return "grammar"

    if abs(len(orig_words) - len(corr_words)) <= 2:
        return "grammar"

    return "semantics"


# ============================================================================
# Реалистичная синтетика (spelling/punctuation/grammar/semantics)
# ============================================================================


class RealisticErrorGenerator:
    SPELLING_PAIRS = [
        ("чтобы", "что бы"),
        ("также", "так же"),
        ("тоже", "то же"),
        ("итак", "и так"),
        ("в течение", "в течении"),
        ("вследствие", "в следствии"),
        ("ввиду", "в виду"),
        ("вроде", "в роде"),
        ("насчет", "на счет"),
        ("из-за", "из за"),
        ("из-под", "из под"),
        ("по-моему", "по моему"),
        ("все-таки", "все таки"),
        ("кое-что", "кое что"),
        ("что-то", "что то"),
        ("какой-то", "какой то"),
        ("по-русски", "по русски"),
        ("по-прежнему", "по прежнему"),
        ("во-первых", "во первых"),
    ]

    PUNCTUATION_PAIRS = [
        (" , и ", " и "),
        (" , но ", " но "),
        (" , а ", " а "),
        (" , что ", " что "),
        (" , который ", " который "),
        (" , однако ", " однако "),
        (" , поэтому ", " поэтому "),
        (" , если ", " если "),
        (" , потому что ", " потому что "),
        (" — ", " - "),
        (" - ", " — "),
        ("...", "…"),
        ("…", "..."),
    ]

    GRAMMAR_PAIRS = [
        ("в городе", "в городу"),
        ("к другу", "к друг"),
        ("без воды", "без вод"),
        ("о книге", "о книга"),
        ("с другом", "с другу"),
        ("для мамы", "для маму"),
        ("у сестры", "у сестру"),
        ("по дороге", "по дорогу"),
        ("красивая девушка", "красивый девушка"),
        ("интересная книга", "интересный книга"),
        ("новые идеи", "новая идеи"),
        ("я делаю", "я делал"),
        ("он пишет", "он писал"),
        ("мы читаем", "мы читали"),
        ("они говорят", "они говорили"),
    ]

    SEMANTICS_PAIRS = [
        ("в России", "по России"),
        ("из Москвы", "в Москве"),
        ("благодаря", "из-за"),
        ("из-за", "благодаря"),
        ("несмотря на", "смотря на"),
        ("по сравнению с", "в сравнении с"),
        ("в соответствии с", "соответственно с"),
        ("таким образом", "так образом"),
        ("по мнению", "на мнению"),
        ("в результате", "в результат"),
        ("в процессе", "в процесе"),
    ]

    @staticmethod
    def _replace_preserving_case(text: str, old: str, new: str) -> str:
        pattern = re.compile(re.escape(old), re.IGNORECASE)

        def repl(m: re.Match) -> str:
            s = m.group(0)
            if s and s[0].isupper():
                return new[0].upper() + new[1:] if len(new) > 1 else new.upper()
            return new

        return pattern.sub(repl, text, count=1)

    @classmethod
    def spelling_error(cls, text: str) -> str:
        for correct, incorrect in cls.SPELLING_PAIRS:
            if correct.lower() in text.lower():
                out = cls._replace_preserving_case(text, correct, incorrect)
                if out != text:
                    return out

        words = text.split()
        if len(words) < 2:
            return text

        idx = random.randint(0, len(words) - 1)
        w = words[idx]
        if len(w) < 4 or not w.isalpha() or not w.islower():
            return text

        pos = random.randint(1, len(w) - 2)
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        w2 = w[:pos] + random.choice(alphabet) + w[pos + 1 :]
        if w2 != w:
            words[idx] = w2
            return " ".join(words)

        return text

    @classmethod
    def punctuation_error(cls, text: str) -> str:
        text = str(text)

        if not is_punct_candidate(text):
            return text

        for correct, incorrect in cls.PUNCTUATION_PAIRS:
            if correct in text:
                return text.replace(correct, incorrect, 1)

        conjunctions = [
            "и",
            "но",
            "а",
            "что",
            "который",
            "которая",
            "которые",
            "когда",
            "если",
            "потому что",
            "так как",
            "чтобы",
        ]

        for conj in conjunctions:
            with_comma = f", {conj}"
            without_comma = f" {conj}"

            if without_comma in text and with_comma not in text:
                if text.find(without_comma) > 0:
                    return text.replace(without_comma, with_comma, 1)

            if with_comma in text:
                return text.replace(with_comma, without_comma, 1)

        return text

    @classmethod
    def grammar_error(cls, text: str) -> str:
        for correct, incorrect in cls.GRAMMAR_PAIRS:
            if correct.lower() in text.lower():
                out = cls._replace_preserving_case(text, correct, incorrect)
                if out != text:
                    return out
        return text

    @classmethod
    def semantics_error(cls, text: str) -> str:
        for correct, incorrect in cls.SEMANTICS_PAIRS:
            if correct.lower() in text.lower():
                out = cls._replace_preserving_case(text, correct, incorrect)
                if out != text:
                    return out
        return text


def generate_synthetic_examples(
    correct_texts: List[str], target_count: int, error_type: str
) -> pd.DataFrame:
    generators: Dict[str, Callable[[str], str]] = {
        "spelling": RealisticErrorGenerator.spelling_error,
        "punctuation": RealisticErrorGenerator.punctuation_error,
        "grammar": RealisticErrorGenerator.grammar_error,
        "semantics": RealisticErrorGenerator.semantics_error,
    }

    gen = generators.get(error_type)
    if gen is None or target_count <= 0:
        return pd.DataFrame()

    pool = list(dict.fromkeys([t for t in correct_texts if is_valid_text(t)]))
    if not pool:
        return pd.DataFrame()

    if len(pool) > 5000:
        random.shuffle(pool)
        pool = pool[:5000]

    print(f"  synthetic/{error_type}: target={target_count:,}, pool={len(pool):,}")

    records = []
    seen = set()
    attempts = 0
    i = 0
    max_attempts = target_count * 12

    while len(records) < target_count and attempts < max_attempts:
        attempts += 1

        if i >= len(pool):
            i = 0
            random.shuffle(pool)

        tgt = pool[i]
        i += 1

        src = gen(tgt)
        if src == tgt:
            continue
        if not is_valid_text(src):
            continue

        detected = classify_error_type(src, tgt)
        if detected not in (error_type, "unknown"):
            continue

        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            {
                "source": src,
                "target": tgt,
                "error_category": error_type,
                "type": f"synthetic_{error_type}",
            }
        )

        if len(records) % 5000 == 0:
            print(
                f"    generated {len(records):,}/{target_count:,} (attempts={attempts:,})"
            )

    print(
        f"  synthetic/{error_type}: generated={len(records):,}, attempts={attempts:,}"
    )
    return pd.DataFrame.from_records(records)


# ============================================================================
# Загрузка датасетов
# ============================================================================


def process_kartaslov() -> pd.DataFrame:
    candidates = [
        RAW_DIR / "kartaslov" / "orfo_and_typos.L1_5.csv",
        RAW_DIR / "kartaslov" / "orfo_and_typos.L1_5+PHON.csv",
        Path("orfo_and_typos.L1_5.csv"),
        Path("orfo_and_typos.L1_5+PHON.csv"),
    ]

    dfs = []
    for fp in candidates:
        if not fp.exists():
            continue

        try:
            df = pd.read_csv(fp, sep=";", on_bad_lines="skip")
            if df.empty:
                continue

            cols = {c.upper(): c for c in df.columns}
            correct_col = cols.get("CORRECT")
            mistake_col = cols.get("MISTAKE")
            if not correct_col or not mistake_col:
                continue

            out = pd.DataFrame(
                {
                    "source": df[mistake_col].astype(str).str.strip(),
                    "target": df[correct_col].astype(str).str.strip(),
                    "error_category": "spelling",
                    "type": "kartaslov",
                }
            )

            out = out[
                (out["source"] != out["target"])
                & (out["source"].str.len() >= 2)
                & (out["target"].str.len() >= 2)
                & (out["source"].apply(is_valid_text))
                & (out["target"].apply(is_valid_text))
            ].reset_index(drop=True)

            print(f"  Kartaslov {fp.name}: {len(out):,}")
            dfs.append(out)

        except Exception as e:
            print(f"  Kartaslov {fp.name}: error: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["source", "target"]
    )


def process_lorugec() -> pd.DataFrame:
    candidates = [
        RAW_DIR / "loru" / "LORuGEC.xlsx",
        RAW_DIR / "loru" / "lorugec.xlsx",
        Path("LORuGEC.xlsx"),
        Path("lorugec.xlsx"),
    ]

    for fp in candidates:
        if not fp.exists():
            continue

        try:
            df = pd.read_excel(fp, sheet_name=0)
            if df.empty:
                continue

            cols = list(df.columns)
            cols_lower = [str(c).strip().lower() for c in cols]

            def find_col(keys: List[str]) -> Optional[str]:
                for key in keys:
                    for c, cl in zip(cols, cols_lower):
                        if key in cl:
                            return c
                return None

            initial_col = find_col(["initial", "source", "ошибоч", "исходн"])
            correct_col = find_col(["correct", "target", "исправ", "правильн"])
            section_col = find_col(["section", "раздел", "тип"])

            if not initial_col or not correct_col or not section_col:
                print(f"  LORuGEC {fp.name}: required columns not found")
                continue

            type_map = {
                "Spelling": "spelling",
                "Punctuation": "punctuation",
                "Grammar": "grammar",
                "Semantics": "semantics",
            }

            records = []
            for section_name, err in type_map.items():
                part = df[df[section_col] == section_name]
                if part.empty:
                    continue

                tmp = pd.DataFrame(
                    {
                        "source": part[initial_col].astype(str).str.strip(),
                        "target": part[correct_col].astype(str).str.strip(),
                        "error_category": err,
                        "type": "lorugec",
                    }
                )

                tmp = tmp[
                    (tmp["source"] != tmp["target"])
                    & (tmp["source"].str.len() >= 3)
                    & (tmp["target"].str.len() >= 3)
                    & (tmp["source"].apply(is_valid_text))
                    & (tmp["target"].apply(is_valid_text))
                ]

                records.append(tmp)

            if not records:
                return pd.DataFrame()

            out = pd.concat(records, ignore_index=True).drop_duplicates(
                subset=["source", "target"]
            )
            print(f"  LORuGEC {fp.name}: {len(out):,}")
            return out

        except Exception as e:
            print(f"  LORuGEC {fp.name}: error: {e}")

    return pd.DataFrame()


# ============================================================================
# Сборка финального датасета
# ============================================================================


def build_dataset() -> pd.DataFrame:
    print("Step 1: load original datasets")
    kart = process_kartaslov()
    loru = process_lorugec()

    parts = []
    if not kart.empty:
        kart.to_csv(PROCESSED_DIR / "kartaslov_spelling.csv", index=False)
        parts.append(kart)
    if not loru.empty:
        loru.to_csv(PROCESSED_DIR / "lorugec_all.csv", index=False)
        parts.append(loru)

    if not parts:
        raise RuntimeError("No source data found. Put datasets into data/raw/..")

    original = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["source", "target"])
        .reset_index(drop=True)
    )

    print(f"  original total: {len(original):,}")
    print("Original distribution:")
    for t in ERROR_TYPES:
        cnt = int((original["error_category"] == t).sum())
        print(f"  {t:12} {cnt:,}")

    # Пулы правильных текстов для синтетики по типам
    print("Step 1.5: prepare text pools for synthetic generation")

    # Для spelling можно брать из всего датасета (слова тоже ок)
    spelling_pool = [
        t for t in original["target"].dropna().astype(str).tolist() if is_valid_text(t)
    ]

    # Для punctuation/grammar/semantics только из LORuGEC и только предложения
    loru_targets = (
        original.loc[original["type"] == "lorugec", "target"]
        .dropna()
        .astype(str)
        .tolist()
    )

    punct_pool = [t for t in loru_targets if is_valid_text(t) and is_punct_candidate(t)]
    grammar_pool = [t for t in loru_targets if is_valid_text(t) and is_sentence_like(t)]
    sem_pool = [t for t in loru_targets if is_valid_text(t) and is_sentence_like(t)]

    print(
        f"  pools: spelling={len(set(spelling_pool)):,}, punct={len(set(punct_pool)):,}, "
        f"grammar={len(set(grammar_pool)):,}, sem={len(set(sem_pool)):,}"
    )

    print("Step 2: generate synthetic to reach TARGET_PER_TYPE")

    pool_by_type = {
        "spelling": spelling_pool,
        "punctuation": punct_pool,
        "grammar": grammar_pool,
        "semantics": sem_pool,
    }

    synth_parts = []
    for t in ERROR_TYPES:
        current = int((original["error_category"] == t).sum())
        need = max(0, TARGET_PER_TYPE - current)
        if need == 0:
            print(f"  {t}: already have {current:,}, skipping synthetic")
            continue

        print(f"  {t}: need {need:,} more (have {current:,})")
        synth_df = generate_synthetic_examples(pool_by_type[t], need, t)
        if not synth_df.empty:
            synth_parts.append(synth_df)

    synthetic = (
        pd.concat(synth_parts, ignore_index=True) if synth_parts else pd.DataFrame()
    )

    if not synthetic.empty:
        synthetic = synthetic.drop_duplicates(subset=["source", "target"]).reset_index(
            drop=True
        )
        synthetic.to_csv(PROCESSED_DIR / "synthetic_errors.csv", index=False)
        print(f"  synthetic total: {len(synthetic):,}")

    print("Step 3: merge + save")
    final_df = (
        pd.concat([original, synthetic], ignore_index=True)
        if not synthetic.empty
        else original.copy()
    )

    final_df = final_df.drop_duplicates(subset=["source", "target"]).reset_index(
        drop=True
    )

    final_df.to_csv(PROCESSED_DIR / "all_train.csv", index=False)

    train_df = final_df[["source", "target", "error_category"]].copy()
    train_df.columns = ["input_text", "output_text", "error_type"]
    train_df.to_csv(PROCESSED_DIR / "all_train_enhanced.csv", index=False)

    print(f"  all_train.csv: {len(final_df):,}")
    print(f"  all_train_enhanced.csv: {len(train_df):,}")

    print("Final distribution:")
    for t in ERROR_TYPES:
        cnt = int((final_df["error_category"] == t).sum())
        print(f"  {t:12} {cnt:,}")

    return final_df


def main():
    random.seed(42)

    print("Data processing started")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    build_dataset()
    print("Done")


if __name__ == "__main__":
    main()
