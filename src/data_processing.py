"""
Data processing pipeline for training a Russian text error correction model (ruT5).

This script prepares a high-quality dataset for sequence-to-sequence training by:
    - Loading real error correction data from public sources:
        - Kartaslov dataset: spelling/typo corrections.
        - LORuGEC: grammar, punctuation, semantics, and some spelling.
    - Generating realistic synthetic errors (spelling, punctuation, grammar, semantics).
    - Balancing the dataset to ensure equal representation per error type.
    - Saving structured CSV files for model training.

Output files:
    - data/processed/all_train.csv:
        Full dataset with source, target, error_category, and type (original/synthetic).
    - data/processed/all_train_enhanced.csv:
        Simplified format for training: input_text, output_text, error_type.

Usage:
    $ python data_processing.py

Dependencies:
    - pandas
    - python-dotenv (optional)
    - openpyxl (for .xlsx)

Note:
    Place raw datasets in `data/raw/kartaslov/` and `data/raw/loru/`.
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
# Configuration
# ============================================================================

BASE_DIR = Path(".")
"""Project root directory."""

RAW_DIR = BASE_DIR / "data" / "raw"
"""Directory for raw input datasets."""

PROCESSED_DIR = BASE_DIR / "data" / "processed"
"""Directory for output processed files. Created if not exists."""

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ERROR_TYPES = ["spelling", "punctuation", "grammar", "semantics"]
"""List of supported error types."""

TARGET_PER_TYPE = 30000
"""Target number of examples per error type (used for synthetic balancing)."""


# ============================================================================
# Text validation utilities
# ============================================================================


def is_russian_text(text: str) -> bool:
    """
    Checks if text contains Cyrillic characters.

    Args:
        text (str): Input text.

    Returns:
        bool: True if Russian (Cyrillic) characters are present.
    """
    return bool(re.search(r"[а-яёА-ЯЁ]", str(text)))


def looks_like_code(text: str) -> bool:
    """
    Heuristically detects if text resembles source code.

    Args:
        text (str): Input string.

    Returns:
        bool: True if likely code (e.g., contains {}, =>, class, etc.).
    """
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
    """
    Detects if text is in Markdown, HTML, or similar format.

    Args:
        text (str): Input string.

    Returns:
        bool: True if likely markup (e.g., headers, lists, links).
    """
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
    """
    Determines if a string is a valid candidate for training data.

    Filters:
        - Short texts (<3 chars)
        - Non-Russian text
        - Code/markup
        - URLs

    Args:
        text (str): Text to validate.

    Returns:
        bool: True if valid and safe for training.
    """
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
    """
    Heuristic: checks if text looks like a full sentence.

    Args:
        text (str): Input text.

    Returns:
        bool: True if likely a sentence (length > 25, at least 4 words).
    """
    text = str(text).strip()
    if len(text) < 25:
        return False
    if len(text.split()) < 4:
        return False
    return True


def is_punct_candidate(text: str) -> bool:
    """
    Checks if text is a good candidate for punctuation error generation.

    Args:
        text (str): Input text.

    Returns:
        bool: True if text contains commas or coordinating conjunctions.
    """
    text = str(text)
    if not is_sentence_like(text):
        return False
    if "," in text:
        return True
    return any(
        w in text.lower() for w in [" но ", " а ", " и ", " что ", " если ", " когда "]
    )


# ============================================================================
# Error type classification (simple heuristic)
# ============================================================================


def classify_error_type(original: str, corrected: str) -> str:
    """
    Classifies the type of correction based on string differences.

    Args:
        original (str): Original (erroneous) text.
        corrected (str): Corrected version.

    Returns:
        str: One of 'punctuation', 'spelling', 'grammar', 'semantics', or 'unknown'.
    """
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
# Realistic synthetic error generation
# ============================================================================


class RealisticErrorGenerator:
    """
    Generates realistic synthetic errors of four types:
        - spelling
        - punctuation
        - grammar
        - semantics

    Designed to mimic common Russian language errors.
    """

    SPELLING_PAIRS = [
        ("чтобы", "что бы"), ("также", "так же"), ("тоже", "то же"),
        ("итак", "и так"), ("в течение", "в течении"), ("вследствие", "в следствии"),
        ("ввиду", "в виду"), ("вроде", "в роде"), ("насчет", "на счет"),
        ("из-за", "из за"), ("из-под", "из под"), ("по-моему", "по моему"),
        ("все-таки", "все таки"), ("кое-что", "кое что"), ("что-то", "что то"),
        ("какой-то", "какой то"), ("по-русски", "по русски"), ("по-прежнему", "по прежнему"),
        ("во-первых", "во первых"),
    ]
    """Common spelling/typo confusion pairs in Russian."""

    PUNCTUATION_PAIRS = [
        (" , и ", " и "), (" , но ", " но "), (" , а ", " а "),
        (" , что ", " что "), (" , который ", " который "), (" , однако ", " однако "),
        (" , поэтому ", " поэтому "), (" , если ", " если "), (" , потому что ", " потому что "),
        (" — ", " - "), (" - ", " — "), ("...", "…"), ("…", "..."),
    ]
    """Common punctuation errors: extra/missing commas, dash styles."""

    GRAMMAR_PAIRS = [
        ("в городе", "в городу"), ("к другу", "к друг"), ("без воды", "без вод"),
        ("о книге", "о книга"), ("с другом", "с другу"), ("для мамы", "для маму"),
        ("у сестры", "у сестру"), ("по дороге", "по дорогу"), ("красивая девушка", "красивый девушка"),
        ("интересная книга", "интересный книга"), ("новые идеи", "новая идеи"),
        ("я делаю", "я делал"), ("он пишет", "он писал"), ("мы читаем", "мы читали"),
        ("они говорят", "они говорили"),
    ]
    """Common grammar errors: case, gender, tense mistakes."""

    SEMANTICS_PAIRS = [
        ("в России", "по России"), ("из Москвы", "в Москве"), ("благодаря", "из-за"),
        ("из-за", "благодаря"), ("несмотря на", "смотря на"), ("по сравнению с", "в сравнении с"),
        ("в соответствии с", "соответственно с"), ("таким образом", "так образом"),
        ("по мнению", "на мнению"), ("в результате", "в результат"), ("в процессе", "в процесе"),
    ]
    """Common semantic errors: incorrect prepositions, connectors."""

    @staticmethod
    def _replace_preserving_case(text: str, old: str, new: str) -> str:
        """
        Replaces a substring with another, preserving the original capitalization.

        Args:
            text (str): Input text.
            old (str): Substring to find.
            new (str): Substring to insert.

        Returns:
            str: Text with replacement, case-preserved.
        """
        pattern = re.compile(re.escape(old), re.IGNORECASE)

        def repl(m: re.Match) -> str:
            s = m.group(0)
            if s and s[0].isupper():
                return new[0].upper() + new[1:] if len(new) > 1 else new.upper()
            return new

        return pattern.sub(repl, text, count=1)

    @classmethod
    def spelling_error(cls, text: str) -> str:
        """
        Applies a spelling or typo-like error.

        Uses predefined confusion pairs or introduces a random character typo.

        Args:
            text (str): Correct input text.

        Returns:
            str: Text with one spelling error, or unchanged.
        """
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
        w2 = w[:pos] + random.choice(alphabet) + w[pos + 1:]
        if w2 != w:
            words[idx] = w2
            return " ".join(words)

        return text

    @classmethod
    def punctuation_error(cls, text: str) -> str:
        """
        Introduces a punctuation error (extra/missing comma, wrong dash, etc.).

        Args:
            text (str): Input text.

        Returns:
            str: Text with a punctuation change, or unchanged.
        """
        text = str(text)
        if not is_punct_candidate(text):
            return text

        for correct, incorrect in cls.PUNCTUATION_PAIRS:
            if correct in text:
                return text.replace(correct, incorrect, 1)

        conjunctions = [
            "и", "но", "а", "что", "который", "которая", "которые",
            "когда", "если", "потому что", "так как", "чтобы",
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
        """
        Applies a grammar error (case, agreement, tense).

        Args:
            text (str): Input text.

        Returns:
            str: Text with a grammar mistake, or unchanged.
        """
        for correct, incorrect in cls.GRAMMAR_PAIRS:
            if correct.lower() in text.lower():
                out = cls._replace_preserving_case(text, correct, incorrect)
                if out != text:
                    return out
        return text

    @classmethod
    def semantics_error(cls, text: str) -> str:
        """
        Applies a semantic error (wrong preposition, connector).

        Args:
            text (str): Input text.

        Returns:
            str: Text with a semantic mistake, or unchanged.
        """
        for correct, incorrect in cls.SEMANTICS_PAIRS:
            if correct.lower() in text.lower():
                out = cls._replace_preserving_case(text, correct, incorrect)
                if out != text:
                    return out
        return text


def generate_synthetic_examples(
    correct_texts: List[str], target_count: int, error_type: str
) -> pd.DataFrame:
    """
    Generates synthetic error-correction pairs.

    Args:
        correct_texts (List[str]): Pool of correct sentences to corrupt.
        target_count (int): Target number of synthetic examples.
        error_type (str): One of 'spelling', 'punctuation', 'grammar', 'semantics'.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - source: erroneous text
            - target: correct text
            - error_category: error type
            - type: synthetic_{type}
    """
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
# Dataset loading
# ============================================================================


def process_kartaslov() -> pd.DataFrame:
    """
    Loads and processes the Kartaslov dataset (spelling/typo corrections).

    Searches for known file names in `data/raw/kartaslov/` or current directory.

    Returns:
        pd.DataFrame: Processed spelling corrections or empty DataFrame.
    """
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
    """
    Loads and processes the LORuGEC dataset (grammar, punctuation, semantics).

    Returns:
        pd.DataFrame: Processed corrections or empty DataFrame.
    """
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
# Final dataset assembly
# ============================================================================


def build_dataset() -> pd.DataFrame:
    """
    Builds the final balanced training dataset.

    Steps:
        1. Load Kartaslov (spelling) and LORuGEC (all types).
        2. Group correct texts by error type for synthetic generation.
        3. Generate synthetic errors to balance each type to TARGET_PER_TYPE.
        4. Merge, deduplicate, and save outputs.

    Returns:
        pd.DataFrame: Final dataset with all examples.
    """
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

    print("Step 1.5: prepare text pools for synthetic generation")

    spelling_pool = [
        t for t in original["target"].dropna().astype(str).tolist() if is_valid_text(t)
    ]
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
    """
    Entry point: processes data and builds the training dataset.
    """
    random.seed(42)

    print("Data processing started")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    build_dataset()
    print("Done")


if __name__ == "__main__":
    main()
