"""
Скрипт для скачивания и обработки датасетов с опечатками.
Фильтрует технические строки и сохраняет только чистый русский текст.
"""

import os
import json
import requests
import zipfile
import gzip
import subprocess
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Создаём директории
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки обработки
MAX_GITHUB_LINES = (
    None  # Максимальное количество строк из GitHub Typo Corpus (None = все)
)
MIN_TEXT_LENGTH = 3  # Минимальная длина текста для обработки
MAX_TEXT_LENGTH = 200  # Максимальная длина текста для чистого русского текста
ONLY_RUSSIAN = True  # Только русский язык
MAX_WORD_LENGTH = 30  # Максимальная длина слова
MIN_RUSSIAN_RATIO = 0.7  # Минимум 70% русских символов


def is_clean_russian_text(text: str) -> bool:
    """Строгая проверка на чистый русский текст без технических элементов."""
    if not text or len(text) < MIN_TEXT_LENGTH:
        return False

    text_lower = text.lower()

    # Игнорируем строки с техническими префиксами
    technical_prefixes = [
        "msgstr",
        "msgid",
        "msgctxt",
        '"""',
        "``",
        "''",
        "`",
        '"',
        "\\input{",
        "\\label{",
        "\\linespread{",
        "\\",
        "findoutmore:",
        "about_text:",
        "clipboarderror",
        "claim:",
        "section{",
        "ection{",
        "clearallfilters:",
        'empty":',
        'name":',
        'aggregatedname":',
        "cmdshowin_os",
        "invalid delay",
        "google reader",
        "tiny tiny rss",
        "newsblur",
        "the old reader",
        "@link",
        "@api",
        "ng.",
        "api/ng.",
        "{@link",
        "# ",
        "## ",
        "### ",
        "brackets.",
        "hbibtex-styles:",
        "bibtex-styles:",
        ":1234:",
        "|   |",
        "|",
    ]

    for prefix in technical_prefixes:
        if text_lower.startswith(prefix) or prefix in text_lower:
            return False

    # Игнорируем строки с фигурными скобками (JSON/LaTeX)
    if "{" in text or "}" in text:
        # Проверяем, не является ли это LaTeX командой
        if "\\" in text:  # LaTeX команды
            return False
        # Если много скобок = скорее всего JSON/техническое
        if text.count("{") > 1 or text.count("}") > 1:
            return False

    # Игнорируем строки с HTML/XML тегами
    if "<" in text and ">" in text:
        return False

    # Игнорируем строки с множеством кавычек или спецсимволов
    if text.count('"') > 3 or text.count("'") > 3:
        return False

    # Проверяем процент русских символов
    russian_chars = sum(1 for c in text if "а" <= c.lower() <= "я" or c in "ёЁ -.,!?;:")
    total_chars = len(text)

    if total_chars == 0:
        return False

    # Строгая проверка на русский язык
    if (russian_chars / total_chars) < MIN_RUSSIAN_RATIO:
        return False

    # Проверяем на URL/технические символы
    url_indicators = [
        "http://",
        "https://",
        "www.",
        ".com",
        ".org",
        ".ru",
        "%s",
        "%d",
        "%u",
        "%t",
        "%n",
        "%v",
        "%(",
        "${",
        "$",
        "&nbsp;",
        "&amp;",
        "&lt;",
        "&gt;",
    ]

    for indicator in url_indicators:
        if indicator in text_lower:
            return False

    # Игнорируем строки с кодом (много специальных символов)
    code_chars = sum(1 for c in text if c in "{}[]();=<>/\\|&%$#@!~`")
    if len(text) > 0 and (code_chars / len(text)) > 0.1:
        return False

    # Игнорируем строки только из знаков препинания/скобок
    clean_text = "".join(c for c in text if c.isalpha() or c.isspace())
    if len(clean_text.strip()) < 2:
        return False

    # Игнорируем строки, которые выглядят как пути/имена файлов
    if "/" in text and ("." in text or "\\" in text):
        return False

    return True


def clean_text_for_training(text: str) -> str:
    """Очищает текст для обучения модели."""
    if not text:
        return ""

    text = text.strip()

    # Убираем тройные кавычки (JSON)
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()

    # Убираем msgstr и подобные префиксы
    msg_prefixes = ["msgstr ", "msgid ", "msgctxt ", "findoutmore:", "about_text:"]
    for prefix in msg_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            # Убираем возможные кавычки после префикса
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            break

    # Убираем оставшиеся технические префиксы
    tech_prefixes = ["claim:", "clipboardError:", 'empty":', 'name":']
    for prefix in tech_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # Убираем одиночные кавычки по краям если они остались
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    return text


def is_russian_text(text: str) -> bool:
    """Проверяет, является ли текст русским (базовая проверка)."""
    if not text:
        return False

    # Подсчитываем русские символы
    russian_chars = sum(1 for c in text if "а" <= c.lower() <= "я" or c in "ёЁ")
    total_chars = sum(1 for c in text if c.isalpha())

    if total_chars == 0:
        return False

    # Если больше 50% русских символов - считаем русским
    return (russian_chars / total_chars) > 0.5


def is_technical_text(text: str) -> bool:
    """Проверяет, является ли текст техническим (код, API и т.д.)."""
    if not text:
        return False

    text_lower = text.lower()

    # Технические маркеры
    technical_markers = [
        "api",
        "json",
        "xml",
        "html",
        "css",
        "js",
        "http",
        "https",
        "url",
        "function",
        "method",
        "class",
        "import",
        "export",
        "return",
        "var",
        "let",
        "const",
        "config",
        "parameter",
        "option",
        "setting",
        "default",
        "react",
        "angular",
        "vue",
        "node",
        "npm",
        "package",
        "sql",
        "database",
        "query",
        "table",
        "column",
        "git",
        "commit",
        "branch",
        "merge",
        "pull",
        "push",
        "//",
        "/*",
        "*/",
        "<?",
        "?>",
        "<script",
        "</script>",
        "<div",
        "</div>",
        "=",
        ";",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "->",
        "=>",
        "::",
        "&nbsp;",
        "&amp;",
        "&lt;",
        "&gt;",
    ]

    # Проверяем наличие технических маркеров
    for marker in technical_markers:
        if marker in text_lower:
            return True

    # Проверяем наличие слишком много специальных символов (код)
    special_chars = sum(1 for c in text if c in "{}[]();=<>/\\|&%$#@!~`")
    if len(text) > 0 and (special_chars / len(text)) > 0.1:
        return True

    return False


def clean_markdown(text: str) -> str:
    """Убирает markdown/разметку и лишние пробелы."""
    if not text:
        return ""

    t = text
    # Ссылки [text](url) и изображения ![alt](url)
    t = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", t)
    t = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", t)
    # Кодовые блоки и инлайны
    t = re.sub(r"`{3}.*?`{3}", " ", t, flags=re.S)
    t = re.sub(r"`[^`]+`", " ", t)
    # Жирный/курсив
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"[_*]{1,2}([^_*]+)[_*]{1,2}", r"\1", t)
    # Заголовки/цитаты/списки
    t = re.sub(r"^[>#\-\+\*]+\s*", "", t, flags=re.M)
    # HTML-теги
    t = re.sub(r"<[^>]+>", " ", t)
    # Удаляем лишние пробелы/переводы строк
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_valid_typo_pair(original: str, corrected: str) -> bool:
    """Проверяет, является ли пара валидной опечаткой."""
    if not original or not corrected:
        return False

    original_words = original.split()
    corrected_words = corrected.split()

    # Для предложений (2+ слов) допускаем различия — оставляем для модели
    if len(original_words) > 1 or len(corrected_words) > 1:
        # Простейшая отсечка по длине, чтобы убрать мусор
        length_diff = abs(len(original) - len(corrected))
        return length_diff <= len(original) * 0.8

    # Для одиночных слов — строгая проверка сходства
    if len(original_words) == 1 and len(corrected_words) == 1:
        orig_word = original_words[0].lower()
        corr_word = corrected_words[0].lower()

        length_diff = abs(len(orig_word) - len(corr_word))
        if length_diff > len(orig_word) * 0.5:
            return False

        # Если слова слишком разные - не опечатка
        if orig_word != corr_word:
            # Простая проверка на схожесть
            common_chars = sum(1 for a, b in zip(orig_word, corr_word) if a == b)
            min_len = min(len(orig_word), len(corr_word))
            if min_len > 0 and (common_chars / min_len) < 0.5:
                return False

    return True


def download_file(url: str, filepath: Path):
    """Скачивает файл по URL."""
    if filepath.exists():
        print(f"Файл {filepath} уже существует, пропускаем скачивание")
        return

    print(f"Скачиваем {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Скачано: {filepath}")


def download_kartaslov_safe():
    """Безопасное скачивание Kartaslov с обходом SSL проблем."""
    print("\n=== Обработка Kartaslov ===")

    kartaslov_dir = RAW_DIR / "kartaslov"

    # Если уже скачано
    if kartaslov_dir.exists() and any(kartaslov_dir.iterdir()):
        print(f"Kartaslov уже скачан в {kartaslov_dir}")
        return kartaslov_dir

    print("Пробуем скачать Kartaslov...")

    # Способ 1: Клонирование через git (самый надежный)
    print("Способ 1: Клонируем репозиторий через git...")
    try:
        # Удаляем старый каталог если есть
        if kartaslov_dir.exists():
            import shutil

            shutil.rmtree(kartaslov_dir)

        result = subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/dkulagin/kartaslov.git",
                str(kartaslov_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print(f"Успешно клонирован в {kartaslov_dir}")
            return kartaslov_dir
        else:
            print(f"Git clone не удался: {result.stderr}")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        print(f"Ошибка git clone: {e}")

    # Способ 2: Скачивание отдельных файлов через raw.githubusercontent.com
    print("Способ 2: Скачиваем отдельные файлы...")
    try:
        kartaslov_dir.mkdir(parents=True, exist_ok=True)

        # Скачиваем основной CSV файл с опечатками
        csv_url = "https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5.csv"
        csv_path = kartaslov_dir / "orfo_and_typos.L1_5.csv"

        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()

        with open(csv_path, "wb") as f:
            f.write(response.content)

        print(f"Скачан CSV файл: {csv_path}")

        # Также скачиваем второй файл если нужен
        csv_url2 = "https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5+PHON.csv"
        csv_path2 = kartaslov_dir / "orfo_and_typos.L1_5+PHON.csv"

        response2 = requests.get(csv_url2, timeout=30)
        if response2.status_code == 200:
            with open(csv_path2, "wb") as f:
                f.write(response2.content)
            print(f"Скачан второй CSV файл: {csv_path2}")

        return kartaslov_dir

    except Exception as e:
        print(f"Ошибка при скачивании отдельных файлов: {e}")

    # Способ 3: Альтернативный URL для скачивания ZIP
    print("Способ 3: Пробуем альтернативный URL...")
    try:
        alt_url = "https://github.com/dkulagin/kartaslov/archive/master.zip"
        zip_path = RAW_DIR / "kartaslov_master.zip"

        response = requests.get(alt_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Скачан архив: {zip_path}")

        # Распаковываем
        extract_dir = RAW_DIR / "kartaslov-master"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)

        # Переименовываем если нужно
        if extract_dir.exists():
            shutil.move(extract_dir, kartaslov_dir)

        print(f"Распаковано в: {kartaslov_dir}")
        return kartaslov_dir

    except Exception as e:
        print(f"Ошибка при альтернативном скачивании: {e}")

    print("Не удалось скачать Kartaslov автоматически")
    return None


def process_github_typo_corpus():
    """Обрабатывает GitHub Typo Corpus с СТРОГОЙ фильтрацией."""
    print("\n=== Обработка GitHub Typo Corpus (строгая фильтрация) ===")

    # Скачиваем сам датасет (JSONL файл)
    dataset_url = "https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz"
    dataset_path = RAW_DIR / "github-typo-corpus.v1.0.0.jsonl.gz"
    jsonl_path = RAW_DIR / "github-typo-corpus.v1.0.0.jsonl"

    # Скачиваем сжатый файл
    if not dataset_path.exists():
        download_file(dataset_url, dataset_path)

    # Распаковываем если нужно
    if dataset_path.exists() and not jsonl_path.exists():
        print("Распаковываем gzip архив...")
        try:
            with gzip.open(dataset_path, "rb") as f_in:
                file_size = os.path.getsize(dataset_path)
                chunk_size = 1024 * 1024  # 1MB
                with open(jsonl_path, "wb") as f_out:
                    with tqdm(
                        total=file_size, unit="B", unit_scale=True, desc="Распаковка"
                    ) as pbar:
                        while True:
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            pbar.update(len(chunk))
            print(f"Распаковано: {jsonl_path}")
        except Exception as e:
            print(f"Ошибка при распаковке: {e}")
            return []

    pairs = []
    processed_commits = 0
    processed_edits = 0
    skipped_technical = 0
    skipped_language = 0
    skipped_length = 0

    # Обрабатываем JSONL файл
    if jsonl_path.exists():
        file_size = os.path.getsize(jsonl_path)
        print(
            f"Обрабатываем {jsonl_path} (размер: {file_size / 1024 / 1024:.2f} MB)..."
        )

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                # Читаем построчно (JSONL формат)
                for line_num, line in enumerate(tqdm(f, desc="Обработка строк"), 1):
                    try:
                        commit = json.loads(line.strip())
                        if "edits" not in commit:
                            continue

                        processed_commits += 1

                        # Обрабатываем каждую правку
                        for edit in commit.get("edits", []):
                            src_text = edit.get("src", {}).get("text", "").strip()
                            tgt_text = edit.get("tgt", {}).get("text", "").strip()
                            lang = edit.get("src", {}).get("lang", "")

                            # Чистим markdown/разметку
                            src_text = clean_markdown(src_text)
                            tgt_text = clean_markdown(tgt_text)

                            # Пропускаем пустые
                            if not src_text or not tgt_text or src_text == tgt_text:
                                continue

                            # ФИЛЬТРАЦИЯ ШАГ 1: Только русский язык
                            if ONLY_RUSSIAN:
                                if not (
                                    lang == "rus"
                                    or is_russian_text(src_text)
                                    or is_russian_text(tgt_text)
                                ):
                                    skipped_language += 1
                                    continue

                            # ФИЛЬТРАЦИЯ ШАГ 2: Только чистый русский текст
                            if not (
                                is_clean_russian_text(src_text)
                                and is_clean_russian_text(tgt_text)
                            ):
                                skipped_technical += 1
                                continue

                            # ФИЛЬТРАЦИЯ ШАГ 3: Длина текста
                            if (
                                len(src_text) < MIN_TEXT_LENGTH
                                or len(tgt_text) < MIN_TEXT_LENGTH
                            ):
                                skipped_length += 1
                                continue

                            # ФИЛЬТРАЦИЯ ШАГ 4: Максимальная длина
                            if (
                                len(src_text) > MAX_TEXT_LENGTH
                                or len(tgt_text) > MAX_TEXT_LENGTH
                            ):
                                skipped_length += 1
                                continue

                            # ФИЛЬТРАЦИЯ ШАГ 5: Очищаем текст
                            src_clean = clean_text_for_training(src_text)
                            tgt_clean = clean_text_for_training(tgt_text)

                            if not src_clean or not tgt_clean or src_clean == tgt_clean:
                                continue

                            # ФИЛЬТРАЦИЯ ШАГ 6: Проверяем валидность пары
                            src_words = src_clean.split()
                            tgt_words = tgt_clean.split()

                            if len(src_words) == 1 and len(tgt_words) == 1:
                                # Для одиночных слов проверяем сходство
                                if not is_valid_typo_pair(src_clean, tgt_clean):
                                    continue

                            # ФИЛЬТРАЦИЯ ШАГ 7: Убираем строки с числами/спецсимволами
                            if any(c.isdigit() for c in src_clean) or any(
                                c.isdigit() for c in tgt_clean
                            ):
                                continue

                            # УСПЕХ: добавляем пару
                            pairs.append((src_clean, tgt_clean))
                            processed_edits += 1

                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue

            print(f"\nСтатистика обработки GitHub Typo Corpus:")
            print(f"  Обработано коммитов: {processed_commits}")
            print(f"  Обработано правок: {processed_edits}")
            print(f"  Пропущено (не русский): {skipped_language}")
            print(f"  Пропущено (технические строки): {skipped_technical}")
            print(f"  Пропущено (длина): {skipped_length}")
            print(f"  Сохранено чистых русских пар: {len(pairs)}")

            # Показываем примеры сохранённых пар
            if pairs:
                print(f"\nПримеры сохранённых пар (первые 5):")
                for i, (src, tgt) in enumerate(pairs[:5]):
                    print(f"  {i+1}. '{src}' → '{tgt}'")

        except Exception as e:
            print(f"Ошибка при обработке файла: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Файл датасета не найден. Пропускаем GitHub Typo Corpus.")

    return pairs


def process_kartaslov():
    """Обрабатывает датасет Kartaslov."""
    kartaslov_dir = download_kartaslov_safe()

    if not kartaslov_dir or not kartaslov_dir.exists():
        print("Не удалось получить данные Kartaslov")
        return []

    pairs = []

    # Ищем CSV файлы в директории
    csv_files = []

    # Проверяем возможные пути
    possible_paths = [
        kartaslov_dir / "dataset" / "orfo_and_typos",
        kartaslov_dir / "orfo_and_typos",
        kartaslov_dir,  # Если файлы в корне
    ]

    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = path
            break

    if not data_dir:
        print(f"Не найдена директория с данными Kartaslov в {kartaslov_dir}")
        # Ищем любые CSV файлы в kartaslov_dir
        for csv_file in kartaslov_dir.rglob("*.csv"):
            csv_files.append(csv_file)
    else:
        # Ищем CSV файлы в найденной директории
        csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"CSV файлы не найдены в {kartaslov_dir}")
        return []

    print(f"Найдено {len(csv_files)} CSV файлов для обработки")

    for csv_file in csv_files:
        print(f"Обрабатываем {csv_file.name}...")
        try:
            # Пробуем разные разделители
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(
                        csv_file, sep=sep, encoding="utf-8", on_bad_lines="skip"
                    )
                    if len(df.columns) >= 2:
                        break
                except:
                    continue

            # Пробуем без указания разделителя
            if len(df.columns) < 2:
                df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")

            print(
                f"  Загружено {len(df)} строк, {len(df.columns)} столбцов: {list(df.columns)}"
            )

            # Пытаемся найти столбцы с ошибками и исправлениями
            mistake_col = None
            correct_col = None

            # Ищем по названиям столбцов
            for col in df.columns:
                col_lower = str(col).lower()
                if any(
                    word in col_lower
                    for word in ["mistake", "error", "wrong", "неправильно", "ошибка"]
                ):
                    mistake_col = col
                if any(
                    word in col_lower
                    for word in ["correct", "исправление", "правильно", "правка"]
                ):
                    correct_col = col
                if any(word in col_lower for word in ["typo", "опечатка"]):
                    if not mistake_col:
                        mistake_col = col
                    if not correct_col:
                        correct_col = col

            # Если не нашли по названиям, берем первые два столбца
            if not mistake_col or not correct_col:
                if len(df.columns) >= 2:
                    mistake_col = df.columns[0]
                    correct_col = df.columns[1]
                    print(
                        f"  Используем столбцы по умолчанию: {mistake_col} -> {correct_col}"
                    )
                else:
                    print(f"  Недостаточно столбцов в файле, пропускаем")
                    continue

            file_pairs = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {csv_file.name}"):
                try:
                    mistake = (
                        str(row[mistake_col]).strip()
                        if pd.notna(row[mistake_col])
                        else ""
                    )
                    correct = (
                        str(row[correct_col]).strip()
                        if pd.notna(row[correct_col])
                        else ""
                    )

                    # Пропускаем пустые
                    if not mistake or not correct or mistake == correct:
                        continue

                    # Фильтруем только русские слова
                    if ONLY_RUSSIAN:
                        if not is_russian_text(mistake) or not is_russian_text(correct):
                            continue

                    # Фильтруем слишком длинные слова
                    if len(mistake) > MAX_WORD_LENGTH or len(correct) > MAX_WORD_LENGTH:
                        continue

                    # Проверяем валидность пары
                    if not is_valid_typo_pair(mistake, correct):
                        continue

                    # Добавляем пару (ошибка -> исправление)
                    pairs.append((mistake, correct))
                    file_pairs += 1

                except Exception as e:
                    continue  # Пропускаем проблемные строки

            print(f"  Получено {file_pairs} пар из {csv_file.name}")

        except Exception as e:
            print(f"Ошибка при обработке {csv_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"Всего получено {len(pairs)} пар из Kartaslov")

    # Показываем примеры
    if pairs:
        print(f"\nПримеры пар из Kartaslov (первые 5):")
        for i, (wrong, correct) in enumerate(pairs[:5]):
            print(f"  {i+1}. '{wrong}' → '{correct}'")

    return pairs


def process_loru_gec():
    """
    Обрабатывает датасет LoRuGEC (грамматика/стилистика/семантика).
    """
    xlsx_url = "https://github.com/ReginaNasyrova/LORuGEC/raw/main/LORuGEC.xlsx"
    xlsx_path = RAW_DIR / "LORuGEC.xlsx"
    pairs: List[Tuple[str, str]] = []

    # Скачиваем Excel, если его ещё нет
    if not xlsx_path.exists():
        try:
            print("\n=== Скачивание LoRuGEC ===")
            download_file(xlsx_url, xlsx_path)
        except Exception as e:
            print(f"Не удалось скачать LoRuGEC: {e}")
            return pairs

    print("\n=== Обработка LoRuGEC (LORuGEC.xlsx) ===")
    try:
        df = pd.read_excel(xlsx_path)
        print(f"Найдено столбцов в LORuGEC.xlsx: {list(df.columns)}")

        # Эвристика: ищем два текстовых столбца — с ошибочным и правильным предложением.
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) < 2:
            print("Не удалось найти два текстовых столбца в LORuGEC.xlsx, пропускаем.")
            return pairs

        # Попробуем угадать имена: один столбец с ошибками ("Initial sentence"/ungrammatical),
        # другой с исправлениями ("Correct sentence"/correct).
        def score_src(name: str) -> int:
            n = name.lower()
            score = 0
            if "initial" in n:
                score += 2
            if any(k in n for k in ["ungram", "error", "wrong", "src", "source"]):
                score += 1
            return score

        def score_tgt(name: str) -> int:
            n = name.lower()
            score = 0
            if "correct" in n:
                score += 2
            if any(k in n for k in ["gram", "corr", "tgt", "target"]):
                score += 1
            return score

        # По умолчанию берём первые два текстовых столбца
        src_col, tgt_col = text_cols[0], text_cols[1]

        # Пытаемся подобрать по эвристикам
        best_src = max(text_cols, key=score_src)
        best_tgt = max(text_cols, key=score_tgt)
        if best_src != best_tgt and score_tgt(best_tgt) > 0:
            # Если нашли хороший столбец с исправлениями — фиксируем его
            tgt_col = best_tgt
            # Для исходного предложения стараемся выбрать "Initial sentence"
            if score_src(best_src) > 0:
                src_col = best_src
            else:
                for c in text_cols:
                    if "initial" in c.lower():
                        src_col = c
                        break

        print(f"Используем столбцы LoRuGEC: src='{src_col}', tgt='{tgt_col}'")

        clean_pairs = 0
        skipped_pairs = 0

        for _, row in df.iterrows():
            src_text = str(row.get(src_col, "")).strip()
            tgt_text = str(row.get(tgt_col, "")).strip()

            src_text = clean_markdown(src_text)
            tgt_text = clean_markdown(tgt_text)

            if not src_text or not tgt_text or src_text == tgt_text:
                continue

            # Фильтруем технические строки и проверяем русский язык
            if is_clean_russian_text(src_text) and is_clean_russian_text(tgt_text):
                src_clean = clean_text_for_training(src_text)
                tgt_clean = clean_text_for_training(tgt_text)

                if src_clean and tgt_clean and src_clean != tgt_clean:
                    pairs.append((src_clean, tgt_clean))
                    clean_pairs += 1
                else:
                    skipped_pairs += 1
            else:
                skipped_pairs += 1

        print(f"Получено {clean_pairs} чистых пар из LoRuGEC")
        print(f"Пропущено {skipped_pairs} пар (технические/не русские)")

        # Показываем примеры
        if pairs:
            print(f"\nПримеры пар из LoRuGEC (первые 5):")
            for i, (wrong, correct) in enumerate(pairs[:5]):
                print(f"  {i+1}. '{wrong}' → '{correct}'")

    except Exception as e:
        print(f"Ошибка при обработке LORuGEC.xlsx: {e}")

    return pairs


def create_synthetic_pairs():
    """Создаёт синтетические пары для дополнения датасета."""
    print("\n=== Создание синтетических пар ===")

    synthetic_pairs = [
        # Орфографические ошибки (пропуски букв)
        ("програмирование", "программирование"),
        ("унивеситет", "университет"),
        ("компютер", "компьютер"),
        ("телевизор", "телевизор"),
        ("холодильник", "холодильник"),
        ("автомобиль", "автомобиль"),
        # Орфографические ошибки (замены букв)
        ("карова", "корова"),
        ("малако", "молоко"),
        ("харашо", "хорошо"),
        ("диревня", "деревня"),
        ("синий", "синий"),
        ("желтый", "жёлтый"),
        # Орфографические ошибки (перестановки)
        ("слонец", "солнце"),
        ("дгур", "друг"),
        ("кинга", "книга"),
        ("школа", "школа"),
        ("учитель", "учитель"),
        # Грамматические ошибки (пунктуация)
        ("привет как дела", "привет, как дела"),
        ("здравствуйте как вы", "здравствуйте, как вы"),
        ("добрый день чем могу помочь", "добрый день, чем могу помочь"),
        ("извините я опоздал", "извините, я опоздал"),
        ("спасибо за помощь", "спасибо за помощь"),
        # Грамматические ошибки (согласование)
        ("красивая мальчик", "красивый мальчик"),
        ("умный девочка", "умная девочка"),
        ("большой дом", "большой дом"),
        ("маленькая стол", "маленький стол"),
        ("интересный книга", "интересная книга"),
        # Смысловые ошибки
        ("пить суп", "есть суп"),
        ("есть воду", "пить воду"),
        ("носить шапку на ногах", "носить носки на ногах"),
        ("одеть часы", "надеть часы"),
        ("горячий лёд", "холодный лёд"),
    ]

    print(f"Создано {len(synthetic_pairs)} синтетических пар")

    # Показываем примеры
    print(f"\nПримеры синтетических пар (первые 5):")
    for i, (wrong, correct) in enumerate(synthetic_pairs[:5]):
        print(f"  {i+1}. '{wrong}' → '{correct}'")

    return synthetic_pairs


def save_processed_data(pairs: List[Tuple[str, str]], filename: str):
    """Сохраняет обработанные данные в CSV."""
    df = pd.DataFrame(pairs, columns=["original", "corrected"])
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Сохранено {len(pairs)} пар в {output_path}")

    # Показываем первые несколько строк
    print(f"Первые 3 строки файла {filename}:")
    print(df.head(3).to_string(index=False))
    print()

    return output_path


def filter_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Финальная фильтрация пар (общая логика для всех источников)."""
    # Удаляем дубликаты и пустые строки
    unique_pairs = list(
        {
            (orig.strip(), corr.strip())
            for orig, corr in pairs
            if str(orig).strip() and str(corr).strip()
        }
    )

    filtered_pairs: List[Tuple[str, str]] = []

    for orig, corr in unique_pairs:
        orig_words = orig.split()
        corr_words = corr.split()

        is_sentence = len(orig_words) > 1 or len(corr_words) > 1

        # Фильтруем по языку (если включено)
        if ONLY_RUSSIAN:
            if not (is_russian_text(orig) or is_russian_text(corr)):
                continue

        # Для предложений: проверяем на чистый русский текст
        if is_sentence:
            if not (is_clean_russian_text(orig) and is_clean_russian_text(corr)):
                continue
            filtered_pairs.append((orig, corr))
            continue

        # Для одиночных слов — строгие проверки
        if is_technical_text(orig) or is_technical_text(corr):
            continue
        if not is_valid_typo_pair(orig, corr):
            continue

        filtered_pairs.append((orig, corr))

    return filtered_pairs


def main():
    """Основная функция обработки данных."""
    print("=" * 60)
    print("Начинаем обработку датасетов с СТРОГОЙ фильтрацией")
    print("=" * 60)

    github_pairs: List[Tuple[str, str]] = []
    kartaslov_pairs: List[Tuple[str, str]] = []
    loru_pairs: List[Tuple[str, str]] = []
    synthetic_pairs: List[Tuple[str, str]] = []

    # Обрабатываем GitHub Typo Corpus (со строгой фильтрацией)
    print("\n" + "=" * 60)
    try:
        github_pairs = process_github_typo_corpus()
        print(f"\nИтог GitHub Typo Corpus: {len(github_pairs)} чистых пар")
    except Exception as e:
        print(f"Ошибка при обработке GitHub Typo Corpus: {e}")

    # Обрабатываем Kartaslov
    print("\n" + "=" * 60)
    try:
        kartaslov_pairs = process_kartaslov()
        print(f"\nИтог Kartaslov: {len(kartaslov_pairs)} чистых пар")
    except Exception as e:
        print(f"Ошибка при обработке Kartaslov: {e}")

    # Обрабатываем LoRuGEC
    print("\n" + "=" * 60)
    try:
        loru_pairs = process_loru_gec()
        if loru_pairs:
            print(f"\nИтог LoRuGEC: {len(loru_pairs)} чистых пар")
    except Exception as e:
        print(f"Ошибка при обработке LoRuGEC: {e}")

    # Создаём синтетические пары (если нужно больше данных)
    print("\n" + "=" * 60)
    try:
        synthetic_pairs = create_synthetic_pairs()
    except Exception as e:
        print(f"Ошибка при создании синтетических пар: {e}")

    # Финальная фильтрация по источникам
    print("\n" + "=" * 60)
    print("Финальная фильтрация и объединение пар...")

    filtered_github = filter_pairs(github_pairs) if github_pairs else []
    filtered_kartaslov = filter_pairs(kartaslov_pairs) if kartaslov_pairs else []
    filtered_loru = filter_pairs(loru_pairs) if loru_pairs else []
    filtered_synthetic = filter_pairs(synthetic_pairs) if synthetic_pairs else []

    print(f"Всего уникальных пар из GitHub после фильтрации: {len(filtered_github)}")
    print(
        f"Всего уникальных пар из Kartaslov после фильтрации: {len(filtered_kartaslov)}"
    )
    if filtered_loru:
        print(f"Всего уникальных пар из LoRuGEC после фильтрации: {len(filtered_loru)}")
    if filtered_synthetic:
        print(f"Всего синтетических пар после фильтрации: {len(filtered_synthetic)}")

    # Объединяем все пары для общего файла
    all_pairs = (
        filtered_github + filtered_kartaslov + filtered_loru + filtered_synthetic
    )
    print(f"\nВсего уникальных пар после объединения: {len(all_pairs)}")

    # Сохраняем
    if all_pairs:
        print("\n" + "=" * 60)
        print("Сохранение обработанных данных...")

        # Отдельные файлы по источникам
        if filtered_github:
            save_processed_data(filtered_github, "typo_github_clean.csv")
        if filtered_kartaslov:
            save_processed_data(filtered_kartaslov, "typo_kartaslov_clean.csv")
        if filtered_loru:
            save_processed_data(filtered_loru, "typo_loru_clean.csv")
        if filtered_synthetic:
            save_processed_data(filtered_synthetic, "typo_synthetic_clean.csv")

        # Общий файл для обучения по умолчанию
        save_processed_data(all_pairs, "typo_corpus_clean.csv")

        print("\n" + "=" * 60)
        print("Обработка данных успешно завершена!")
        print(f"Итоговый датасет для обучения: {len(all_pairs)} пар")
        print("=" * 60)

        # Показываем распределение по типам ошибок
        print("\nРаспределение по типам текстов:")

        single_word = sum(1 for orig, _ in all_pairs if len(orig.split()) == 1)
        multi_word = len(all_pairs) - single_word

        print(
            f"  Одиночные слова: {single_word} ({single_word/len(all_pairs)*100:.1f}%)"
        )
        print(f"  Предложения: {multi_word} ({multi_word/len(all_pairs)*100:.1f}%)")

        # Примеры итогового датасета
        print(f"\nПримеры из итогового датасета (первые 10):")
        for i, (orig, corr) in enumerate(all_pairs[:10]):
            print(f"  {i+1:2d}. '{orig}' → '{corr}'")

    else:
        print("\n" + "=" * 60)
        print("Внимание: не удалось получить данные из датасетов.")
        print("Создаём минимальный тестовый датасет...")

        test_pairs = [
            ("привет как дела", "привет, как дела"),
            ("он сказал что придет", "он сказал, что придёт"),
            ("преподататели", "преподаватели"),
            ("програмирование", "программирование"),
            ("карова", "корова"),
            ("я думаю что это правильно", "я думаю, что это правильно"),
        ]

        save_processed_data(test_pairs, "typo_corpus_clean.csv")

        print("\nСоздан минимальный тестовый датасет (6 пар)")


if __name__ == "__main__":
    main()
