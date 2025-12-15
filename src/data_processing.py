"""
Скрипт для скачивания и обработки датасетов с опечатками.
"""

import os
import json
import requests
import zipfile
import gzip
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

# Создаём директории
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки обработки
MAX_GITHUB_LINES = None    # Максимальное количество строк из GitHub Typo Corpus (None = все)
MIN_TEXT_LENGTH = 3        # Минимальная длина текста для обработки
MAX_TEXT_LENGTH = 600      # Максимальная длина текста (оставляем длинные предложения)
ONLY_RUSSIAN = True        # Только русский язык
MAX_WORD_LENGTH = 30       # Максимальная длина слова


def is_russian_text(text: str) -> bool:
    """Проверяет, является ли текст русским."""
    if not text:
        return False
    
    # Подсчитываем русские символы
    russian_chars = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c in 'ёЁ')
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
        'api', 'json', 'xml', 'html', 'css', 'js', 'http', 'https', 'url',
        'function', 'method', 'class', 'import', 'export', 'return', 'var', 'let', 'const',
        'config', 'parameter', 'parameter', 'option', 'setting', 'default',
        'react', 'angular', 'vue', 'node', 'npm', 'package',
        'sql', 'database', 'query', 'table', 'column',
        'git', 'commit', 'branch', 'merge', 'pull', 'push',
        '//', '/*', '*/', '<?', '?>', '<script', '</script>', '<div', '</div>',
        '=', ';', '{', '}', '[', ']', '(', ')', '->', '=>', '::',
    ]
    
    # Проверяем наличие технических маркеров
    for marker in technical_markers:
        if marker in text_lower:
            return True
    
    # Проверяем наличие слишком много специальных символов (код)
    special_chars = sum(1 for c in text if c in '{}[]();=<>/\\|&%$#@!~`')
    if len(text) > 0 and (special_chars / len(text)) > 0.1:
        return True
    
    return False


def clean_markdown(text: str) -> str:
    """Убирает markdown/разметку и лишние пробелы."""
    import re
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
    t = re.sub(r"\\s+", " ", t).strip()
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
            # Простая проверка на схожесть (расстояние Левенштейна можно использовать)
            common_chars = sum(1 for a, b in zip(orig_word, corr_word) if a == b)
            min_len = min(len(orig_word), len(corr_word))
            if min_len > 0 and (common_chars / min_len) < 0.5:  # Меньше 50% общих символов
                return False
    
    return True


def download_file(url: str, filepath: Path):
    """Скачивает файл по URL."""
    if filepath.exists():
        print(f"Файл {filepath} уже существует, пропускаем скачивание")
        return
    
    print(f"Скачиваем {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"Скачано: {filepath}")


def process_github_typo_corpus():
    """Обрабатывает GitHub Typo Corpus."""
    print("\n=== Обработка GitHub Typo Corpus ===")
    
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
            with gzip.open(dataset_path, 'rb') as f_in:
                file_size = os.path.getsize(dataset_path)
                chunk_size = 1024 * 1024  # 1MB
                with open(jsonl_path, 'wb') as f_out:
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Распаковка") as pbar:
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
    
    # Обрабатываем JSONL файл
    if jsonl_path.exists():
        file_size = os.path.getsize(jsonl_path)
        print(f"Обрабатываем {jsonl_path} (размер: {file_size / 1024 / 1024:.2f} MB)...")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                # Читаем построчно (JSONL формат)
                # Ограничиваем количество строк для ускорения
                max_lines = MAX_GITHUB_LINES
                total_desc = f"Обработка строк (макс. {max_lines})" if max_lines else "Обработка строк"
                
                for line_num, line in enumerate(tqdm(f, desc=total_desc), 1):
                    if max_lines and line_num > max_lines:
                        break
                    
                    try:
                        commit = json.loads(line.strip())
                        if 'edits' not in commit:
                            continue
                        
                        processed_commits += 1
                        
                        # Обрабатываем каждую правку (берём все изменения, т.к. русские is_typo размечены плохо)
                        for edit in commit.get('edits', []):
                            src_text = edit.get('src', {}).get('text', '').strip()
                            tgt_text = edit.get('tgt', {}).get('text', '').strip()
                            lang = edit.get('src', {}).get('lang', '')

                            # Чистим markdown/разметку
                            src_text = clean_markdown(src_text)
                            tgt_text = clean_markdown(tgt_text)
                            
                            # Пропускаем пустые тексты
                            if not src_text or not tgt_text or src_text == tgt_text:
                                continue
                            
                            # Фильтруем по языку (берём только русский).
                            # Принимаем: метка rus ИЛИ русские буквы в тексте, даже если lang=eng/None.
                            if ONLY_RUSSIAN:
                                if not (
                                    lang == "rus"
                                    or is_russian_text(src_text)
                                    or is_russian_text(tgt_text)
                                ):
                                    continue
                            
                            # Проверяем минимальную длину (верхний предел не ограничиваем для GitHub)
                            if len(src_text) < MIN_TEXT_LENGTH or len(tgt_text) < MIN_TEXT_LENGTH:
                                continue

                            # Для предложений из GitHub оставляем без строгой проверки валидности:
                            # это реальные правки, даже если отличаются сильно (пунктуация/перестановки).
                            pairs.append((src_text, tgt_text))
                            processed_edits += 1
                    
                    except json.JSONDecodeError as e:
                        if line_num < 10:  # Показываем только первые ошибки
                            print(f"Ошибка JSON на строке {line_num}: {e}")
                        continue
                    except Exception as e:
                        if line_num < 10:
                            print(f"Ошибка на строке {line_num}: {e}")
                        continue
            
            print(f"Обработано коммитов: {processed_commits}")
            print(f"Обработано правок: {processed_edits}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Файл датасета не найден. Пропускаем GitHub Typo Corpus.")
    
    return pairs


def process_kartaslov():
    """Обрабатывает датасет Kartaslov."""
    print("\n=== Обработка Kartaslov ===")
    
    # Скачиваем репозиторий
    repo_url = "https://github.com/dkulagin/kartaslov/archive/refs/heads/master.zip"
    zip_path = RAW_DIR / "kartaslov.zip"
    
    download_file(repo_url, zip_path)
    
    # Распаковываем
    extract_dir = RAW_DIR / "kartaslov-master"  # Исправляем путь
    if not extract_dir.exists():
        print("Распаковываем архив...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    
    pairs = []
    dataset_dir = extract_dir / "dataset" / "orfo_and_typos"
    
    if not dataset_dir.exists():
        print(f"Директория {dataset_dir} не найдена!")
        return pairs
    
    # Обрабатываем CSV файлы
    csv_files = [
        dataset_dir / "orfo_and_typos.L1_5.csv",
        dataset_dir / "orfo_and_typos.L1_5+PHON.csv"
    ]
    
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Файл {csv_file} не найден, пропускаем")
            continue
        
        print(f"Обрабатываем {csv_file.name}...")
        try:
            # Читаем CSV с разделителем ';'
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8', on_bad_lines='skip')
            
            # Проверяем наличие нужных столбцов
            if 'CORRECT' in df.columns and 'MISTAKE' in df.columns:
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {csv_file.name}"):
                    correct = str(row['CORRECT']).strip()
                    mistake = str(row['MISTAKE']).strip()
                    
                    # Пропускаем пустые
                    if not correct or not mistake or correct == mistake:
                        continue
                    
                    # Фильтруем только русские слова
                    if ONLY_RUSSIAN:
                        if not is_russian_text(correct) or not is_russian_text(mistake):
                            continue
                    
                    # Фильтруем слишком длинные слова
                    if len(correct) > MAX_WORD_LENGTH or len(mistake) > MAX_WORD_LENGTH:
                        continue
                    
                    # Проверяем валидность пары
                    if not is_valid_typo_pair(mistake, correct):
                        continue
                    
                    # Добавляем пару (ошибка -> исправление)
                    pairs.append((mistake, correct))
            else:
                print(f"  Неверный формат столбцов в {csv_file.name}")
                print(f"  Найденные столбцы: {list(df.columns)}")
        
        except Exception as e:
            print(f"Ошибка при обработке {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Получено {len(pairs)} пар из Kartaslov")
    return pairs


def process_loru_gec():
    """
    Обрабатывает датасет LoRuGEC (грамматика/стилистика/семантика).
    Использует файл LORuGEC.xlsx из репозитория:
    https://github.com/ReginaNasyrova/LORuGEC

    Мы автоматически скачиваем LORuGEC.xlsx в data/raw/
    и извлекаем пары (исходное предложение -> исправленное предложение).
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

        for _, row in df.iterrows():
            src_text = str(row.get(src_col, "")).strip()
            tgt_text = str(row.get(tgt_col, "")).strip()

            src_text = clean_markdown(src_text)
            tgt_text = clean_markdown(tgt_text)

            if not src_text or not tgt_text or src_text == tgt_text:
                continue

            pairs.append((src_text, tgt_text))

        print(f"Получено {len(pairs)} пар из LoRuGEC")
    except Exception as e:
        print(f"Ошибка при обработке LORuGEC.xlsx: {e}")

    return pairs


def save_processed_data(pairs: List[Tuple[str, str]], filename: str):
    """Сохраняет обработанные данные в CSV."""
    df = pd.DataFrame(pairs, columns=['original', 'corrected'])
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Сохранено {len(pairs)} пар в {output_path}")
    return output_path


def filter_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Финальная фильтрация пар (общая логика для всех источников)."""
    # Удаляем дубликаты и пустые строки
    unique_pairs = list({
        (orig.strip(), corr.strip())
        for orig, corr in pairs
        if str(orig).strip() and str(corr).strip()
    })

    filtered_pairs: List[Tuple[str, str]] = []

    for orig, corr in unique_pairs:
        orig_words = orig.split()
        corr_words = corr.split()

        is_sentence = len(orig_words) > 1 or len(corr_words) > 1

        # Фильтруем по языку (если включено)
        if ONLY_RUSSIAN:
            if not (is_russian_text(orig) or is_russian_text(corr)):
                continue

        # Для предложений: допускаем больше различий и не режем по техническому фильтру.
        if is_sentence:
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
    print("Начинаем обработку датасетов...")

    github_pairs: List[Tuple[str, str]] = []
    kartaslov_pairs: List[Tuple[str, str]] = []
    loru_pairs: List[Tuple[str, str]] = []

    # Обрабатываем GitHub Typo Corpus
    try:
        github_pairs = process_github_typo_corpus()
        print(f"Получено {len(github_pairs)} пар из GitHub Typo Corpus")
    except Exception as e:
        print(f"Ошибка при обработке GitHub Typo Corpus: {e}")
    
    # Обрабатываем Kartaslov
    try:
        kartaslov_pairs = process_kartaslov()
        print(f"Получено {len(kartaslov_pairs)} пар из Kartaslov")
    except Exception as e:
        print(f"Ошибка при обработке Kartaslov: {e}")

    # Обрабатываем LoRuGEC (если есть)
    try:
        loru_pairs = process_loru_gec()
        if loru_pairs:
            print(f"Получено {len(loru_pairs)} пар из LoRuGEC")
    except Exception as e:
        print(f"Ошибка при обработке LoRuGEC: {e}")

    # Финальная фильтрация по источникам
    print("\nФинальная фильтрация пар...")
    filtered_github = filter_pairs(github_pairs) if github_pairs else []
    filtered_kartaslov = filter_pairs(kartaslov_pairs) if kartaslov_pairs else []
    filtered_loru = filter_pairs(loru_pairs) if loru_pairs else []

    print(f"Всего уникальных пар из GitHub после фильтрации: {len(filtered_github)}")
    print(f"Всего уникальных пар из Kartaslov после фильтрации: {len(filtered_kartaslov)}")
    if filtered_loru:
        print(f"Всего уникальных пар из LoRuGEC после фильтрации: {len(filtered_loru)}")

    # Объединяем все пары для общего файла
    all_pairs = filtered_github + filtered_kartaslov + filtered_loru
    print(f"Всего уникальных пар после объединения: {len(all_pairs)}")

    # Сохраняем
    if all_pairs:
        # Отдельные файлы по источникам
        if filtered_github:
            save_processed_data(filtered_github, "typo_github.csv")
        if filtered_kartaslov:
            save_processed_data(filtered_kartaslov, "typo_kartaslov.csv")
        if filtered_loru:
            save_processed_data(filtered_loru, "typo_loru.csv")

        # Общий файл для обучения по умолчанию
        save_processed_data(all_pairs, "typo_corpus.csv")
        print("\nОбработка данных завершена!")
    else:
        print("\nВнимание: не удалось получить данные из датасетов.")
        print("Создаём тестовый датасет для демонстрации...")
        # Создаём тестовый датасет для демонстрации
        test_pairs = [
            # Орфографические ошибки
            ("привет как дела", "привет, как дела"),
            ("он сказал что придет", "он сказал, что придёт"),
            ("он не пришел на встречу", "он не пришёл на встречу"),
            ("я приду завтра", "я приду завтра"),
            ("он придет позже", "он придёт позже"),
            # Грамматические ошибки (запятые)
            ("я думаю что это правильно", "я думаю, что это правильно"),
            ("мама сказала что придет", "мама сказала, что придёт"),
            ("он знал что делать", "он знал, что делать"),
            ("я вижу что ты прав", "я вижу, что ты прав"),
            ("она сказала что устала", "она сказала, что устала"),
            # Согласование
            ("красивый дом", "красивый дом"),
            ("мама мыла раму", "мама мыла раму"),
            ("это очень интересно", "это очень интересно"),
            ("я хочу купить машину", "я хочу купить машину"),
            ("сегодня хорошая погода", "сегодня хорошая погода"),
            ("я люблю программирование", "я люблю программирование"),
            # Дополнительные примеры
            ("когда я пришел дом был пуст", "когда я пришёл, дом был пуст"),
            ("он не знал что сказать", "он не знал, что сказать"),
            ("я видел что происходит", "я видел, что происходит"),
            ("она поняла что ошиблась", "она поняла, что ошиблась"),
        ]
        save_processed_data(test_pairs, "typo_corpus.csv")


if __name__ == "__main__":
    main()
