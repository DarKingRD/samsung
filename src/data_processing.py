import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import re
import random
from typing import List, Tuple, Dict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# СТРУКТУРА КАТАЛОГОВ
# ============================================================================
BASE_DIR = Path('.')
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ФУНКЦИИ ОЧИСТКИ
# ============================================================================

def is_russian_text(text: str) -> bool:
    """Проверяет, содержит ли текст русские символы"""
    russian_chars = re.findall(r'[а-яёА-ЯЁ]', str(text))
    return len(russian_chars) > 0

def looks_like_code(text: str) -> bool:
    """Проверяет, похож ли текст на код"""
    if not isinstance(text, str):
        return False
    
    code_patterns = [
        r'<[^>]+>',
        r'\$\{[^}]+\}',
        r'function|const|let|var|=>',
        r'def |class |import ',
        r'[(){}\[\]{}]',
        r'^\s*//',
        r'^\s*\*',
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_patterns)

def looks_like_markup(text: str) -> bool:
    """Проверяет, похож ли текст на разметку"""
    if not isinstance(text, str):
        return False
    
    markup_patterns = [
        r'<[^>]+>',
        r'^\s*#+\s+',
        r'^\s*[-*]\s+',
        r'\[.+\]\(.+\)',
        r'```|~~~',
        r'``.+``',
    ]
    
    return any(re.search(pattern, text, re.MULTILINE | re.IGNORECASE) 
               for pattern in markup_patterns)

def is_valid_text(text: str) -> bool:
    """Проверяет, валидный ли текст для обучения"""
    text = str(text).strip()
    
    if not text:
        return False
    if looks_like_code(text) or looks_like_markup(text):
        return False
    if len(text) < 3:
        return False
    if not is_russian_text(text):
        return False
    
    return True

# ============================================================================
# СИНТЕТИЧЕСКИЕ ДАННЫЕ - ВСЕ ТИПЫ ОШИБОК
# ============================================================================

class SyntheticErrorGenerator:
    """Генератор синтетических ошибок ДЛЯ ВСЕХ ТИПОВ"""
    
    # Похожие русские буквы
    SIMILAR_CHARS = {
        'о': 'а',
        'а': 'о',
        'е': 'и',
        'и': 'е',
        'с': 'ц',
        'ц': 'с',
        'ш': 'щ',
        'щ': 'ш',
        'л': 'н',
        'н': 'л',
        'р': 'п',
        'п': 'р',
        'б': 'в',
        'в': 'б',
        'д': 'т',
        'т': 'д',
        'х': 'к',
        'к': 'х',
        'ж': 'з',
        'з': 'ж',
        'г': 'к',
        'м': 'л',
    }
    
    # ===== ОРФОГРАФИЧЕСКИЕ ОШИБКИ =====
    
    @staticmethod
    def typo_swap_chars(word: str) -> str:
        """Замена соседних символов (транспозиция)"""
        if len(word) < 2:
            return word
        pos = random.randint(0, len(word) - 2)
        word_list = list(word)
        word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
        return ''.join(word_list)
    
    @staticmethod
    def typo_delete_char(word: str) -> str:
        """Удаление символа (пропуск буквы)"""
        if len(word) < 2:
            return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos + 1:]
    
    @staticmethod
    def typo_duplicate_char(word: str) -> str:
        """Дублирование символа"""
        if len(word) < 1:
            return word
        pos = random.randint(0, len(word) - 1)
        return word[:pos + 1] + word[pos] + word[pos + 1:]
    
    @staticmethod
    def typo_replace_similar(word: str) -> str:
        """Замена на похожий символ"""
        for char, similar in SyntheticErrorGenerator.SIMILAR_CHARS.items():
            if char in word:
                pos = word.index(char)
                return word[:pos] + similar + word[pos + 1:]
        return word
    
    @staticmethod
    def typo_insert_char(word: str) -> str:
        """Вставка символа"""
        if len(word) < 1:
            return word
        russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        pos = random.randint(0, len(word))
        char = random.choice(russian_alphabet)
        return word[:pos] + char + word[pos:]
    
    @staticmethod
    def typo_replace_char(word: str) -> str:
        """Замена на случайный символ"""
        if len(word) < 1:
            return word
        russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        pos = random.randint(0, len(word) - 1)
        char = random.choice(russian_alphabet)
        return word[:pos] + char + word[pos + 1:]
    
    # ===== ПУНКТУАЦИОННЫЕ ОШИБКИ =====
    
    @staticmethod
    def punctuation_remove_comma(text: str) -> str:
        """Удаление запятой"""
        if ',' not in text:
            return text
        pos = text.index(',')
        return text[:pos] + text[pos + 1:]
    
    @staticmethod
    def punctuation_add_comma(text: str) -> str:
        """Добавление лишней запятой"""
        words = text.split()
        if len(words) < 2:
            return text
        pos = random.randint(0, len(words) - 2)
        words[pos] = words[pos] + ','
        return ' '.join(words)
    
    @staticmethod
    def punctuation_period_to_comma(text: str) -> str:
        """Замена точки на запятую"""
        if '.' not in text:
            return text
        return text.replace('.', ',', 1)
    
    @staticmethod
    def punctuation_remove_period(text: str) -> str:
        """Удаление точки в конце"""
        if text.endswith('.'):
            return text[:-1]
        return text
    
    # ===== ГРАММАТИЧЕСКИЕ ОШИБКИ =====
    
    @staticmethod
    def grammar_ne_agreement(text: str) -> str:
        """Ошибка в написании "не" с словами"""
        words = text.split()
        
        # Примеры: не смелость -> несмелость
        ne_words = {
            'не смелость': 'несмелость',
            'не зрелость': 'незрелость',
            'не доверие': 'недоверие',
            'не внимание': 'невнимание',
            'не обход': 'необход',
            'не нужно': 'ненужно',
            'не должно': 'недолжно',
        }
        
        for i in range(len(words) - 1):
            phrase = (words[i] + ' ' + words[i + 1]).lower()
            if phrase in ne_words:
                # Меняем обратно: правильный -> неправильный
                words[i] = ne_words[phrase]
                words[i + 1] = ''
                return ' '.join(w for w in words if w)
        
        return text
    
    @staticmethod
    def grammar_case_error(text: str) -> str:
        """Ошибка в падеже (управление)"""
        # Примеры
        replacements = [
            ('с отпуска', 'из отпуска'),
            ('по прошествию', 'по прошествии'),
            ('до нас', 'нас'),
            ('в деле', 'на деле'),
        ]
        
        for incorrect, correct in replacements:
            if incorrect in text:
                return text.replace(incorrect, correct, 1)
        
        return text
    
    @staticmethod
    def grammar_tense_error(text: str) -> str:
        """Ошибка в времени глагола"""
        replacements = [
            ('был', 'было'),
            ('была', 'было'),
            ('имею', 'имею'),
            ('делаю', 'делал'),
        ]
        
        for old, new in replacements:
            if old in text:
                return text.replace(old, new, 1)
        
        return text
    
    # ===== СМЫСЛОВЫЕ ОШИБКИ =====
    
    @staticmethod
    def semantic_word_order(text: str) -> str:
        """Ошибка в порядке слов"""
        words = text.split()
        
        if len(words) >= 3:
            # Случайно переставляем два слова
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
        
        return ' '.join(words)
    
    @staticmethod
    def semantic_wrong_word(text: str) -> str:
        """Замена слова на похожее по смыслу но неправильное"""
        replacements = [
            ('благодаря', 'из-за'),
            ('однако', 'потому что'),
            ('хотя', 'так как'),
            ('потому что', 'несмотря на'),
        ]
        
        for old, new in replacements:
            if old in text:
                return text.replace(old, new, 1)
        
        return text
    
    # ===== СТИЛИСТИЧЕСКИЕ ОШИБКИ =====
    
    @staticmethod
    def stylistic_repetition(text: str) -> str:
        """Повторение слова"""
        words = text.split()
        
        if len(words) >= 2:
            # Повторяем случайное слово
            i = random.randint(0, len(words) - 1)
            words.insert(i + 1, words[i])
        
        return ' '.join(words)
    
    @staticmethod
    def stylistic_formal_to_informal(text: str) -> str:
        """Смешивание стилей: формальное -> неформальное"""
        replacements = [
            ('высокоуважаемый', 'привет'),
            ('позвольте', 'давайте'),
            ('содействие', 'помощь'),
            ('препятствие', 'проблема'),
        ]
        
        for old, new in replacements:
            if old in text:
                return text.replace(old, new, 1)
        
        return text
    
    # ===== ГЕНЕРАТОР =====
    
    @classmethod
    def generate(cls, text: str, error_type: str = None) -> Tuple[str, str]:
        """
        Генерирует ошибку в тексте
        
        Returns:
            (error_text, error_category)
        """
        if len(text) < 3:
            return text, 'none'
        
        generators = [
            # Орфографические
            (cls.typo_swap_chars, 'spelling'),
            (cls.typo_delete_char, 'spelling'),
            (cls.typo_duplicate_char, 'spelling'),
            (cls.typo_replace_similar, 'spelling'),
            (cls.typo_insert_char, 'spelling'),
            (cls.typo_replace_char, 'spelling'),
            
            # Пунктуационные
            (cls.punctuation_remove_comma, 'punctuation'),
            (cls.punctuation_add_comma, 'punctuation'),
            (cls.punctuation_period_to_comma, 'punctuation'),
            (cls.punctuation_remove_period, 'punctuation'),
            
            # Грамматические
            (cls.grammar_ne_agreement, 'grammar'),
            (cls.grammar_case_error, 'grammar'),
            (cls.grammar_tense_error, 'grammar'),
            
            # Смысловые
            (cls.semantic_word_order, 'semantics'),
            (cls.semantic_wrong_word, 'semantics'),
            
            # Стилистические
            (cls.stylistic_repetition, 'stylistic'),
            (cls.stylistic_formal_to_informal, 'stylistic'),
        ]
        
        if error_type:
            # Фильтруем по типу
            generators = [g for g in generators if g[1] == error_type]
            if not generators:
                return text, 'none'
        
        generator, error_cat = random.choice(generators)
        
        try:
            # Для методов со словами нужно применить к отдельному слову
            if error_cat == 'spelling' and hasattr(generator, '__name__') and 'word' in str(text).lower():
                words = text.split()
                if words:
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = generator(words[idx])
                    return ' '.join(words), error_cat
            
            result = generator(text)
            if result != text:
                return result, error_cat
            else:
                return text, 'none'
        except:
            return text, 'none'

def generate_synthetic_dataset(source_texts: List[str], 
                              num_per_text: int = 3,
                              error_types: List[str] = None) -> pd.DataFrame:
    """
    Генерирует синтетические ошибки из правильных текстов
    
    Args:
        source_texts: правильные тексты
        num_per_text: количество ошибок на текст
        error_types: типы ошибок для генерации
    """
    if error_types is None:
        error_types = ['spelling', 'punctuation', 'grammar', 'semantics', 'stylistic']
    
    records = []
    
    for text in tqdm(source_texts, desc="🤖 Синтетические данные", leave=False):
        for error_type in error_types[:num_per_text]:
            try:
                error_text, actual_type = SyntheticErrorGenerator.generate(text, error_type)
                
                if error_text != text and is_valid_text(error_text):
                    records.append({
                        'source': error_text,
                        'target': text,
                        'weight': 1.0,
                        'type': 'synthetic',
                        'error_category': actual_type
                    })
            except:
                continue
    
    return pd.DataFrame(records)

# ============================================================================
# ОБРАБОТЧИКИ ДАННЫХ
# ============================================================================

def process_kartaslov() -> pd.DataFrame:
    """
    Обработка КАРТАСЛОВА
    Тип: ОРФОГРАФИЧЕСКИЕ ОШИБКИ (опечатки)
    """
    dfs = []
    
    files = [
        RAW_DIR / "kartaslov" / "orfo_and_typos.L1_5.csv",
        RAW_DIR / "kartaslov" / "orfo_and_typos.L1_5-PHON.csv",
        Path("orfo_and_typos.L1_5.csv"),
        Path("orfo_and_typos.L1_5-PHON.csv"),
    ]
    
    for csv_file in files:
        if not csv_file.exists():
            continue
        
        try:
            df = pd.read_csv(csv_file, sep=';', on_bad_lines='skip')
            
            if len(df) == 0:
                continue
            
            print(f"📄 {csv_file.name}")
            
            cols = {col.upper(): col for col in df.columns}
            correct_col = cols.get('CORRECT')
            mistake_col = cols.get('MISTAKE')
            weight_col = cols.get('WEIGHT')
            
            if not correct_col or not mistake_col:
                continue
            
            df_clean = pd.DataFrame({
                'source': df[mistake_col].astype(str).str.strip(),
                'target': df[correct_col].astype(str).str.strip(),
                'weight': df[weight_col].astype(float) if weight_col else 1.0,
                'type': 'kartaslov',
                'error_category': 'spelling'
            })
            
            # Фильтрация
            before = len(df_clean)
            df_clean = df_clean[
                (df_clean['source'] != df_clean['target']) &
                (df_clean['source'].str.len() >= 2) &
                (df_clean['target'].str.len() >= 2) &
                (df_clean['source'].apply(is_russian_text)) &
                (df_clean['target'].apply(is_russian_text))
            ].reset_index(drop=True)
            after = len(df_clean)
            
            print(f"   ✓ {after:,} примеров орфографических ошибок")
            dfs.append(df_clean)
            
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_lorugec() -> Dict[str, pd.DataFrame]:
    """
    Обработка LORUGEC
    Типы: ГРАММАТИЧЕСКИЕ, ПУНКТУАЦИОННЫЕ, СМЫСЛОВЫЕ ошибки
    
    Returns:
        dict с ошибками по типам: grammar, punctuation, semantics
    """
    
    files = [
        RAW_DIR / "loru" / "LORuGEC.xlsx",
        Path("LORuGEC.xlsx"),
    ]
    
    for xlsx_file in files:
        if not xlsx_file.exists():
            continue
        
        try:
            df = pd.read_excel(xlsx_file, sheet_name=0)
            
            print(f"📄 {xlsx_file.name}")
            
            # Основные столбцы
            initial_col = next((col for col in df.columns if 'Initial' in col), None)
            correct_col = next((col for col in df.columns if 'Correct' in col and 'Initial' not in col), None)
            section_col = next((col for col in df.columns if 'section' in col.lower()), None)
            
            if not initial_col or not correct_col or not section_col:
                print(f"   ⚠️  Не найдены нужные столбцы")
                continue
            
            # Маппинг типов ошибок
            type_map = {
                'Spelling': 'spelling',
                'Punctuation': 'punctuation',
                'Grammar': 'grammar',
                'Semantics': 'semantics',
            }
            
            result = {}
            
            for section, error_cat in type_map.items():
                section_df = df[df[section_col] == section].copy()
                
                if len(section_df) == 0:
                    continue
                
                df_clean = pd.DataFrame({
                    'source': section_df[initial_col].astype(str).str.strip(),
                    'target': section_df[correct_col].astype(str).str.strip(),
                    'weight': 1.0,
                    'type': 'lorugec',
                    'error_category': error_cat
                })
                
                # Фильтрация
                before = len(df_clean)
                df_clean = df_clean[
                    (df_clean['source'] != df_clean['target']) &
                    (df_clean['source'].str.len() >= 3) &
                    (df_clean['target'].str.len() >= 3) &
                    (df_clean['source'].apply(is_valid_text)) &
                    (df_clean['target'].apply(is_valid_text))
                ].reset_index(drop=True)
                after = len(df_clean)
                
                if after > 0:
                    print(f"   ✓ {after:,} примеров {error_cat}")
                    result[error_cat] = df_clean
            
            return result
            
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
    
    return {}

# ============================================================================
# ОСНОВНОЙ ПРОЦЕСС
# ============================================================================

print("\n" + "="*80)
print("🚀 ОБРАБОТКА ДАННЫХ ДЛЯ СИСТЕМЫ ИСПРАВЛЕНИЯ ОШИБОК")
print("="*80 + "\n")

print("Задача: Исправление опечаток ВСЕХ типов")
print("  • Орфографические (опечатки, замены, пропуски)")
print("  • Пунктуационные (запятые, точки)")
print("  • Грамматические (согласование, падежи, времена)")
print("  • Смысловые (порядок слов, выбор слова)")
print("  • Стилистические (повторения, стиль)\n")

print("="*80)
print("📋 ШАГ 1: ИСХОДНЫЕ ДАННЫЕ")
print("="*80 + "\n")

# Картасlov - орфографические
print("1️⃣  КАРТАСLOV (орфографические ошибки)")
print("-" * 80)
kartaslov_df = process_kartaslov()

# LORuGEC - грамматические, пунктуационные, смысловые
print("\n2️⃣  LORUGEC (грамматические, пунктуационные, смысловые)")
print("-" * 80)
lorugec_dict = process_lorugec()

# ШАГ 2: Объединяем исходные
print("\n" + "="*80)
print("🔗 ШАГ 2: ОБЪЕДИНЕНИЕ ИСХОДНЫХ ДАННЫХ")
print("="*80 + "\n")

all_original = []

if len(kartaslov_df) > 0:
    kartaslov_df.to_csv(PROCESSED_DIR / "kartaslov_spelling.csv", index=False)
    all_original.append(kartaslov_df)
    print(f"✅ kartaslov_spelling.csv: {len(kartaslov_df):,} примеров")

for error_cat, df in lorugec_dict.items():
    if len(df) > 0:
        filename = f"lorugec_{error_cat}.csv"
        df.to_csv(PROCESSED_DIR / filename, index=False)
        all_original.append(df)
        print(f"✅ {filename}: {len(df):,} примеров")

if all_original:
    original_combined = pd.concat(all_original, ignore_index=True)
    original_combined = original_combined.drop_duplicates(subset=['source', 'target'])
    print(f"\n✅ Всего исходных примеров: {len(original_combined):,}")
else:
    original_combined = pd.DataFrame()
    print("❌ Нет исходных данных")

# ШАГ 3: Синтетические данные
print("\n" + "="*80)
print("🤖 ШАГ 3: СИНТЕТИЧЕСКИЕ ДАННЫЕ (для всех типов ошибок)")
print("="*80 + "\n")

if len(original_combined) > 0:
    # Берем правильные варианты для генерации синтетических ошибок
    correct_texts = original_combined['target'].unique().tolist()
    
    print(f"Генерирую синтетические ошибки из {len(correct_texts):,} текстов...")
    print("  • Орфографические (swap, delete, duplicate, replace)")
    print("  • Пунктуационные (запятые, точки)")
    print("  • Грамматические (согласование, управление)")
    print("  • Смысловые (порядок слов, выбор слова)")
    print("  • Стилистические (повторения, стиль)\n")
    
    synthetic_df = generate_synthetic_dataset(
        correct_texts,
        num_per_text=5,  # 5 типов ошибок на каждый текст
        error_types=['spelling', 'punctuation', 'grammar', 'semantics', 'stylistic']
    )
    
    if len(synthetic_df) > 0:
        synthetic_df.to_csv(PROCESSED_DIR / "synthetic_errors.csv", index=False)
        print(f"\n✅ synthetic_errors.csv: {len(synthetic_df):,} примеров")
        
        # По категориям
        print("\nПо типам ошибок:")
        for cat, count in synthetic_df['error_category'].value_counts().items():
            print(f"  • {cat}: {count:,}")
    else:
        synthetic_df = pd.DataFrame()
        print("⚠️  Не удалось сгенерировать синтетические данные")
else:
    synthetic_df = pd.DataFrame()

# ШАГ 4: Финальный датасет
print("\n" + "="*80)
print("✨ ШАГ 4: ФИНАЛЬНЫЙ ДАТАСЕТ")
print("="*80 + "\n")

all_data = []

if len(original_combined) > 0:
    all_data.append(original_combined)

if len(synthetic_df) > 0:
    all_data.append(synthetic_df)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['source', 'target'])
    final_df = final_df.reset_index(drop=True)
    
    # Сохраняем полный датасет
    final_df.to_csv(PROCESSED_DIR / "all_train.csv", index=False)
    print(f"✅ all_train.csv: {len(final_df):,} примеров")
    
    # Готовый файл для обучения
    train_df = final_df[['source', 'target', 'error_category', 'weight']].copy()
    train_df.columns = ['input_text', 'output_text', 'error_type', 'weight']
    train_df.to_csv(PROCESSED_DIR / "all_train_enhanced.csv", index=False)
    print(f"✅ all_train_enhanced.csv: {len(train_df):,} примеров ← ДЛЯ ОБУЧЕНИЯ")
else:
    final_df = pd.DataFrame()
    print("❌ Нет данных")

# ============================================================================
# СТАТИСТИКА И ПРИМЕРЫ
# ============================================================================

print("\n" + "="*80)
print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
print("="*80 + "\n")

if len(final_df) > 0:
    print(f"📈 Всего примеров: {len(final_df):,}\n")
    
    # По источникам
    print("По источникам:")
    for src, count in final_df['type'].value_counts().items():
        print(f"  • {src}: {count:,}")
    
    # По типам ошибок
    print("\nПо типам ошибок:")
    for error_type, count in final_df['error_category'].value_counts().items():
        print(f"  • {error_type}: {count:,}")
    
    # Примеры
    print(f"\n📝 Примеры (по типам ошибок):")
    
    for error_type in ['spelling', 'punctuation', 'grammar', 'semantics', 'stylistic']:
        sample = final_df[final_df['error_category'] == error_type].head(1)
        if len(sample) > 0:
            row = sample.iloc[0]
            print(f"\n  🔹 {error_type.upper()}")
            print(f"     ❌ {row['source'][:70]}")
            print(f"     ✅ {row['target'][:70]}")

print("\n" + "="*80)
print("✅ ГОТОВО!")
print("="*80)
print("\n📂 Выходные файлы в data/processed/:")
print("   ✅ kartaslov_spelling.csv - орфографические")
print("   ✅ lorugec_*.csv - грамматические, пунктуационные, смысловые")
print("   ✅ synthetic_errors.csv - синтетические (все типы)")
print("   ✅ all_train.csv - полный датасет")
print("   ✅ all_train_enhanced.csv - для обучения модели")
print("\n🚀 Дальше: python improved_train.py\n")
