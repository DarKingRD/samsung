"""
Модуль для обнаружения различных типов ошибок в тексте.
Обнаруживает орфографические, грамматические, пунктуационные и стилистические ошибки.
"""

import re
from typing import List, Dict, Any
import pymorphy2


class ErrorDetector:
    """Класс для обнаружения различных типов ошибок."""
    
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        
        # Правила для пунктуации
        self.comma_rules = [
            (r'\b(\w+)\s+(что|который|которая|которое|которые|когда|где|куда|откуда|почему|зачем)\s+', 
             'запятая перед союзом'),
            (r'\b(\w+)\s+(а|но|однако|зато)\s+', 'запятая перед союзом'),
            (r'\b(\w+)\s+(и|да)\s+(\w+)\s+(и|да)\s+', 'запятая в однородных членах'),
        ]
        
        # Правила для грамматики
        self.grammar_rules = [
            (r'\b(\w+)\s+(\w+)\s+(\w+)', self._check_agreement),
        ]
    
    def detect_errors(self, original_text: str, corrected_text: str) -> List[Dict[str, Any]]:
        """
        Обнаруживает ошибки, сравнивая исходный и исправленный текст.
        
        Args:
            original_text: исходный текст с ошибками
            corrected_text: исправленный текст
        
        Returns:
            список словарей с информацией об ошибках
        """
        errors = []
        
        if original_text == corrected_text:
            return errors
        
        # Разбиваем на слова для более точного сравнения
        original_words = original_text.split()
        corrected_words = corrected_text.split()
        
        # Находим различия
        i = 0
        j = 0
        current_error = None
        
        while i < len(original_words) or j < len(corrected_words):
            orig_word = original_words[i] if i < len(original_words) else None
            corr_word = corrected_words[j] if j < len(corrected_words) else None
            
            if orig_word == corr_word:
                if current_error:
                    errors.append(current_error)
                    current_error = None
                i += 1
                j += 1
            else:
                # Найдено различие
                if not current_error:
                    # Начало новой ошибки
                    start_pos = original_text.find(orig_word) if orig_word else len(original_text)
                    error_type = self._classify_error(orig_word, corr_word)
                    
                    current_error = {
                        'start': start_pos,
                        'end': start_pos + len(orig_word) if orig_word else len(original_text),
                        'original': orig_word or '',
                        'suggestions': [corr_word] if corr_word else [],
                        'confidence': 0.9,
                        'type': error_type
                    }
                else:
                    # Продолжение ошибки
                    if orig_word:
                        current_error['original'] += ' ' + orig_word
                        current_error['end'] = original_text.find(orig_word, current_error['start']) + len(orig_word)
                    if corr_word:
                        if current_error['suggestions']:
                            current_error['suggestions'][0] += ' ' + corr_word
                        else:
                            current_error['suggestions'] = [corr_word]
                
                if orig_word:
                    i += 1
                if corr_word:
                    j += 1
        
        if current_error:
            errors.append(current_error)
        
        # Дополнительно проверяем пунктуацию и грамматику
        punctuation_errors = self._detect_punctuation_errors(original_text)
        errors.extend(punctuation_errors)
        
        grammar_errors = self._detect_grammar_errors(original_text)
        errors.extend(grammar_errors)
        
        # Удаляем дубликаты и сортируем по позиции
        errors = self._deduplicate_errors(errors)
        errors.sort(key=lambda x: x['start'])
        
        return errors
    
    def _classify_error(self, original: str, corrected: str) -> str:
        """Классифицирует тип ошибки."""
        if not original or not corrected:
            return 'неизвестная ошибка'
        
        # Орфографическая ошибка (одно слово, похожие по написанию)
        if len(original.split()) == 1 and len(corrected.split()) == 1:
            if self._is_similar_spelling(original, corrected):
                return 'орфографическая ошибка'
        
        # Пунктуационная ошибка (разница только в знаках препинания)
        if self._is_punctuation_diff(original, corrected):
            return 'пунктуационная ошибка'
        
        # Грамматическая ошибка (разное количество слов или порядок)
        if len(original.split()) != len(corrected.split()):
            return 'грамматическая ошибка'
        
        # Стилистическая ошибка
        if self._is_style_diff(original, corrected):
            return 'стилистическая ошибка'
        
        return 'ошибка'
    
    def _is_similar_spelling(self, word1: str, word2: str) -> bool:
        """Проверяет, похожи ли слова по написанию (расстояние Левенштейна)."""
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        # Простая проверка: считаем количество одинаковых символов
        matches = sum(1 for a, b in zip(word1.lower(), word2.lower()) if a == b)
        similarity = matches / max(len(word1), len(word2))
        return similarity > 0.7
    
    def _is_punctuation_diff(self, text1: str, text2: str) -> bool:
        """Проверяет, отличается ли текст только пунктуацией."""
        # Удаляем пунктуацию и сравниваем
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
        return text1_clean == text2_clean
    
    def _is_style_diff(self, text1: str, text2: str) -> bool:
        """Проверяет, является ли разница стилистической."""
        # Проверяем регистр, порядок слов и т.д.
        if text1.lower() == text2.lower():
            return True
        return False
    
    def _detect_punctuation_errors(self, text: str) -> List[Dict[str, Any]]:
        """Обнаруживает пунктуационные ошибки."""
        errors = []
        
        for pattern, description in self.comma_rules:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Проверяем, есть ли уже запятая перед союзом
                pos = match.start()
                if pos > 0 and text[pos-1] != ',':
                    errors.append({
                        'start': pos,
                        'end': match.end(),
                        'original': match.group(),
                        'suggestions': [match.group().replace(' ', ', ', 1)],
                        'confidence': 0.8,
                        'type': 'пунктуационная ошибка'
                    })
        
        return errors
    
    def _detect_grammar_errors(self, text: str) -> List[Dict[str, Any]]:
        """Обнаруживает грамматические ошибки (согласование и т.д.)."""
        errors = []
        words = text.split()
        
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]
            
            # Проверяем согласование прилагательного и существительного
            parsed1 = self.morph.parse(word1)[0]
            parsed2 = self.morph.parse(word2)[0]
            
            # Если первое слово - прилагательное, а второе - существительное
            is_adj = 'ADJF' in parsed1.tag or 'ADJS' in parsed1.tag
            is_noun = 'NOUN' in parsed2.tag
            
            if is_adj and is_noun:
                # Проверяем согласование
                if not self._check_agreement(parsed1, parsed2):
                    start_pos = text.find(word1 + ' ' + word2)
                    errors.append({
                        'start': start_pos,
                        'end': start_pos + len(word1 + ' ' + word2),
                        'original': word1 + ' ' + word2,
                        'suggestions': [],  # Будет предложено моделью
                        'confidence': 0.7,
                        'type': 'грамматическая ошибка (согласование)'
                    })
        
        return errors
    
    def _check_agreement(self, adj_parsed, noun_parsed) -> bool:
        """Проверяет согласование прилагательного и существительного."""
        adj_gender = adj_parsed.tag.gender
        noun_gender = noun_parsed.tag.gender
        
        if adj_gender and noun_gender:
            return adj_gender == noun_gender
        
        return True
    
    def _deduplicate_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Удаляет дубликаты ошибок."""
        seen = set()
        unique_errors = []
        
        for error in errors:
            key = (error['start'], error['end'], error['original'])
            if key not in seen:
                seen.add(key)
                unique_errors.append(error)
        
        return unique_errors

