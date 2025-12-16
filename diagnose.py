from src.inference import TypoCorrectionInference
import torch


def diagnose_model():
    """Диагностика загруженной модели."""
    corrector = TypoCorrectionInference()

    print("=" * 60)
    print("ДИАГНОСТИКА МОДЕЛИ")
    print("=" * 60)

    # Проверяем, загружена ли модель
    if not corrector.use_model:
        print("✗ Модель НЕ загружена!")
        print(f"Путь к модели: {corrector.model_path}")
        print("Файлы в папке:")
        if corrector.model_path.exists():
            for f in corrector.model_path.iterdir():
                print(f"  - {f.name}")
        return

    print("✓ Модель загружена")
    print(f"Путь: {corrector.model_path}")
    print(f"Тип модели: {type(corrector.model).__name__}")
    print(f"Тип токенизатора: {type(corrector.tokenizer).__name__}")

    # Проверяем, обучена ли модель на исправлении опечаток
    test_text = "привит как дел"
    print(f"\nТестируем на тексте: '{test_text}'")

    # 1. Проверяем токенизацию
    tokens = corrector.tokenizer.tokenize(test_text)
    print(f"Токены: {tokens}")

    # 2. Проверяем, что модель делает без промпта
    print("\n" + "-" * 40)
    print("Тест 1: Простая генерация без промпта")
    inputs = corrector.tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = corrector.model.generate(
            **inputs,
            max_length=50,
            num_beams=1,
            do_sample=True,
            temperature=0.9,
        )
    decoded = corrector.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Результат: '{decoded}'")

    # 3. Проверяем с разными промптами
    print("\n" + "-" * 40)
    print("Тест 2: Разные промпты")

    prompts = [
        f"Исправь опечатки: {test_text}",
        f"Найди и исправь ошибки: {test_text}",
        f"Поправь орфографию: {test_text}",
        f"Исправь текст: {test_text}",
        f"Correction: {test_text}",
        f"Correct typos: {test_text}",
    ]

    for prompt in prompts:
        inputs = corrector.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = corrector.model.generate(
                **inputs,
                max_length=50,
                num_beams=1,
                do_sample=False,
            )
        decoded = corrector.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Промпт: '{prompt[:30]}...'")
        print(f"Результат: '{decoded}'")
        print()

    # 4. Проверяем словарную коррекцию отдельно
    print("\n" + "-" * 40)
    print("Тест 3: Только словарная коррекция (без модели)")

    # Временно отключаем модель
    corrector.use_model = False
    dict_result = corrector._correct_with_dict_and_rules(test_text, top_k=3)
    corrector.use_model = True

    print("Словарная коррекция:")
    for corr, conf in dict_result:
        print(f"  - {corr} (уверенность: {conf:.2%})")

    # 5. Проверяем словарь
    print("\n" + "-" * 40)
    print("Словарь опечаток:")
    if "привит" in corrector.typo_dict:
        print(f"  'привит' → {corrector.typo_dict['привит']}")
    if "дел" in corrector.typo_dict:
        print(f"  'дел' → {corrector.typo_dict['дел']}")

    # Проверяем похожие слова
    print("\nПохожие слова через Левенштейн:")
    for word in ["привит", "дел"]:
        similar, confidence = corrector._find_similar_word(word)
        print(f"  '{word}' → '{similar}' (уверенность: {confidence:.2%})")


if __name__ == "__main__":
    diagnose_model()
