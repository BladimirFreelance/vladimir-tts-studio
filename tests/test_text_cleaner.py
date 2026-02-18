from dataset.text_cleaner import normalize_text, remove_garbage_lines


def test_normalize_text_applies_replacements_percent_and_abbreviations() -> None:
    raw = "\ufeffПривет,  50 % и т.д."
    cleaned = normalize_text(raw, replacements={"Привет": "Здравствуйте"})

    assert cleaned == "Здравствуйте, 50 процентов и и так далее"


def test_normalize_text_can_skip_abbreviation_expansion() -> None:
    raw = "г. Москва"

    cleaned = normalize_text(raw, replacements={}, expand_abbreviations=False)

    assert cleaned == "г. Москва"


def test_remove_garbage_lines_filters_short_and_broken_lines() -> None:
    lines = ["ok", "   длинная строка   ", "валидно", "битая\ufffdстрока"]

    result = remove_garbage_lines(lines, min_chars=6)

    assert result == ["длинная строка", "валидно"]
