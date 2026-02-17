from dataset.segmenter import indexed_segments, split_to_segments


def test_split_to_segments_by_regex_and_max_chars() -> None:
    text = "Коротко. Очень длинная фраза для теста разбивки"

    segments = split_to_segments(text, split_regex=r"[.]", max_chars=12)

    assert segments == ["Коротко", "Очень", "длинная", "фраза для", "теста", "разбивки"]


def test_indexed_segments_uses_prefix_and_zero_padding() -> None:
    items = indexed_segments(["a", "b"], prefix="proj")

    assert items == [("proj_00001", "a"), ("proj_00002", "b")]
