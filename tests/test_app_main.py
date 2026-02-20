from __future__ import annotations

from app.main import DEFAULT_PROJECT_NAME, _find_default_text_file, build_parser


def test_train_supports_check_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--project", "demo", "--check"])
    assert args.dry_run is True


def test_train_supports_dry_run_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--project", "demo", "--dry-run"])
    assert args.dry_run is True


def test_project_defaults_to_repo_name() -> None:
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.project == DEFAULT_PROJECT_NAME


def test_prepare_finds_text_file_in_priority_order(tmp_path) -> None:
    (tmp_path / "data" / "input_texts").mkdir(parents=True)
    fallback = tmp_path / "dataset" / "z_source.txt"
    fallback.parent.mkdir(parents=True)
    preferred = tmp_path / "data" / "input_texts" / "source.txt"
    fallback.write_text("fallback", encoding="utf-8")
    preferred.write_text("preferred", encoding="utf-8")

    found = _find_default_text_file(tmp_path)

    assert found == preferred
