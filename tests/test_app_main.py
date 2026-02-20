from __future__ import annotations

from app.main import build_parser


def test_train_supports_check_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--project", "demo", "--check"])
    assert args.dry_run is True


def test_train_supports_dry_run_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--project", "demo", "--dry-run"])
    assert args.dry_run is True
