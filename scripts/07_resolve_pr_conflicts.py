from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFLICT_PATHS = [
    "training/train.py",
    "tests/test_training_train.py",
]


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=capture_output)


def current_branch() -> str:
    out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)
    return out.stdout.strip()


def conflicted_paths() -> set[str]:
    out = run(["git", "diff", "--name-only", "--diff-filter=U"], capture_output=True)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def resolve_known_conflicts(paths: list[str]) -> bool:
    unresolved = conflicted_paths()
    if not unresolved:
        return False

    for path in paths:
        if path in unresolved:
            run(["git", "checkout", "--ours", path])
            run(["git", "add", path])

    remaining = conflicted_paths()
    if remaining:
        print("[!] Остались неразрешенные конфликты:")
        for item in sorted(remaining):
            print(f"    - {item}")
        return False

    return True


def has_staged_changes() -> bool:
    out = run(["git", "diff", "--cached", "--name-only"], capture_output=True)
    return bool(out.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Синхронизирует текущую ветку с origin/main, автоматически разрешает типовые конфликты "
            "в training/train.py и tests/test_training_train.py, и запускает проверки перед PR."
        )
    )
    parser.add_argument("--base", default="origin/main", help="Базовая ветка для merge (по умолчанию origin/main)")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=DEFAULT_CONFLICT_PATHS,
        help="Пути, где конфликт разрешается автоматически через --ours",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Не запускать pytest/проверку конфликт-маркеров",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    print(f"[i] Repo: {repo_root}")

    branch = current_branch()
    if branch in {"main", "master"}:
        print("[!] Запустите скрипт из feature-ветки, а не из main/master")
        return 2

    run(["git", "fetch", "origin"])

    merge_result = run(["git", "merge", "--no-edit", args.base], check=False)
    if merge_result.returncode != 0:
        print("[i] Обнаружены конфликты, пробую авто-разрешение известных файлов...")
        if not resolve_known_conflicts(args.paths):
            print("[!] Авто-разрешение не завершено. Исправьте оставшиеся конфликты вручную и повторите.")
            return 2

        if has_staged_changes():
            run(["git", "commit", "-m", f"Merge {args.base} with auto conflict resolution"])

    if not args.no_tests:
        run(["rg", "-n", "^(<<<<<<<|=======|>>>>>>>)", "-S"], check=False)
        run(["pytest", "-q", "tests/test_training_train.py", "tests/test_training_utils.py", "tests/test_doctor.py"])

    print("[✓] Ветка синхронизирована и проверена. Можно открывать/обновлять PR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
