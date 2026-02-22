#!/usr/bin/env python
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = "ddn_vladimir"
EPOCH_RE = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
SPEED_RE = re.compile(r"([\d.]+\s*(?:it/s|steps/s|s/it))", re.IGNORECASE)
ETA_RE = re.compile(r"ETA\s*[:=]\s*([^,\]|]+)", re.IGNORECASE)


class WinTrainGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Windows Training GUI")
        self.root.geometry("1120x760")

        self.training_proc: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self._tail_pos = 0
        self._tail_stop = threading.Event()
        self._last_epoch: int | None = None
        self._last_speed = "-"
        self._last_eta = "-"

        self.project_var = tk.StringVar(value=DEFAULT_PROJECT)
        self.mode_var = tk.StringVar(value="scratch")
        self.add_epochs_var = tk.StringVar(value="500")
        self.target_epochs_var = tk.StringVar(value="500")
        self.prefer_best_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle")

        self.ckpt_items: list[dict[str, object]] = []

        self._build_ui()
        self._schedule_target_update()
        self.refresh_ckpts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        row_project = ttk.Frame(top)
        row_project.pack(fill=tk.X)
        ttk.Label(row_project, text="Project:").pack(side=tk.LEFT)
        project_entry = ttk.Entry(row_project, textvariable=self.project_var, width=38)
        project_entry.pack(side=tk.LEFT, padx=(8, 12))
        project_entry.bind("<Return>", lambda _e: self.refresh_ckpts())

        ttk.Button(row_project, text="Refresh", command=self.refresh_ckpts).pack(side=tk.LEFT)
        ttk.Button(row_project, text="Browse ckpt", command=self.browse_ckpt).pack(side=tk.LEFT, padx=6)
        ttk.Button(row_project, text="Open models folder", command=self.open_models_folder).pack(side=tk.LEFT)
        ttk.Button(row_project, text="Open runs folder", command=self.open_runs_folder).pack(side=tk.LEFT, padx=6)

        list_frame = ttk.LabelFrame(top, text="Checkpoints", padding=8)
        list_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 8))

        columns = ("path", "epoch", "global_step")
        self.ckpt_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)
        self.ckpt_tree.heading("path", text="Path")
        self.ckpt_tree.heading("epoch", text="Epoch")
        self.ckpt_tree.heading("global_step", text="Global step")
        self.ckpt_tree.column("path", width=740, anchor=tk.W)
        self.ckpt_tree.column("epoch", width=120, anchor=tk.CENTER)
        self.ckpt_tree.column("global_step", width=140, anchor=tk.CENTER)
        self.ckpt_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ckpt_tree.bind("<<TreeviewSelect>>", lambda _e: self.update_target_epochs())

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.ckpt_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ckpt_tree.configure(yscrollcommand=scrollbar.set)

        mode_frame = ttk.LabelFrame(top, text="Training", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Radiobutton(mode_frame, text="Scratch", variable=self.mode_var, value="scratch", command=self.update_target_epochs).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Resume", variable=self.mode_var, value="resume", command=self.update_target_epochs).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(mode_frame, text="Prefer best.ckpt if exists", variable=self.prefer_best_var, command=self.update_target_epochs).grid(row=0, column=2, sticky="w", padx=(18, 0))

        ttk.Label(mode_frame, text="Add epochs:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        add_entry = ttk.Entry(mode_frame, textvariable=self.add_epochs_var, width=12)
        add_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))
        add_entry.bind("<KeyRelease>", lambda _e: self.update_target_epochs())

        ttk.Label(mode_frame, text="Target max_epochs:").grid(row=1, column=2, sticky="e", pady=(8, 0), padx=(18, 6))
        ttk.Entry(mode_frame, textvariable=self.target_epochs_var, width=14, state="readonly").grid(row=1, column=3, sticky="w", pady=(8, 0))

        btn_row = ttk.Frame(top)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Start", command=self.start_training).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Stop", command=self.stop_training).pack(side=tk.LEFT, padx=6)

        log_frame = ttk.LabelFrame(top, text="Logs", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = ScrolledText(log_frame, wrap=tk.NONE, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(top, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(fill=tk.X, pady=(8, 0))

    def _project(self) -> str:
        p = self.project_var.get().strip()
        return p or DEFAULT_PROJECT

    def _models_dir(self) -> Path:
        return REPO_ROOT / "data" / "models" / self._project()

    def _runs_dir(self) -> Path:
        return REPO_ROOT / "data" / "projects" / self._project() / "runs"

    def _helper_python(self) -> Path:
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"

    def _refresh_tree(self) -> None:
        for item in self.ckpt_tree.get_children():
            self.ckpt_tree.delete(item)
        for idx, item in enumerate(self.ckpt_items):
            epoch = item.get("epoch")
            step = item.get("global_step")
            epoch_text = "-" if epoch is None else str(epoch)
            step_text = "-" if step is None else str(step)
            self.ckpt_tree.insert("", tk.END, iid=str(idx), values=(item["path"], epoch_text, step_text))

    def refresh_ckpts(self) -> None:
        project = self._project()
        models_dir = REPO_ROOT / "data" / "models" / project
        runs_dir = REPO_ROOT / "data" / "projects" / project / "runs"
        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[Path] = sorted(models_dir.glob("*.ckpt"))
        for special in (runs_dir / "best.ckpt", runs_dir / "last.ckpt"):
            if special.exists() and special not in candidates:
                candidates.append(special)

        helper = REPO_ROOT / "tools" / "ckpt_info.py"
        py_exec = self._helper_python()
        if not py_exec.exists():
            py_exec = Path(sys.executable)

        items: list[dict[str, object]] = []
        for ckpt in candidates:
            info: dict[str, object] = {
                "path": str(ckpt),
                "epoch": None,
                "global_step": None,
            }
            try:
                proc = subprocess.run(
                    [str(py_exec), str(helper), str(ckpt)],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                line = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else ""
                if line:
                    parsed = json.loads(line)
                    info.update(parsed)
                elif proc.stderr:
                    info["error"] = proc.stderr.strip()
            except Exception as exc:  # noqa: BLE001
                info["error"] = f"{exc.__class__.__name__}: {exc}"
            items.append(info)

        items.sort(key=lambda x: (Path(str(x["path"])).name.lower(), str(x["path"])))
        self.ckpt_items = items
        self._refresh_tree()
        self._auto_select_best()
        self.update_target_epochs()
        self.status_var.set(f"Found {len(items)} checkpoint(s) for project '{project}'.")

    def _auto_select_best(self) -> None:
        if not self.prefer_best_var.get():
            return
        for idx, item in enumerate(self.ckpt_items):
            if str(item.get("path", "")).lower().endswith("best.ckpt"):
                self.ckpt_tree.selection_set(str(idx))
                self.ckpt_tree.focus(str(idx))
                return

    def get_selected_ckpt(self) -> dict[str, object] | None:
        selected = self.ckpt_tree.selection()
        if not selected:
            return None
        idx = int(selected[0])
        if 0 <= idx < len(self.ckpt_items):
            return self.ckpt_items[idx]
        return None

    def browse_ckpt(self) -> None:
        initial = self._models_dir()
        initial.mkdir(parents=True, exist_ok=True)
        chosen = filedialog.askopenfilename(
            title="Select checkpoint",
            initialdir=str(initial),
            filetypes=[("Checkpoint", "*.ckpt"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        existing = {str(item.get("path")): idx for idx, item in enumerate(self.ckpt_items)}
        if str(path) not in existing:
            self.ckpt_items.append({"path": str(path), "epoch": None, "global_step": None})
            self._refresh_tree()
            idx = len(self.ckpt_items) - 1
        else:
            idx = existing[str(path)]
        self.ckpt_tree.selection_set(str(idx))
        self.ckpt_tree.focus(str(idx))
        self.update_target_epochs()

    def open_models_folder(self) -> None:
        folder = self._models_dir()
        folder.mkdir(parents=True, exist_ok=True)
        self._open_in_explorer(folder)

    def open_runs_folder(self) -> None:
        folder = self._runs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        self._open_in_explorer(folder)

    def _open_in_explorer(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("Info", f"Open manually: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder", f"Failed to open folder:\n{exc}")

    def _current_add_epochs(self) -> int:
        text = self.add_epochs_var.get().strip()
        try:
            value = int(text)
            return max(0, value)
        except ValueError:
            return 0

    def _resume_epoch(self) -> int:
        if self.mode_var.get() != "resume":
            return 0
        selected = self.get_selected_ckpt()
        if not selected:
            if self.prefer_best_var.get():
                for item in self.ckpt_items:
                    if str(item.get("path", "")).lower().endswith("best.ckpt"):
                        epoch = item.get("epoch")
                        return int(epoch) if isinstance(epoch, int) else 0
            return 0
        epoch = selected.get("epoch")
        return int(epoch) if isinstance(epoch, int) else 0

    def update_target_epochs(self) -> None:
        target = self._resume_epoch() + self._current_add_epochs()
        self.target_epochs_var.set(str(target))

    def _schedule_target_update(self) -> None:
        self.update_target_epochs()
        self.root.after(1000, self._schedule_target_update)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def start_training(self) -> None:
        if self.training_proc and self.training_proc.poll() is None:
            messagebox.showwarning("Training", "Training process is already running.")
            return

        project = self._project()
        target_epochs = self.target_epochs_var.get().strip()
        if not target_epochs.isdigit() or int(target_epochs) <= 0:
            messagebox.showerror("Training", "Target max_epochs must be a positive integer.")
            return

        py_exec = self._helper_python()
        if not py_exec.exists():
            py_exec = Path(sys.executable)

        cmd = [
            str(py_exec),
            "-m",
            "app.main",
            "train",
            "--project",
            project,
            "--epochs",
            target_epochs,
        ]

        if self.mode_var.get() == "resume":
            selected = self.get_selected_ckpt()
            if selected is None and self.prefer_best_var.get():
                selected = next(
                    (i for i in self.ckpt_items if str(i.get("path", "")).lower().endswith("best.ckpt")),
                    None,
                )
            if not selected:
                messagebox.showerror("Training", "Resume mode selected but no checkpoint chosen.")
                return
            cmd.extend(["--resume-ckpt", str(selected["path"])])

        runs_dir = self._runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = runs_dir / "train_console.log"
        self.log_path.touch(exist_ok=True)
        self._tail_pos = self.log_path.stat().st_size
        self._tail_stop.clear()
        self._last_epoch = None
        self._last_speed = "-"
        self._last_eta = "-"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            log_fp = open(self.log_path, "a", encoding="utf-8")
            self.training_proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                creationflags=creationflags,
            )
            log_fp.close()
        except Exception as exc:  # noqa: BLE001
            self.training_proc = None
            messagebox.showerror("Training", f"Failed to start training:\n{exc}")
            return

        self.status_var.set(f"Training started (pid={self.training_proc.pid}).")
        self._append_log(f"\n[GUI] started: {' '.join(cmd)}\n")
        self._start_tail_thread()
        self.root.after(500, self._poll_process)

    def _start_tail_thread(self) -> None:
        th = threading.Thread(target=self._tail_worker, daemon=True)
        th.start()

    def _tail_worker(self) -> None:
        while not self._tail_stop.is_set():
            if not self.log_path or not self.log_path.exists():
                self._tail_stop.wait(0.5)
                continue
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="replace") as fp:
                    fp.seek(self._tail_pos)
                    data = fp.read()
                    self._tail_pos = fp.tell()
                if data:
                    self.root.after(0, self._process_new_log_chunk, data)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, self.status_var.set, f"Log read error: {exc}")
            self._tail_stop.wait(0.5)

    def _process_new_log_chunk(self, chunk: str) -> None:
        self._append_log(chunk)
        for line in chunk.splitlines():
            m = EPOCH_RE.search(line)
            if m:
                try:
                    self._last_epoch = int(m.group(1))
                except ValueError:
                    pass
            sm = SPEED_RE.search(line)
            if sm:
                self._last_speed = sm.group(1)
            em = ETA_RE.search(line)
            if em:
                self._last_eta = em.group(1).strip()
        epoch_txt = "?" if self._last_epoch is None else str(self._last_epoch)
        self.status_var.set(f"Epoch {epoch_txt} / speed {self._last_speed} / ETA {self._last_eta}")

    def _poll_process(self) -> None:
        proc = self.training_proc
        if not proc:
            return
        code = proc.poll()
        if code is None:
            self.root.after(1000, self._poll_process)
            return
        self._tail_stop.set()
        self.status_var.set(f"Training finished with exit code {code}.")
        self._append_log(f"\n[GUI] training finished with exit code {code}\n")

    def stop_training(self) -> None:
        proc = self.training_proc
        if not proc or proc.poll() is not None:
            self.status_var.set("No active training process.")
            return

        try:
            if os.name == "nt":
                proc.send_signal(subprocess.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            try:
                proc.terminate()
            except Exception as exc:  # noqa: BLE001
                self.status_var.set(f"Failed to stop process: {exc}")
                return

        self.status_var.set("Stop signal sent.")
        self._append_log("\n[GUI] stop requested\n")

    def _on_close(self) -> None:
        self._tail_stop.set()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    WinTrainGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
