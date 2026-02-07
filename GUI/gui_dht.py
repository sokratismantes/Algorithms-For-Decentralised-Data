"""
Slim GUI runner for Chord + Pastry experiments.
- Loads CSV
- Runs experiments in a background thread
- Renders results into the same tables as the original UI
"""

from __future__ import annotations

from pathlib import Path
import threading
import queue
from typing import Any, Dict, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

from data_read import load_and_preprocess_csv

from experiments import RunConfig, RunResult, run_chord, run_pastry
from ui_helpers import build_result_tab, clear_tree, render_result


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DHT Runner (Chord vs Pastry)")
        self.geometry("1100x720")

        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.views: Dict[str, Dict[str, ttk.Treeview]] = {}

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Nodes (N):").grid(row=0, column=0, sticky="w")
        self.nodes_var = tk.StringVar(value="32")
        ttk.Entry(top, textvariable=self.nodes_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(top, text="Lookups (K):").grid(row=0, column=2, sticky="w")
        self.k_var = tk.StringVar(value="3")
        ttk.Entry(top, textvariable=self.k_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(top, text="Seed:").grid(row=0, column=4, sticky="w")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(top, textvariable=self.seed_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 18))

        ttk.Label(top, text="Dataset (CSV) path:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.path_var = tk.StringVar(value=str((Path(".").resolve() / "data_movies_clean.csv")))
        ttk.Entry(top, textvariable=self.path_var, width=90).grid(
            row=1, column=1, columnspan=5, sticky="we", pady=(8, 0)
        )

        titles_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        titles_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(
            titles_frame,
            text="Optional titles (one per line). If fewer than K, the rest are random:",
        ).pack(side=tk.TOP, anchor="w", pady=(10, 4))

        self.titles_text = tk.Text(titles_frame, height=6)
        self.titles_text.pack(side=tk.TOP, fill=tk.X)

        btn_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        self.run_btn = ttk.Button(btn_frame, text="Run Chord + Pastry", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(btn_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        # progress bars 
        self.pb_all = ttk.Progressbar(btn_frame, mode="determinate", length=220, maximum=100)
        self.pb_all.pack(side=tk.LEFT, padx=(12, 6))
        self.pb_stage = ttk.Progressbar(btn_frame, mode="determinate", length=220, maximum=100)
        self.pb_stage.pack(side=tk.LEFT)

        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tab_chord = ttk.Frame(self.nb)
        tab_pastry = ttk.Frame(self.nb)
        self.nb.add(tab_chord, text="Chord")
        self.nb.add(tab_pastry, text="Pastry")

        self.views["chord"] = build_result_tab(tab_chord)
        self.views["pastry"] = build_result_tab(tab_pastry)

        log_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)

        ttk.Label(log_frame, text="Log:").pack(side=tk.TOP, anchor="w")
        self.log_text = tk.Text(log_frame, height=7)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _append_log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _clear_results(self):
        for v in self.views.values():
            clear_tree(v["summary"])
            clear_tree(v["lookup"])

    def on_run(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Running", "A run is already in progress.")
            return

        try:
            n = int(self.nodes_var.get().strip())
            k = int(self.k_var.get().strip())
            seed = int(self.seed_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", "Nodes, K, and Seed must be integers.")
            return

        if n < 2:
            messagebox.showerror("Invalid input", "Nodes must be >= 2.")
            return
        if k < 1:
            messagebox.showerror("Invalid input", "K must be >= 1.")
            return

        csv_path = Path(self.path_var.get().strip())
        if not csv_path.exists():
            messagebox.showerror("Missing file", f"CSV not found:\n{csv_path}")
            return

        titles = [t.strip() for t in self.titles_text.get("1.0", tk.END).splitlines() if t.strip()]
        cfg = RunConfig(num_nodes=n, k_lookups=k, titles=titles, seed=seed)

        self._clear_results()
        self.log_text.delete("1.0", tk.END)
        self.status_var.set("Loading dataset...")
        self.run_btn.config(state=tk.DISABLED)

        self.pb_all["value"] = 0
        self.pb_stage["value"] = 0

        self._worker = threading.Thread(target=self._run_worker, args=(csv_path, cfg), daemon=True)
        self._worker.start()

    def _run_worker(self, csv_path: Path, cfg: RunConfig):
        def log(msg: str):
            self._queue.put(("log", msg))

        def progress(stage: str, cur: int, total: int, overall_pct: Optional[float] = None):
            self._queue.put(("progress", (stage, cur, total, overall_pct)))

        try:
            progress("Loading dataset", 0, 1, 0)
            df = load_and_preprocess_csv(str(csv_path), max_rows=cfg.max_rows, seed=1)
            log(f"[OK] Loaded {len(df)} rows from {csv_path}")
            progress("Loading dataset", 1, 1, 5)

            self._queue.put(("status", "Running Chord..."))
            chord_res = run_chord(df, cfg, log_cb=log, progress_cb=progress)
            self._queue.put(("result", chord_res))

            self._queue.put(("status", "Running Pastry..."))
            pastry_res = run_pastry(df, cfg, log_cb=log, progress_cb=progress)
            self._queue.put(("result", pastry_res))

            self._queue.put(("status", "Done."))
            self._queue.put(("progress", ("Done", 1, 1, 100)))
            self._queue.put(("done", None))

        except Exception as e:
            self._queue.put(("error", str(e)))
            self._queue.put(("done", None))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()

                if kind == "log":
                    self._append_log(payload)

                elif kind == "status":
                    self.status_var.set(payload)

                elif kind == "progress":
                    stage, cur, total, overall = payload
                    self.status_var.set(stage)
                    self.pb_stage["value"] = 0 if total == 0 else (cur * 100 / total)
                    if overall is not None:
                        self.pb_all["value"] = float(overall)

                elif kind == "result":
                    assert isinstance(payload, RunResult)
                    if payload.name.lower() == "chord":
                        render_result(self.views["chord"], payload)
                    else:
                        render_result(self.views["pastry"], payload)

                elif kind == "error":
                    messagebox.showerror("Run failed", payload)
                    self._append_log("[ERROR] " + payload)

                elif kind == "done":
                    self.run_btn.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
