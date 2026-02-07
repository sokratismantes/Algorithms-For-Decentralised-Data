# ui_helpers.py
"""
Small UI helper utilities to keep gui_dht.py short.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tkinter as tk
from tkinter import ttk

from experiments import RunResult


SUMMARY_COLS = ("metric", "n", "avg", "median", "p95", "min", "max")
LOOKUP_COLS = ("title", "popularity", "hops")


def make_tree(parent: ttk.Frame, cols: Tuple[str, ...], widths: Dict[str, int], height: int) -> ttk.Treeview:
    tree = ttk.Treeview(parent, columns=cols, show="headings", height=height)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=widths.get(c, 100), anchor="center")
    return tree


def clear_tree(tree: ttk.Treeview) -> None:
    for item in tree.get_children():
        tree.delete(item)


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and not x.is_integer():
        return f"{x:.2f}"
    return str(int(x)) if isinstance(x, float) else str(x)


def build_result_tab(parent: ttk.Frame) -> Dict[str, ttk.Treeview]:
    """
    Builds the exact same layout as before:
    - Summary table (metric,n,avg,median,p95,min,max)
    - Lookups table (title,popularity,hops)
    Returns dict with keys: 'summary', 'lookup'
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)
    parent.rowconfigure(3, weight=1)

    ttk.Label(parent, text="Summary (routing hops & moved records):").grid(
        row=0, column=0, sticky="w", pady=(10, 4), padx=10
    )

    summary_widths = {"metric": 260, "n": 90, "avg": 90, "median": 90, "p95": 90, "min": 90, "max": 90}
    summary = make_tree(parent, SUMMARY_COLS, summary_widths, height=9)
    summary.column("metric", anchor="w")
    summary.grid(row=1, column=0, sticky="nsew", padx=10)

    ttk.Label(parent, text="Lookups (title → popularity, hops):").grid(
        row=2, column=0, sticky="w", pady=(14, 4), padx=10
    )

    lookup_widths = {"title": 680, "popularity": 140, "hops": 140}
    lookup = make_tree(parent, LOOKUP_COLS, lookup_widths, height=10)
    lookup.column("title", anchor="w")
    lookup.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))

    return {"summary": summary, "lookup": lookup}


def render_result(views: Dict[str, ttk.Treeview], res: RunResult) -> None:
    summary = views["summary"]
    lookup = views["lookup"]

    for metric, st in res.summary_rows:
        summary.insert(
            "",
            tk.END,
            values=(
                metric,
                fmt(st.get("n")),
                fmt(st.get("avg")),
                fmt(st.get("median")),
                fmt(st.get("p95")),
                fmt(st.get("min")),
                fmt(st.get("max")),
            ),
        )

    for row in res.lookup_rows:
        pop = "" if row.popularity is None else str(row.popularity)
        lookup.insert("", tk.END, values=(row.title, pop, row.hops))