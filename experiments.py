"""
experiments.py
-------------------
Experimental evaluation for Chord vs Pastry (per assignment):

- Measure hops for: build / insert / update / lookup / delete / node join / node leave
- Produce CSV + plots
- Print full summaries after completion
- Run a K-concurrent lookup demo (popularities) for both protocols

Important fix vs v1:
- For join/leave we separate:
    locate_hops: routing hops to locate position/owner (used for plots)
    move_hops: additional overhead for key movement (not used in plots)
    moved_keys: how many keys moved
  and we set 'hops' = locate_hops for join/leave so the plot reflects DHT routing cost.

Colab example:
  !python experiments.py --file /content/data_movies_clean.csv --nodes 16,32,64 --max_rows 50000 --records 20000 --updates 2000 --queries 5000 --deletes 2000 --joins 10 --leaves 10 --K 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from threading import Thread, Lock

import pandas as pd
import matplotlib.pyplot as plt

from data_read import load_and_preprocess_csv
from chord import ChordRing
from pastry import PastryRing


# ------------------------- utilities -------------------------
def _parse_nodes_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _uniform_node_ids(m: int, n: int) -> List[int]:
    max_id = 2**m - 1
    return [(i * max_id) // n for i in range(1, n + 1)]


def _random_start_chord(ring: ChordRing):
    return random.choice(ring.nodes) if ring.nodes else None


def _random_start_pastry(ring: PastryRing):
    return random.choice(ring.nodes) if ring.nodes else None


def _ensure_unique_node_id(existing: set, m: int) -> int:
    space = 2**m
    while True:
        cand = random.randrange(0, space)
        if cand not in existing:
            return cand


def _pick_records(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df2 = df.dropna(subset=["title"]).copy()
    if len(df2) == 0:
        raise ValueError("No valid rows with 'title' found.")
    if n >= len(df2):
        return df2.reset_index(drop=True)
    return df2.sample(n=n, random_state=seed).reset_index(drop=True)


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------- plotting -------------------------
def plot_results(df_res: pd.DataFrame, outdir: Path):
    """
    For join/leave we plot df_res['hops'], which we set to locate_hops.
    For build/update we plot df_res['hops'] like any other operation.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    for op in ["build", "insert", "update", "lookup", "delete", "join", "leave"]:
        d = df_res[df_res["operation"] == op]
        if d.empty:
            continue

        grouped = d.groupby(["protocol", "N"])["hops"].mean().reset_index()

        plt.figure()
        for protocol in grouped["protocol"].unique():
            g = grouped[grouped["protocol"] == protocol].sort_values("N")
            plt.plot(g["N"], g["hops"], marker="o", label=protocol)

        plt.xlabel("Number of nodes (N)")
        plt.ylabel("Average hops")
        if op in ("join", "leave"):
            plt.title(f"Avg locate-hops vs N — {op}")
        else:
            plt.title(f"Avg hops vs N — {op}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"{op}.png", dpi=150)
        plt.close()


# ------------------------- K-concurrent lookup demo -------------------------
def _concurrent_lookup_chord(ring: ChordRing, titles: List[str]) -> Dict[str, Any]:
    results = {}
    lock = Lock()

    def worker(t: str):
        recs, hops = ring.lookup(t, start_node=_random_start_chord(ring))
        pop = recs[0].get("popularity") if recs else None
        with lock:
            results[t] = {"popularity": pop, "hops": hops}

    threads = [Thread(target=worker, args=(t,)) for t in titles]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


def _concurrent_lookup_pastry(ring: PastryRing, titles: List[str]) -> Dict[str, Any]:
    results = {}
    lock = Lock()

    def worker(t: str):
        recs, hops = ring.lookup(t, start_node=_random_start_pastry(ring))
        pop = recs[0].get("popularity") if recs else None
        with lock:
            results[t] = {"popularity": pop, "hops": hops}

    threads = [Thread(target=worker, args=(t,)) for t in titles]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results


# ------------------------- core experiment -------------------------
def _row(protocol: str, operation: str, phase: str, N: int, hops: int,
         locate_hops: int = None, move_hops: int = None, moved_keys: int = None) -> Dict[str, Any]:
    """
    Standard output row.
    For join/leave:
      hops == locate_hops (used for plots and "DHT hops" comparison)
      plus: move_hops, moved_keys for overhead explanation
    For other ops:
      hops is the measured operation hops; locate_hops/move_hops/moved_keys remain None.
    """
    return {
        "protocol": protocol,
        "operation": operation,
        "phase": phase,
        "N": N,
        "hops": hops,
        "locate_hops": locate_hops,
        "move_hops": move_hops,
        "moved_keys": moved_keys,
    }


def run_for_N(
    df: pd.DataFrame,
    m: int,
    N: int,
    records_n: int,
    updates_n: int,   # NEW
    queries_n: int,
    deletes_n: int,
    joins_n: int,
    leaves_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Produces rows with:
      protocol, operation, phase, N,
      hops (main), locate_hops, move_hops, moved_keys
    """
    rows: List[Dict[str, Any]] = []

    chord = ChordRing(m=m, btree_size=32)
    pastry = PastryRing(m=m, leaf_size=8, btree_size=32)

    # ---------- Build initial overlay with N nodes ----------
    base_node_ids = _uniform_node_ids(m, N)
    chord_ids = set(base_node_ids)
    pastry_ids = set(base_node_ids)

    # Track build cost (average initial join locate hops per protocol)
    build_sum_c = 0
    build_sum_p = 0
    build_cnt = 0

    # Initial joins: keep them for completeness
    for nid in base_node_ids:
        # Chord join_node returns (new_node, total_hops, moved_count)
        # For initial build, we consider locate_hops ~= total_hops because moved_keys is ~0
        _, total_c, moved_c = chord.join_node(nid)
        locate_c = total_c  # initial phase OK
        move_c = 0
        rows.append(_row("Chord", "join", "initial", N, hops=locate_c, locate_hops=locate_c, move_hops=move_c, moved_keys=moved_c))
        build_sum_c += locate_c

        # Pastry join_node returns (new_node, total_hops, moved_count) where total includes locate+move in your ring
        _, total_p, moved_p = pastry.join_node(nid)
        locate_p = total_p  # initial phase OK
        move_p = 0
        rows.append(_row("Pastry", "join", "initial", N, hops=locate_p, locate_hops=locate_p, move_hops=move_p, moved_keys=moved_p))
        build_sum_p += locate_p

        build_cnt += 1

    # NEW: Add a distinct "build" query result per protocol (avg initial join locate hops)
    if build_cnt > 0:
        rows.append(_row("Chord", "build", "initial", N, hops=int(round(build_sum_c / build_cnt))))
        rows.append(_row("Pastry", "build", "initial", N, hops=int(round(build_sum_p / build_cnt))))

    # ---------- Prepare records ----------
    df_rec = _pick_records(df, records_n, seed=seed + N)
    titles = df_rec["title"].astype(str).tolist()
    records = [row.to_dict() for _, row in df_rec.iterrows()]

    # ---------- Inserts ----------
    for title, rec in zip(titles, records):
        hops_c = chord.insert_title(title, rec, start_node=_random_start_chord(chord))
        rows.append(_row("Chord", "insert", "workload", N, hops=hops_c))

        hops_p = pastry.insert_title(title, rec, start_node=_random_start_pastry(pastry))
        rows.append(_row("Pastry", "insert", "workload", N, hops=hops_p))

    # ---------- Updates (NEW) ----------
    # We measure hops as the routing hops to locate the owner (same as lookup), and then update the stored record in-place.
    if len(titles) > 0 and updates_n > 0:
        u_titles = random.sample(titles, k=min(updates_n, len(titles)))
        for t in u_titles:
            recs_c, hops_c = chord.lookup(t, start_node=_random_start_chord(chord))
            if recs_c:
                old = recs_c[0].get("popularity")
                # set a deterministic-but-different numeric value when possible
                try:
                    new_val = float(old) + 0.001
                except Exception:
                    new_val = old
                recs_c[0]["popularity"] = new_val
            rows.append(_row("Chord", "update", "workload", N, hops=hops_c))

            recs_p, hops_p = pastry.lookup(t, start_node=_random_start_pastry(pastry))
            if recs_p:
                old = recs_p[0].get("popularity")
                try:
                    new_val = float(old) + 0.001
                except Exception:
                    new_val = old
                recs_p[0]["popularity"] = new_val
            rows.append(_row("Pastry", "update", "workload", N, hops=hops_p))

    # ---------- Lookups (exact match) ----------
    if len(titles) > 0:
        q_titles = random.choices(titles, k=min(queries_n, len(titles)))
        for t in q_titles:
            _, hops_c = chord.lookup(t, start_node=_random_start_chord(chord))
            rows.append(_row("Chord", "lookup", "workload", N, hops=hops_c))

            _, hops_p = pastry.lookup(t, start_node=_random_start_pastry(pastry))
            rows.append(_row("Pastry", "lookup", "workload", N, hops=hops_p))

    # ---------- Deletes ----------
    if len(titles) > 0:
        d_titles = random.sample(titles, k=min(deletes_n, len(titles)))
        for t in d_titles:
            hops_c = chord.delete_title(t, start_node=_random_start_chord(chord))
            rows.append(_row("Chord", "delete", "workload", N, hops=hops_c))

            hops_p = pastry.delete_title(t, start_node=_random_start_pastry(pastry))
            rows.append(_row("Pastry", "delete", "workload", N, hops=hops_p))

    # ---------- Dynamic joins ----------
    # We split join into: locate_hops + move_hops
    # locate_hops: routing to locate where node would attach (or its own id)
    # move_hops: extra overhead reported by join_node implementation (key movement)
    for _ in range(joins_n):
        # Chord: estimate locate_hops by doing a find_successor on the new id from a random start node
        nid = _ensure_unique_node_id(chord_ids, m)
        chord_ids.add(nid)
        start_c = _random_start_chord(chord)
        # locate cost prior to actual join
        try:
            _, locate_c = chord.find_successor(nid, start_node=start_c)
        except Exception:
            locate_c = 0
        _, total_c, moved_c = chord.join_node(nid, start_node=start_c)
        move_c = max(0, total_c - locate_c)
        # For plots and "DHT hops", use locate_hops
        rows.append(_row("Chord", "join", "dynamic", N, hops=locate_c, locate_hops=locate_c, move_hops=move_c, moved_keys=moved_c))

        # Pastry: locate via routing to node_id from random start node
        nid2 = _ensure_unique_node_id(pastry_ids, m)
        pastry_ids.add(nid2)
        start_p = _random_start_pastry(pastry)
        try:
            _, locate_p = pastry._route(start_p, nid2)  # internal, but OK for measurement
        except Exception:
            locate_p = 0
        _, total_p, moved_p = pastry.join_node(nid2)
        move_p = max(0, total_p - locate_p)
        rows.append(_row("Pastry", "join", "dynamic", N, hops=locate_p, locate_hops=locate_p, move_hops=move_p, moved_keys=moved_p))

    # ---------- Dynamic leaves ----------
    # locate_hops: routing to find owners for moved keys is not "one routing", but leave in DHT typically contacts successor/neighbor.
    # For a clean DHT metric, we define leave locate_hops as 1 "neighbor contact" hop (conceptual) or 0 if trivial.
    # However, since your leave_node returns total cost (routing + transfers), we separate:
    # locate_hops: 1 (constant neighbor update) -> plotted
    # move_hops: rest (total - locate_hops)
    for _ in range(leaves_n):
        if len(chord.nodes) > 2:
            leave_node = random.choice(chord.nodes[1:])
            ok, total_c, moved_c = chord.leave_node(leave_node.node_id, start_node=_random_start_chord(chord))
            if ok:
                locate_c = 1  # conceptual overlay neighbor update
                move_c = max(0, total_c - locate_c)
                rows.append(_row("Chord", "leave", "dynamic", N, hops=locate_c, locate_hops=locate_c, move_hops=move_c, moved_keys=moved_c))

        if len(pastry.nodes) > 2:
            leave_node = random.choice(pastry.nodes[1:])
            ok, total_p, moved_p = pastry.leave_node(leave_node.id)
            if ok:
                locate_p = 1
                move_p = max(0, total_p - locate_p)
                rows.append(_row("Pastry", "leave", "dynamic", N, hops=locate_p, locate_hops=locate_p, move_hops=move_p, moved_keys=moved_p))

    return rows


# ------------------------- printing summaries -------------------------
def print_assignment_outputs(df: pd.DataFrame, res: pd.DataFrame, outdir: Path, K: int, seed: int):
    _print_header("DATASET SUMMARY (post-sampling)")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required = ["id", "title", "popularity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("[WARN] Missing required columns:", missing)
    else:
        print("[OK] Required columns exist:", required)

    _print_header("EXPERIMENTS SUMMARY — counts per operation")
    print(res["operation"].value_counts())

    _print_header("EXPERIMENTS SUMMARY — avg hops by (operation, protocol, N)")
    summary = (res
               .groupby(["operation", "protocol", "N"])["hops"]
               .agg(["count", "mean", "median", "std", "min", "max"])
               .reset_index()
               .sort_values(["operation", "N", "protocol"]))
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 160):
        print(summary.to_string(index=False))

    _print_header("JOIN/LEAVE OVERHEAD — moved_keys and move_hops (NOT used in plots)")
    jl = res[res["operation"].isin(["join", "leave"])]
    if jl.empty:
        print("[WARN] No join/leave rows found.")
    else:
        overhead = (jl.groupby(["operation", "phase", "protocol", "N"])
                    .agg(
                        events=("hops", "count"),
                        avg_locate_hops=("locate_hops", "mean"),
                        avg_move_hops=("move_hops", "mean"),
                        avg_moved_keys=("moved_keys", "mean"),
                        max_moved_keys=("moved_keys", "max"),
                    )
                    .reset_index()
                    .sort_values(["operation", "phase", "N", "protocol"]))
        with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 160):
            print(overhead.to_string(index=False))

    _print_header("OUTPUT FILES")
    print("CSV:", str(outdir / "results.csv"))
    for op in ["build", "insert", "update", "lookup", "delete", "join", "leave"]:
        p = outdir / f"{op}.png"
        if p.exists():
            print("Plot:", str(p))
        else:
            print("[WARN] Missing plot:", str(p))

    # K-concurrent lookups demo (popularities)
    _print_header(f"K-CONCURRENT LOOKUP DEMO (K={K}) — returns popularity + hops")
    random.seed(seed)

    # Build demo rings
    m = 40
    demo_N = 32
    chord = ChordRing(m=m, btree_size=32)
    pastry = PastryRing(m=m, leaf_size=8, btree_size=32)
    for nid in _uniform_node_ids(m, demo_N):
        chord.join_node(nid)
        pastry.join_node(nid)

    # Insert demo dataset (so lookups definitely exist)
    demo_df = _pick_records(df, n=min(10000, len(df)), seed=seed + 999)  # insert 10k for demo
    inserted_titles = []
    for _, row in demo_df.iterrows():
        title = str(row["title"])
        rec = row.to_dict()
        chord.insert_title(title, rec, start_node=_random_start_chord(chord))
        pastry.insert_title(title, rec, start_node=_random_start_pastry(pastry))
        inserted_titles.append(title)

    # Pick K titles from those inserted (guaranteed present)
    K = min(K, len(inserted_titles))
    demo_titles = random.sample(inserted_titles, k=K)

    chord_results = _concurrent_lookup_chord(chord, demo_titles)
    pastry_results = _concurrent_lookup_pastry(pastry, demo_titles)

    print("\nChord results:")
    for t in demo_titles:
        r = chord_results.get(t, {})
        print(f'  "{t}" -> popularity={r.get("popularity")} hops={r.get("hops")}')

    print("\nPastry results:")
    for t in demo_titles:
        r = pastry_results.get(t, {})
        print(f'  "{t}" -> popularity={r.get("popularity")} hops={r.get("hops")}')


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to data_movies_clean.csv")
    ap.add_argument("--max_rows", type=int, default=50_000, help="Max CSV rows to use (default 50k).")
    ap.add_argument("--m", type=int, default=40, help="Keyspace bits (default 40).")
    ap.add_argument("--nodes", type=str, default="16,32,64", help="Comma-separated node counts, e.g. 16,32,64")
    ap.add_argument("--records", type=int, default=20_000, help="Number of records to insert per N.")
    ap.add_argument("--updates", type=int, default=2_000, help="Number of updates per N. (NEW)")
    ap.add_argument("--queries", type=int, default=5_000, help="Number of lookups per N.")
    ap.add_argument("--deletes", type=int, default=2_000, help="Number of deletes per N.")
    ap.add_argument("--joins", type=int, default=10, help="Number of dynamic joins per N.")
    ap.add_argument("--leaves", type=int, default=10, help="Number of dynamic leaves per N.")
    ap.add_argument("--K", type=int, default=10, help="K titles for concurrent lookup demo.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for CSV and plots.")
    args = ap.parse_args()

    random.seed(args.seed)

    df = load_and_preprocess_csv(args.file, max_rows=args.max_rows, seed=args.seed)

    required_cols = ["id", "title", "popularity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    nodes_list = _parse_nodes_list(args.nodes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _print_header("RUN CONFIG")
    print("file:", args.file)
    print("max_rows:", args.max_rows)
    print("m:", args.m)
    print("nodes:", nodes_list)
    print("records:", args.records, "updates:", args.updates, "queries:", args.queries, "deletes:", args.deletes, "joins:", args.joins, "leaves:", args.leaves)
    print("K (concurrent demo):", args.K)
    print("seed:", args.seed)
    print("outdir:", str(outdir))

    all_rows: List[Dict[str, Any]] = []
    for N in nodes_list:
        _print_header(f"RUNNING N={N}")
        all_rows.extend(
            run_for_N(
                df=df,
                m=args.m,
                N=N,
                records_n=min(args.records, len(df)),
                updates_n=args.updates,
                queries_n=args.queries,
                deletes_n=args.deletes,
                joins_n=args.joins,
                leaves_n=args.leaves,
                seed=args.seed,
            )
        )

    res = pd.DataFrame(all_rows)
    res.to_csv(outdir / "results.csv", index=True)

    plot_results(res, outdir)

    print(f"\n[OK] Wrote {len(res)} measurements to {outdir / 'results.csv'}")
    print(f"[OK] Plots saved under: {outdir}")

    print_assignment_outputs(df=df, res=res, outdir=outdir, K=args.K, seed=args.seed)


if __name__ == "__main__":
    main()
