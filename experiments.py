# experiments.py
"""
Experiment runner for Chord vs Pastry, kept compact and readable.

Returns:
- summary_rows: list[(metric_name, stats_dict)]
- lookup_rows : list[LookupRow(title, popularity, hops)]
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from chord import ChordRing, chord_hash
from pastry import PastryRing


# ---------- stats ----------
def _p95(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return float(s[int(round(0.95 * (len(s) - 1)))])


def summarize(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "avg": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "avg": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p95": _p95(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


# ---------- types ----------
@dataclass
class RunConfig:
    num_nodes: int
    k_lookups: int
    titles: List[str]
    seed: int = 42
    m_bits: int = 40
    updates_n: int = 2000
    deletes_n: int = 2000
    joins_n: int = 10
    leaves_n: int = 10
    max_rows: int = 946_460


@dataclass
class LookupRow:
    title: str
    popularity: Optional[float]
    hops: int


@dataclass
class RunResult:
    name: str  # "Chord" or "Pastry"
    summary_rows: List[Tuple[str, Dict[str, Any]]]
    lookup_rows: List[LookupRow]


# ---------- helpers ----------
def choose_titles(all_titles: List[str], cfg: RunConfig) -> List[str]:
    """Use user titles first (if exist), else fill randomly until K."""
    pool = all_titles
    s = set(pool)
    picked: List[str] = []
    for t in cfg.titles:
        t = t.strip()
        if t:
            picked.append(t if t in s else random.choice(pool))
    while len(picked) < cfg.k_lookups:
        picked.append(random.choice(pool))
    return picked[: cfg.k_lookups]


def _mk_logger(log_cb):
    def log(msg: str):
        if log_cb:
            log_cb(msg)
    return log


def _mk_progress(progress_cb):
    def prog(stage: str, cur: int, total: int, overall: Optional[float] = None):
        if progress_cb:
            progress_cb(stage, cur, total, overall)
    return prog


def _overall_plan():
    # (label, span in percent). Total doesn't have to be 100 exactly; we clamp.
    return [
        ("initial_join", 8),
        ("insert", 33),
        ("dyn_join", 7),
        ("dyn_leave", 7),
        ("update", 10),
        ("delete", 10),
        ("lookup", 15),
    ]


def _overall_value(stage_key: str, ratio: float) -> float:
    ratio = max(0.0, min(1.0, ratio))
    plan = _overall_plan()
    base = 5.0  # GUI uses 0..5% for CSV load
    acc = base
    for k, span in plan:
        if k == stage_key:
            return max(0.0, min(100.0, acc + span * ratio))
        acc += span
    return max(0.0, min(100.0, acc))


# ---------- core pipeline (shared) ----------
@dataclass
class Adapter:
    name: str
    space: int
    initial_join_label: str

    join_one: Callable[[int], Tuple[int, int]]              # (hops, moved)
    insert_one: Callable[[Dict[str, Any]], int]             # hops
    dyn_join_one: Callable[[int], Tuple[int, int]]          # (hops, moved)
    dyn_leave_one: Callable[[], Tuple[bool, int, int]]      # (ok, hops, moved)
    update_one: Callable[[str], int]                        # hops
    delete_one: Callable[[str], int]                        # hops
    lookup_one: Callable[[str], Tuple[Optional[float], int]]  # (popularity, hops)


def _run(df, cfg: RunConfig, ad: Adapter, log_cb=None, progress_cb=None) -> RunResult:
    log = _mk_logger(log_cb)
    prog = _mk_progress(progress_cb)

    random.seed(cfg.seed)

    # node IDs
    node_ids = random.sample(range(ad.space), k=cfg.num_nodes)

    # metrics
    initial_join_hops: List[int] = []
    initial_join_moved: List[int] = []
    insert_hops: List[int] = []
    join_hops: List[int] = []
    join_moved: List[int] = []
    leave_hops: List[int] = []
    leave_moved: List[int] = []
    update_hops: List[int] = []
    delete_hops: List[int] = []
    lookup_hops: List[int] = []
    lookup_rows: List[LookupRow] = []

    # 1) initial joins
    prog(f"{ad.name}: Joining initial nodes", 0, len(node_ids), _overall_value("initial_join", 0))
    for i, nid in enumerate(node_ids, start=1):
        hops, moved = ad.join_one(nid)
        initial_join_hops.append(int(hops))
        initial_join_moved.append(int(moved))
        if i == len(node_ids) or i % max(1, len(node_ids)//20) == 0:
            prog(f"{ad.name}: Joining initial nodes", i, len(node_ids), _overall_value("initial_join", i/len(node_ids)))
    log(f"[{ad.name}] Joined initial nodes")

    # 2) inserts
    total_rows = len(df)
    prog(f"{ad.name}: Inserting dataset", 0, total_rows, _overall_value("insert", 0))
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        insert_hops.append(int(ad.insert_one(row.to_dict())))
        if i == total_rows or i % 2000 == 0:
            prog(f"{ad.name}: Inserting dataset", i, total_rows, _overall_value("insert", i/total_rows))
    log(f"[{ad.name}] Inserted dataset")

    # 3) dynamic joins
    prog(f"{ad.name}: Dynamic joins", 0, cfg.joins_n, _overall_value("dyn_join", 0))
    for i in range(1, cfg.joins_n + 1):
        nid = random.randrange(0, ad.space)
        hops, moved = ad.dyn_join_one(nid)
        join_hops.append(int(hops))
        join_moved.append(int(moved))
        prog(f"{ad.name}: Dynamic joins", i, cfg.joins_n, _overall_value("dyn_join", i/cfg.joins_n))
    log(f"[{ad.name}] Dynamic joins done")

    # 4) dynamic leaves
    done = 0
    safety = 0
    prog(f"{ad.name}: Dynamic leaves", 0, cfg.leaves_n, _overall_value("dyn_leave", 0))
    while done < cfg.leaves_n and safety < 4000:
        safety += 1
        ok, hops, moved = ad.dyn_leave_one()
        if ok:
            leave_hops.append(int(hops))
            leave_moved.append(int(moved))
            done += 1
            prog(f"{ad.name}: Dynamic leaves", done, cfg.leaves_n, _overall_value("dyn_leave", done/cfg.leaves_n))
        # if ok==False we just retry another random candidate
    log(f"[{ad.name}] Dynamic leaves done")

    # titles
    all_titles = df["title"].dropna().astype(str).tolist()
    unique_titles = list(dict.fromkeys(all_titles))

    # 5) updates
    upd_total = min(cfg.updates_n, len(unique_titles))
    prog(f"{ad.name}: Updates", 0, upd_total, _overall_value("update", 0))
    for i, t in enumerate(random.sample(unique_titles, k=upd_total), start=1):
        update_hops.append(int(ad.update_one(t)))
        if i == upd_total or i % 200 == 0:
            prog(f"{ad.name}: Updates", i, upd_total, _overall_value("update", i/upd_total))
    log(f"[{ad.name}] Updates done")

    # 6) deletes
    del_total = min(cfg.deletes_n, len(unique_titles))
    prog(f"{ad.name}: Deletes", 0, del_total, _overall_value("delete", 0))
    for i, t in enumerate(random.sample(unique_titles, k=del_total), start=1):
        delete_hops.append(int(ad.delete_one(t)))
        if i == del_total or i % 200 == 0:
            prog(f"{ad.name}: Deletes", i, del_total, _overall_value("delete", i/del_total))
    log(f"[{ad.name}] Deletes done")

    # 7) lookups (K)
    titles = choose_titles(all_titles, cfg)
    prog(f"{ad.name}: Lookups", 0, len(titles), _overall_value("lookup", 0))
    for i, t in enumerate(titles, start=1):
        pop, hops = ad.lookup_one(t)
        lookup_hops.append(int(hops))
        lookup_rows.append(LookupRow(title=t, popularity=pop, hops=int(hops)))
        prog(f"{ad.name}: Lookups", i, len(titles), _overall_value("lookup", i/len(titles)))
    log(f"[{ad.name}] Lookups done")

    summary_rows = [
        (ad.initial_join_label, summarize(initial_join_hops)),
        ("Initial join moved records", summarize(initial_join_moved)),
        ("Insert hops", summarize(insert_hops)),
        ("Join total hops", summarize(join_hops)),
        ("Join moved records", summarize(join_moved)),
        ("Leave total hops", summarize(leave_hops)),
        ("Leave moved records", summarize(leave_moved)),
        ("Update hops", summarize(update_hops)),
        ("Delete hops", summarize(delete_hops)),
        ("Lookup hops", summarize(lookup_hops)),
    ]

    return RunResult(name=ad.name, summary_rows=summary_rows, lookup_rows=lookup_rows)


# ---------- adapters ----------
def run_chord(df, cfg: RunConfig, log_cb=None, progress_cb=None) -> RunResult:
    ring = ChordRing(m=cfg.m_bits)
    space = 2 ** ring.m

    # join signatures vary a bit; normalize to (hops, moved)
    def join_one(nid: int) -> Tuple[int, int]:
        ret = ring.join_node(nid)
        if isinstance(ret, tuple) and len(ret) >= 3:
            return int(ret[1]), int(ret[2] or 0)
        if isinstance(ret, tuple) and len(ret) == 2:
            return int(ret[1]), 0
        return int(ret), 0

    def insert_one(rec: Dict[str, Any]) -> int:
        key = chord_hash(str(rec["title"]), m=ring.m)
        ret = ring.insert(key, rec)
        return int(ret[1] if isinstance(ret, tuple) else ret)

    def dyn_join_one(nid: int) -> Tuple[int, int]:
        return join_one(nid)

    def dyn_leave_one() -> Tuple[bool, int, int]:
        if len(getattr(ring, "nodes", [])) <= 2:
            return False, 0, 0
        leave_node = random.choice(ring.nodes[1:])
        leave_id = getattr(leave_node, "node_id", getattr(leave_node, "id", None))

        if "start_node" in ring.leave_node.__code__.co_varnames:
            start = random.choice([n for n in ring.nodes if n is not leave_node])
            ret = ring.leave_node(leave_id, start_node=start)
        else:
            ret = ring.leave_node(leave_id)

        if isinstance(ret, tuple) and len(ret) >= 3:
            return bool(ret[0]), int(ret[1]), int(ret[2] or 0)
        if isinstance(ret, tuple) and len(ret) == 2:
            return True, int(ret[1]), 0
        return True, int(ret), 0

    def update_one(title: str) -> int:
        ret = ring.update_movie_field(title, "popularity", 9.5)
        return int(ret[1] if isinstance(ret, tuple) and len(ret) > 1 else (ret if not isinstance(ret, tuple) else 0))

    def delete_one(title: str) -> int:
        if hasattr(ring, "delete_title"):
            ret = ring.delete_title(title)
            return int(ret[1] if isinstance(ret, tuple) else ret)
        key = chord_hash(title, m=ring.m)
        ret = ring.delete(key)
        return int(ret[1] if isinstance(ret, tuple) else ret)

    def lookup_one(title: str) -> Tuple[Optional[float], int]:
        records, hops = ring.lookup(title)
        pop = None
        if records:
            rec = records[0] if isinstance(records, list) else records
            if isinstance(rec, dict) and "popularity" in rec:
                pop = rec["popularity"]
        return pop, int(hops)

    ad = Adapter(
        name="Chord",
        space=space,
        initial_join_label="Initial join hops",
        join_one=join_one,
        insert_one=insert_one,
        dyn_join_one=dyn_join_one,
        dyn_leave_one=dyn_leave_one,
        update_one=update_one,
        delete_one=delete_one,
        lookup_one=lookup_one,
    )
    return _run(df, cfg, ad, log_cb=log_cb, progress_cb=progress_cb)


def run_pastry(df, cfg: RunConfig, log_cb=None, progress_cb=None) -> RunResult:
    ring = PastryRing(m=cfg.m_bits, leaf_size=8, btree_size=32)
    space = 2 ** ring.m

    def join_one(nid: int) -> Tuple[int, int]:
        _, hops, _, moved = ring.join_node(nid)
        return int(hops), int(moved)

    def insert_one(rec: Dict[str, Any]) -> int:
        return int(ring.insert_title(str(rec["title"]), rec, start_node=random.choice(ring.nodes)))

    def dyn_join_one(nid: int) -> Tuple[int, int]:
        return join_one(nid)

    def dyn_leave_one() -> Tuple[bool, int, int]:
        if len(ring.nodes) <= 2:
            return False, 0, 0
        leave_node = random.choice(ring.nodes[1:])
        ok, hops, moved = ring.leave_node(leave_node.id)
        return bool(ok), int(hops), int(moved)

    def update_one(title: str) -> int:
        _, hops = ring.update_movie_field(title, "popularity", 9.5, start_node=random.choice(ring.nodes))
        return int(hops)

    def delete_one(title: str) -> int:
        return int(ring.delete_title(title, start_node=random.choice(ring.nodes)))

    def lookup_one(title: str) -> Tuple[Optional[float], int]:
        records, hops = ring.lookup(title, start_node=random.choice(ring.nodes))
        pop = None
        if records:
            rec = records[0] if isinstance(records, list) else records
            if isinstance(rec, dict) and "popularity" in rec:
                pop = rec["popularity"]
        return pop, int(hops)

    ad = Adapter(
        name="Pastry",
        space=space,
        initial_join_label="Initial join (locate) hops",
        join_one=join_one,
        insert_one=insert_one,
        dyn_join_one=dyn_join_one,
        dyn_leave_one=dyn_leave_one,
        update_one=update_one,
        delete_one=delete_one,
        lookup_one=lookup_one,
    )
    return _run(df, cfg, ad, log_cb=log_cb, progress_cb=progress_cb)