from pathlib import Path
import random
import statistics
from threading import Thread

from plot_chord import plot_main_chord_results, records_per_node_chord

from data_read import load_and_preprocess_csv
from chord import ChordRing, chord_hash


# ---------------- helpers -------------------
def _p95(values):
    if not values:
        return None
    s = sorted(values)
    idx = int(round(0.95 * (len(s) - 1)))
    return s[idx]


def _stats_line(name, values):
    if not values:
        print(f"{name}: (no values)")
        return
    print(
        f"{name}: n={len(values)}  avg={statistics.mean(values):.2f}  "
        f"median={statistics.median(values):.2f}  p95={_p95(values)}  "
        f"min={min(values)}  max={max(values)}"
    )


# --------- main ------------
def main():
    project_root = Path(".").resolve()
    data_path = project_root / "data_movies_clean.csv"

    print(f"Loading dataset from: {data_path}")
    df = load_and_preprocess_csv(str(data_path), max_rows=946_460, seed=1)
    print(f"Loaded {len(df)} rows.\n")

    # params 
    num_nodes = 32
    updates_n = 2_000
    deletes_n = 2_000
    joins_n = 10
    leaves_n = 10

    ring = ChordRing(m=40)

    # -------- initial nodes with random unique IDs --------
    N0 = num_nodes
    space = 2 ** ring.m
    random.seed(42)
    node_ids = random.sample(range(space), k=num_nodes)

    print("=== Joining initial Chord nodes ===")
    join_hops_list = []
    moved_list = []
    for nid in node_ids:
        ret = ring.join_node(nid)
        if isinstance(ret, tuple) and len(ret) >= 3:
            hops, moved = ret[1], ret[2]
        elif isinstance(ret, tuple) and len(ret) == 2:
            hops, moved = ret[1], 0
        else:
            hops, moved = int(ret), 0
        join_hops_list.append(int(hops))
        moved_list.append(int(moved) if moved is not None else 0)

    if hasattr(ring, "print_nodes_summary"):
        ring.print_nodes_summary()
    _stats_line("Initial join hops", join_hops_list)
    _stats_line("Initial join moved records", moved_list)

    # -------- Insert all movies --------
    print("\n=== Inserting movies into Chord ===")
    insert_hops = []
    for _, row in df.iterrows():
        title = str(row["title"])
        key = chord_hash(title, m=ring.m)
        ret = ring.insert(key, row.to_dict())
        hops = ret[1] if isinstance(ret, tuple) else ret
        insert_hops.append(int(hops))

    if hasattr(ring, "print_nodes_summary"):
        ring.print_nodes_summary()
    _stats_line("Insert hops", insert_hops)

    load_after_inserts = records_per_node_chord(ring)
    load_after_inserts.sort(key=lambda x: x[0])

    # -------- 10 Dynamic JOINS --------
    print("\n=== Dynamic JOIN x10 (Chord) ===")
    join_moved_list = []
    join_total_hops_list = []

    existing = set(getattr(n, "node_id", getattr(n, "id", None)) for n in getattr(ring, "nodes", []))
    for _ in range(joins_n):
        nid = random.randrange(0, space)
        while nid in existing:
            nid = random.randrange(0, space)
        existing.add(nid)

        ret = ring.join_node(nid)

        if isinstance(ret, tuple) and len(ret) >= 3:
            locate_hops, moved = ret[1], ret[2]
        elif isinstance(ret, tuple) and len(ret) == 2:
            locate_hops, moved = ret[1], 0
        else:
            locate_hops, moved = int(ret), 0

        # total hops for JOIN 
        join_total_hops_list.append(int(locate_hops))
        join_moved_list.append(int(moved) if moved is not None else 0)

    _stats_line("Join total hops", join_total_hops_list)
    _stats_line("Join moved records", join_moved_list)
    if hasattr(ring, "print_nodes_summary"):
        ring.print_nodes_summary()

    load_after_join = records_per_node_chord(ring)
    load_after_join.sort(key=lambda x: x[0])

    # -------- 10 Dynamic LEAVES --------
    print("\n=== Dynamic LEAVE x10 (Chord) ===")
    leave_moved_list = []
    leave_total_hops_list = []

    leaves_done = 0
    safety = 0
    while leaves_done < leaves_n and len(getattr(ring, "nodes", [])) > 2 and safety < 2000:
        safety += 1
        leave_node = random.choice(ring.nodes[1:])
        leave_id = getattr(leave_node, "node_id", getattr(leave_node, "id", None))

        if "start_node" in ring.leave_node.__code__.co_varnames:
            start = random.choice([n for n in ring.nodes if n is not leave_node])
            ret = ring.leave_node(leave_id, start_node=start)
        else:
            ret = ring.leave_node(leave_id)

        if isinstance(ret, tuple) and len(ret) >= 3:
            ok, routing_hops, moved = ret[0], ret[1], ret[2]
        elif isinstance(ret, tuple) and len(ret) == 2:
            ok, routing_hops, moved = True, ret[1], 0
        else:
            ok, routing_hops, moved = True, int(ret), 0

        # total hops for LEAVE 
        if ok:
            leave_total_hops_list.append(int(routing_hops))
            leave_moved_list.append(int(moved) if moved is not None else 0)
            leaves_done += 1

    print(f"Leaves done={leaves_done}/{leaves_n}")
    _stats_line("Leave total hops", leave_total_hops_list)
    _stats_line("Leave moved records", leave_moved_list)
    if hasattr(ring, "print_nodes_summary"):
        ring.print_nodes_summary()

    load_after_leave = records_per_node_chord(ring)
    load_after_leave.sort(key=lambda x: x[0])

    # -------- UPDATE (2000) --------
    print("\n=== UPDATE (Chord) ===")
    all_titles = df["title"].dropna().astype(str).tolist()
    unique_titles = list(dict.fromkeys(all_titles))

    update_hops = []
    update_ok = 0
    for t in random.sample(unique_titles, k=min(updates_n, len(unique_titles))):
        ret = ring.update_movie_field(t, "popularity", 9.5)
        if isinstance(ret, tuple):
            ok = bool(ret[0]) if isinstance(ret[0], bool) else True
            hops = ret[1] if len(ret) > 1 else 0
        else:
            ok = True
            hops = ret
        update_hops.append(int(hops))
        if ok:
            update_ok += 1

    print(f"Updates attempted={len(update_hops)} success={update_ok}")
    _stats_line("Update hops", update_hops)

    # -------- DELETE (2000) --------
    print("\n=== DELETE (Chord) ===")
    delete_hops = []
    for t in random.sample(unique_titles, k=min(deletes_n, len(unique_titles))):
        if hasattr(ring, "delete_title"):
            ret = ring.delete_title(t)
            hops = ret[1] if isinstance(ret, tuple) else ret
        else:
            key = chord_hash(t, m=ring.m)
            ret = ring.delete(key)
            hops = ret[1] if isinstance(ret, tuple) else ret
        delete_hops.append(int(hops))

    print(f"Deletes attempted={len(delete_hops)}")
    _stats_line("Delete hops", delete_hops)

    # -------- Concurrent lookups demo --------
    def lookup_movie(title, ring_obj, results_dict):
        records, hops = ring_obj.lookup(title)
        if records:
            results_dict[title] = (records[0].get("popularity"), hops)
        else:
            results_dict[title] = (None, hops)

    try:
        K = int(input("Dwse K (plh8os titlwn gia parallel lookup) [default=3]: ").strip() or "3")
    except ValueError:
        K = 3

    all_titles = df["title"].dropna().astype(str).tolist()
    titles_set = set(all_titles)

    titles_to_lookup = []
    for i in range(K):
        user_title = input(f'Dwse titlo #{i+1} (Enter gia tyxaio): ').strip()
        if not user_title:
            titles_to_lookup.append(random.choice(all_titles))
        elif user_title in titles_set:
            titles_to_lookup.append(user_title)
        else:
            print(f'  (Den vrethhke akribws o titlos "{user_title}", epilegw tyxaia apo to dataset)')
            titles_to_lookup.append(random.choice(all_titles))

    results = {}
    threads = []
    for title in titles_to_lookup:
        t = Thread(target=lookup_movie, args=(title, ring, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    print("\n=== Popularities of K movies (Chord) ===")
    lookup_hops = []
    for title, (popularity, hops) in results.items():
        lookup_hops.append(hops)
        if popularity is not None:
            print(f'"{title}": popularity={popularity}, hops={hops}')
        else:
            print(f'"{title}": NOT FOUND, hops={hops}')
    _stats_line("Lookup hops", lookup_hops)

    # -------- BASIC PLOTS --------
    out_dir_basic = project_root / "results_from_main_chord"
    plot_main_chord_results(
        out_dir=out_dir_basic,
        N0=N0,
        insert_hops=insert_hops,
        update_hops=update_hops,
        delete_hops=delete_hops,
        lookup_hops=lookup_hops,
        load_after_inserts=load_after_inserts,
        load_after_join=load_after_join,
        load_after_leave=load_after_leave,
        join_moved_list=join_moved_list,
        join_total_hops_list=join_total_hops_list,
        leave_moved_list=leave_moved_list,
        leave_total_hops_list=leave_total_hops_list,
    )
    print(f"\n[OK] Basic Chord plots saved under: {out_dir_basic}")


if __name__ == "__main__":
    main()
