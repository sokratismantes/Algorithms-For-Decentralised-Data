from pathlib import Path
import random
import statistics

from plot_pastry import plot_main_pastry_results, records_per_node_pastry

from data_read import load_and_preprocess_csv
from pastry import PastryRing


# ---------------- helpers ---------------------
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


def print_nodes_summary(ring: PastryRing):
    print(f"Nodes: {len(ring.nodes)}")
    for n in ring.nodes:
        records = len(n.btree.get_all_items())
        leaf = len(n.leaf_set)
        routing = sum(1 for row in n.route_table for v in row.values() if v is not None)
        print(f"  node={n.id}  leaf={leaf}  routing={routing}  records={records}")


# ---------------- main -----------------
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

    try:
        K = int(input("Dwse K (plh8os titlwn gia parallel lookup) [default=3]: ").strip() or "3")
    except ValueError:
        K = 3

    ring = PastryRing(m=40, leaf_size=8, btree_size=32)

    # -------- initial nodes - random unique IDs --------
    N0 = num_nodes
    space = 2 ** ring.m
    random.seed(42)
    node_ids = random.sample(range(space), k=num_nodes)

    print("=== Joining initial Pastry nodes ===")
    join_hops_list = []
    moved_list = []
    for nid in node_ids:
        _, locate_hops, _, moved = ring.join_node(nid)
        join_hops_list.append(int(locate_hops))
        moved_list.append(int(moved))

    print_nodes_summary(ring)
    _stats_line("Initial join (locate) hops", join_hops_list)
    _stats_line("Initial join moved records", moved_list)

    # -------- Insert all movies --------
    print("\n=== Inserting movies into Pastry ===")
    insert_hops = []
    for _, row in df.iterrows():
        title = str(row["title"])
        hops = ring.insert_title(title, row.to_dict(), start_node=random.choice(ring.nodes))
        insert_hops.append(int(hops))

    print_nodes_summary(ring)
    _stats_line("Insert hops", insert_hops)

    load_after_inserts = records_per_node_pastry(ring)
    load_after_inserts.sort(key=lambda x: x[0])

    # -------- 10 Dynamic JOINS (routing hops) --------
    print("\n=== Dynamic JOIN x10 (Pastry) ===")
    join_moved_list = []
    join_total_hops_list = []

    existing = set(n.id for n in ring.nodes)
    for _ in range(joins_n):
        nid = random.randrange(0, space)
        while nid in existing:
            nid = random.randrange(0, space)
        existing.add(nid)

        _, locate_hops, _, moved = ring.join_node(nid)

        join_total_hops_list.append(int(locate_hops))  
        join_moved_list.append(int(moved))

    _stats_line("Join total hops", join_total_hops_list)
    _stats_line("Join moved records", join_moved_list)
    print_nodes_summary(ring)

    load_after_join = records_per_node_pastry(ring)
    load_after_join.sort(key=lambda x: x[0])

    # -------- 10 Dynamic LEAVES (routing hops) --------
    print("\n=== Dynamic LEAVE x10 (Pastry) ===")
    leave_moved_list = []
    leave_total_hops_list = []

    leaves_done = 0
    safety = 0
    while leaves_done < leaves_n and len(ring.nodes) > 2 and safety < 1000:
        safety += 1
        leave_node = random.choice(ring.nodes[1:])
        ok, routing_hops, moved = ring.leave_node(leave_node.id)
        if ok:
            leave_total_hops_list.append(int(routing_hops))  
            leave_moved_list.append(int(moved))
            leaves_done += 1

    print(f"Leaves done={leaves_done}/{leaves_n}")
    _stats_line("Leave total hops", leave_total_hops_list)
    _stats_line("Leave moved records", leave_moved_list)
    print_nodes_summary(ring)

    load_after_leave = records_per_node_pastry(ring)
    load_after_leave.sort(key=lambda x: x[0])

    # -------- UPDATE --------
    print("\n=== UPDATE (Pastry) ===")
    all_titles = df["title"].dropna().astype(str).tolist()
    unique_titles = list(dict.fromkeys(all_titles))

    update_hops = []
    update_ok = 0
    for t in random.sample(unique_titles, k=min(updates_n, len(unique_titles))):
        start_node = random.choice(ring.nodes)
        updated, hops = ring.update_movie_field(t, "popularity", 9.5, start_node=start_node)
        update_hops.append(int(hops))
        if updated:
            update_ok += 1

    print(f"Updates attempted={len(update_hops)} success={update_ok}")
    _stats_line("Update hops", update_hops)

    # -------- DELETE --------
    print("\n=== DELETE (Pastry) ===")
    delete_hops = []
    for t in random.sample(unique_titles, k=min(deletes_n, len(unique_titles))):
        start_node = random.choice(ring.nodes)
        hops = ring.delete_title(t, start_node=start_node)
        delete_hops.append(int(hops))

    print(f"Deletes attempted={len(delete_hops)}")
    _stats_line("Delete hops", delete_hops)

    # -------- LOOKUP --------
    print("\n=== LOOKUP (Pastry) ===")
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

    print("\n=== Popularities of K movies (Pastry) ===")
    lookup_hops = []

    for title in titles_to_lookup:
        records, hops = ring.lookup(title, start_node=random.choice(ring.nodes))
        lookup_hops.append(int(hops))

        rec = None
        if records:
            rec = records[0] if isinstance(records, list) else records

        if isinstance(rec, dict) and ("popularity" in rec):
            print(f'"{title}": popularity={rec["popularity"]}, hops={hops}')
        else:
            print(f'"{title}": NOT FOUND, hops={hops}')

    _stats_line("Lookup hops", lookup_hops)

    # -------- BASIC PLOTS --------
    out_dir_basic = project_root / "results_from_main_pastry"
    plot_main_pastry_results(
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
    print(f"\n[OK] Basic Pastry plots saved under: {out_dir_basic}")


if __name__ == "__main__":
    main()
