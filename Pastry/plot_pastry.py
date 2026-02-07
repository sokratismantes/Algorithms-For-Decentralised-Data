from pathlib import Path
import matplotlib.pyplot as plt


def records_per_node_pastry(ring):
    return [(n.id, len(n.btree.get_all_items())) for n in ring.nodes]


def plot_main_pastry_results(
    out_dir: Path,
    N0: int,
    insert_hops,
    update_hops,
    delete_hops,
    lookup_hops,
    load_after_inserts,
    load_after_join,
    load_after_leave,
    join_moved_list,
    join_total_hops_list,
    leave_moved_list,
    leave_total_hops_list,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # hops distributions
    def _hist(vals, title, fname):
        if not vals:
            return
        fig = plt.figure(figsize=(8.2, 5.0))
        ax = fig.add_subplot(111)
        lo = int(min(vals))
        hi = int(max(vals))
        ax.hist(vals, bins=range(lo, hi + 2))
        ax.set_title(title)
        ax.set_xlabel("Hops")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    _hist(insert_hops, f"Pastry INSERT hops distribution (N={N0})", "insert_hops_hist.png")
    _hist(lookup_hops, f"Pastry LOOKUP hops distribution (N={N0})", "lookup_hops_hist.png")
    _hist(update_hops, f"Pastry UPDATE hops distribution (N={N0})", "update_hops_hist.png")
    _hist(delete_hops, f"Pastry DELETE hops distribution (N={N0})", "delete_hops_hist.png")

    # load balancing - records per node
    def _load_bar(load, title, fname):
        if not load:
            return
        counts = [x[1] for x in load]
        fig = plt.figure(figsize=(10.5, 5.2))
        ax = fig.add_subplot(111)
        ax.bar(range(len(counts)), counts)
        ax.set_title(title)
        ax.set_xlabel("Node index (sorted by node id)")
        ax.set_ylabel("Records stored")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{i}" for i in range(len(counts))], rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    _load_bar(load_after_inserts, f"Load after INSERTS (records per node) (N={N0})", "load_after_inserts.png")
    _load_bar(load_after_join, f"Load after 10 JOINS (records per node)", "load_after_joins.png")
    _load_bar(load_after_leave, f"Load after 10 LEAVES (records per node)", "load_after_leaves.png")

    # join/leave overhead — moved records distributions
    if join_moved_list and leave_moved_list:
        fig = plt.figure(figsize=(9.2, 5.0))
        ax = fig.add_subplot(111)
        ax.boxplot([join_moved_list, leave_moved_list], labels=["JOIN moved", "LEAVE moved"], showfliers=False)
        ax.set_title("Join/Leave data movement (moved records) — distribution")
        ax.set_ylabel("Moved records")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "join_leave_moved_records_box.png", bbox_inches="tight")
        plt.close(fig)

    # join/leave overhead — routing hops distributions
    if join_total_hops_list and leave_total_hops_list:
        fig = plt.figure(figsize=(9.2, 5.0))
        ax = fig.add_subplot(111)
        ax.boxplot(
            [join_total_hops_list, leave_total_hops_list],
            labels=["JOIN routing hops", "LEAVE routing hops"],
            showfliers=False,
        )
        ax.set_title("Join/Leave routing overhead (routing hops only) — distribution")
        ax.set_ylabel("Routing hops")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "join_leave_routing_hops_box.png", bbox_inches="tight")
        plt.close(fig)

    # heatmap counts hop-count distribution per operation
    ops = {
        "insert": insert_hops,
        "lookup": lookup_hops,
        "update": update_hops,
        "delete": delete_hops,
    }

    all_vals = [v for lst in ops.values() for v in lst]
    if all_vals:
        hop_min = int(min(all_vals))
        hop_max = int(max(all_vals))
        hop_values = list(range(hop_min, hop_max + 1))
        op_names = list(ops.keys())

        M = []
        for op in op_names:
            counts = {h: 0 for h in hop_values}
            for h in ops[op]:
                counts[int(h)] = counts.get(int(h), 0) + 1
            M.append([counts[h] for h in hop_values])

        fig = plt.figure(figsize=(9.6, 4.8))
        ax = fig.add_subplot(111)
        im = ax.imshow(M, aspect="auto")
        ax.set_title(f"Heatmap: hop-count distribution per operation (N={N0})")
        ax.set_xlabel("Hops")
        ax.set_ylabel("Operation")
        ax.set_xticks(range(len(hop_values)))
        ax.set_xticklabels([str(h) for h in hop_values])
        ax.set_yticks(range(len(op_names)))
        ax.set_yticklabels(op_names)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "hops_heatmap_counts.png", bbox_inches="tight")
        plt.close(fig)

    # heatmap (AVG) - average hops per operation 
    op_names = ["insert", "lookup", "update", "delete"]
    avg_ops = []
    for op in op_names:
        vals = ops.get(op, [])
        avg_ops.append((sum(vals) / len(vals)) if vals else 0.0)

    fig = plt.figure(figsize=(4.8, 4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow([[x] for x in avg_ops], aspect="auto")
    ax.set_title(f"Heatmap: average hops per operation (N={N0})")
    ax.set_xlabel("avg")
    ax.set_ylabel("Operation")
    ax.set_xticks([0])
    ax.set_xticklabels(["avg hops"])
    ax.set_yticks(range(len(op_names)))
    ax.set_yticklabels(op_names)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg hops")
    fig.tight_layout()
    fig.savefig(out_dir / "hops_heatmap_avg.png", bbox_inches="tight")
    plt.close(fig)
