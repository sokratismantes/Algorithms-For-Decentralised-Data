from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import hashlib
from b_tree import BPlusTree


def pastry_hash(value, m=40) -> int:
    """Hash a value into the pastry keyspace (0..2^m-1)."""
    h = hashlib.sha1(str(value).encode()).hexdigest()
    return int(h, 16) % (2 ** m)


class PastryNode:
    def __init__(self, node_id: int, m: int, btree_size: int = 32, leaf_size: int = 8):
        self.id = node_id
        self.m = m
        self.btree = BPlusTree(btree_size)
        self.leaf_size = leaf_size

        # represent node id as binary string
        self.bit_len = m
        self.id_bits = f"{self.id:0{self.bit_len}b}"

        # route_table[level][bit] -> PastryNode or None
        self.route_table: List[Dict[int, Optional[PastryNode]]] = [
            {0: None, 1: None} for _ in range(self.bit_len)
        ]

        # leaf set: up to leaf_size numerically closest neighbors
        self.leaf_set: List[PastryNode] = []

    def __repr__(self):
        return f"PNode({self.id})"

    def distance_to(self, key: int) -> int:
        """Return ring distance between this node id and key (modulo space)."""
        max_id = 2 ** self.m
        diff = abs(self.id - key)
        return min(diff, max_id - diff)


class PastryRing:
    def __init__(self, m: int = 40, leaf_size: int = 8, btree_size: int = 32):
        self.m = m
        self.nodes: List[PastryNode] = []
        self.leaf_size = leaf_size
        self.bit_len = m
        self.btree_size = btree_size

    # ------------------------- helpers -------------------------
    def _id_to_bits(self, id_int: int) -> str:
        return f"{id_int:0{self.bit_len}b}"

    def _prefix_len(self, a_bits: str, b_bits: str) -> int:
        i = 0
        for ca, cb in zip(a_bits, b_bits):
            if ca != cb:
                break
            i += 1
        return i

    def _best_leaf_candidate(self, curr: PastryNode, key_int: int) -> PastryNode:
        """Return numerically closest node among curr + leaf_set."""
        best = curr
        best_dist = curr.distance_to(key_int)
        for ln in curr.leaf_set:
            d = ln.distance_to(key_int)
            if d < best_dist:
                best = ln
                best_dist = d
        return best

    # ------------------------- topology maintenance -------------------------
    def _rebuild_all(self):
        """
        Recompute route tables and leaf sets for all nodes.

        NOTE: This is a simplified maintenance step (centralized rebuild) for the project,
        while the routing/hops are measured using ONLY overlay forwarding decisions.
        """
        if not self.nodes:
            return

        self.nodes.sort(key=lambda x: x.id)

        for node in self.nodes:
            node.id_bits = self._id_to_bits(node.id)

        for node in self.nodes:
            idx = self.nodes.index(node)

            # leaf set: collect up to leaf_size neighbors around the node (circular)
            neighbors: List[PastryNode] = []
            left = 1
            right = 1
            while len(neighbors) < self.leaf_size and (left <= len(self.nodes) or right <= len(self.nodes)):
                if left <= len(self.nodes) - 1:
                    nleft = self.nodes[(idx - left) % len(self.nodes)]
                    if nleft is not node and nleft not in neighbors:
                        neighbors.append(nleft)
                if len(neighbors) < self.leaf_size and right <= len(self.nodes) - 1:
                    nright = self.nodes[(idx + right) % len(self.nodes)]
                    if nright is not node and nright not in neighbors:
                        neighbors.append(nright)
                left += 1
                right += 1
            node.leaf_set = neighbors[: self.leaf_size]

            # routing table (binary prefix routing)
            node.route_table = [{0: None, 1: None} for _ in range(node.bit_len)]
            for other in self.nodes:
                if other is node:
                    continue
                pref = self._prefix_len(node.id_bits, other.id_bits)
                if pref < node.bit_len:
                    next_bit = int(other.id_bits[pref])
                    existing = node.route_table[pref].get(next_bit)
                    if existing is None:
                        node.route_table[pref][next_bit] = other
                    else:
                        # keep a "better" representative (closer to node)
                        if other.distance_to(node.id) < existing.distance_to(node.id):
                            node.route_table[pref][next_bit] = other

    # ------------------------- membership operations -------------------------
    def join_node(self, node_id: int) -> Tuple[PastryNode, int, int]:
        """
        Node join.
        Returns (new_node, total_hops, moved_count)
        where total_hops counts:
          - a locate cost (routing to its own id)
          - hops for moved keys (routing + 1 transfer per moved key)
        """
        locate_hops = 0
        if self.nodes:
            start = self.nodes[0]
            _, locate_hops = self._route(start, node_id)

        new_node = PastryNode(node_id, self.m, btree_size=self.btree_size, leaf_size=self.leaf_size)
        self.nodes.append(new_node)
        self._rebuild_all()

        # minimal redistribution: only from numerically close neighbors (leaf set)
        moved = 0
        moved_hops = 0
        affected = list(new_node.leaf_set)

        for n in affected:
            items = n.btree.get_all_items()
            for k, rec in items:
                # if new node is closer to the key than current holder, move it
                if new_node.distance_to(k) < n.distance_to(k):
                    # routing cost from n to key (overlay forwards)
                    _, h = self._route(n, k)
                    moved_hops += h   
                    new_node.btree.insert(rec, k)
                    n.btree.delete(k)
                    moved += 1

        total = locate_hops + moved_hops
        return new_node, total, moved

    def leave_node(self, node_id: int) -> Tuple[bool, int, int]:
        """
        Node leave.
        Returns (ok, total_hops, moved_count)
        where total_hops counts routing to destination + 1 transfer per moved key.
        """
        node = next((n for n in self.nodes if n.id == node_id), None)
        if node is None:
            return False, 0, 0

        if len(self.nodes) == 1:
            self.nodes.remove(node)
            return True, 0, 0

        items = node.btree.get_all_items()
        self.nodes.remove(node)
        self._rebuild_all()

        # choose a start node for routing (any remaining node)
        start = self.nodes[0]

        moved = 0
        total_hops = 0
        for k, rec in items:
            owner, h = self._route(start, k)
            total_hops += h 
            owner.btree.insert(rec, k)
            moved += 1

        return True, total_hops, moved

    # ------------------------- routing -------------------------
    def _route(self, start_node: PastryNode, key_int: int) -> Tuple[PastryNode, int]:
        """
        Simplified Pastry routing:
        - First try to improve numerically using leaf_set.
        - Else use routing table on the first differing prefix bit.
        - Else fallback to best among known candidates (leaf_set + routing entries).
        Hops = number of overlay forwards (node-to-node moves).
        """
        if not self.nodes:
            raise RuntimeError("No nodes in pastry ring")

        key_bits = self._id_to_bits(key_int)
        curr = start_node
        hops = 0
        visited = set()
        max_hops = max(8, len(self.nodes) * 4)

        while True:
            visited.add(curr.id)

            # 1) leaf-set numerical improvement
            best_leaf = self._best_leaf_candidate(curr, key_int)
            if best_leaf is not curr:
                curr = best_leaf
                hops += 1
                if curr.id in visited or hops > max_hops:
                    return curr, hops
                continue

            # 2) prefix routing using routing table
            pref = self._prefix_len(curr.id_bits, key_bits)
            if pref >= self.bit_len:
                return curr, hops

            next_bit = int(key_bits[pref])
            candidate = curr.route_table[pref].get(next_bit)
            if candidate is not None and candidate.id not in visited:
                curr = candidate
                hops += 1
                if hops > max_hops:
                    return curr, hops
                continue

            # 3) fallback to any known candidate that improves prefix or distance
            candidates = []
            for row in curr.route_table:
                for n in row.values():
                    if n is not None:
                        candidates.append(n)
            candidates += curr.leaf_set

            # remove visited + None + self
            cand2 = [c for c in candidates if c is not None and c.id not in visited and c is not curr]
            if not cand2:
                return curr, hops

            best = None
            best_pref = -1
            best_dist = None
            for c in cand2:
                p = self._prefix_len(c.id_bits, key_bits)
                d = c.distance_to(key_int)
                if p > best_pref or (p == best_pref and (best_dist is None or d < best_dist)):
                    best = c
                    best_pref = p
                    best_dist = d

            if best is None:
                return curr, hops

            curr = best
            hops += 1
            if hops > max_hops:
                return curr, hops

    # ------------------------- data operations -------------------------
    def insert(self, key_int: int, record: dict, start_node: Optional[PastryNode] = None) -> int:
        if not self.nodes:
            raise RuntimeError("No nodes in pastry ring")
        if start_node is None:
            start_node = self.nodes[0]
        owner, hops = self._route(start_node, key_int)
        owner.btree.insert(record, key_int)
        return hops

    def insert_title(self, movie_title: str, record: dict, start_node: Optional[PastryNode] = None) -> int:
        key_int = pastry_hash(movie_title, m=self.m)
        return self.insert(key_int, record, start_node=start_node)

    def lookup(self, movie_title: str, start_node: Optional[PastryNode] = None) -> Tuple[Optional[List[dict]], int]:
        key_int = pastry_hash(movie_title, m=self.m)
        if not self.nodes:
            return None, 0
        if start_node is None:
            start_node = self.nodes[0]
        owner, hops = self._route(start_node, key_int)
        records = owner.btree.search_key(key_int)
        return records, hops

    def delete_key(self, key_int: int, start_node: Optional[PastryNode] = None) -> int:
        if not self.nodes:
            return 0
        if start_node is None:
            start_node = self.nodes[0]
        owner, hops = self._route(start_node, key_int)
        owner.btree.delete(key_int)
        return hops

    def delete_title(self, movie_title: str, start_node: Optional[PastryNode] = None) -> int:
        key_int = pastry_hash(movie_title, m=self.m)
        return self.delete_key(key_int, start_node=start_node)

    def update_movie_field(self, title: str, field: str, new_value, start_node: Optional[PastryNode] = None) -> Tuple[bool, int]:
        key_int = pastry_hash(title, m=self.m)
        if not self.nodes:
            return False, 0
        if start_node is None:
            start_node = self.nodes[0]
        owner, hops = self._route(start_node, key_int)
        records = owner.btree.search_key(key_int)
        if not records:
            return False, hops
        rec = records[0]
        rec[field] = new_value
        return True, hops

    def print_nodes_summary(self):
        for n in self.nodes:
            items = n.btree.get_all_items()
            print(f"{n} keys = {len(items)}")
