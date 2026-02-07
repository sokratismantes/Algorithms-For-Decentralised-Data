from b_tree import BPlusTree
import hashlib


def chord_hash(value, m=40):
    h = hashlib.sha1(str(value).encode()).hexdigest()
    num = int(h, 16) % (2**m)
    return num


class ChordNode:
    def __init__(self, node_id, m, btree_size=32):
        self.node_id = node_id
        self.m = m
        self.btree = BPlusTree(btree_size)

        self.successor = None
        self.predecessor = None

        # finger table size m
        self.finger = [None] * m

    def __repr__(self):
        return f"Node({self.node_id})"


class ChordRing:
    def __init__(self, m=160, btree_size=32):
        self.m = m
        self.btree_size = btree_size
        self.nodes = []

    # ------------------------- helpers -------------------------
    def _in_interval(self, key, start, end):
        """(start, end] in circular space."""
        if start < end:
            return start < key <= end
        else:
            return key > start or key <= end

    def _update_links(self):
        n = len(self.nodes)
        for i, node in enumerate(self.nodes):
            node.successor = self.nodes[(i + 1) % n]
            node.predecessor = self.nodes[(i - 1) % n]

    # ------------------------- routing -------------------------
    def find_successor(self, key, start_node=None):
        """
        Find the successor node for a given key using Chord routing.
        Returns (node, hops).
        """
        if not self.nodes:
            return None, 0

        start_node = start_node or self.nodes[0]
        curr = start_node
        hops = 0

        # safety bound to avoid infinite loops 
        max_steps = max(4, len(self.nodes) * 4)

        while True:
            if key == curr.node_id:
                return curr, hops

            # normal success condition
            succ = curr.successor
            if self._in_interval(key, curr.node_id, succ.node_id):
                return succ, hops + 1  # count the final hop to successor

            nxt = self.closest_preceding_finger(curr, key)
            if nxt is None or nxt is curr:
                nxt = succ

            curr = nxt
            hops += 1

            if hops > max_steps:
                # fallback to linear scan 
                return self.find_successor_linear(key), hops

    def find_successor_linear(self, key):
        """Linear successor search."""
        for node in self.nodes:
            if key <= node.node_id:
                return node
        return self.nodes[0]

    def closest_preceding_finger(self, node, key):
        for i in reversed(range(self.m)):
            finger = node.finger[i]
            if finger and self._in_interval(finger.node_id, node.node_id, key):
                return finger
        return node

    # ------------------------- finger maintenance -------------------------
    def fix_all_fingers(self):
        for node in self.nodes:
            self.init_finger_table(node)

    def init_finger_table(self, node):
        max_id = 2**self.m
        for i in range(self.m):
            start = (node.node_id + 2**i) % max_id
            successor = self.find_successor_linear(start)
            node.finger[i] = successor

    # ------------------------- data operations -------------------------
    def insert(self, key_int, record, start_node=None):
        """Insert record under an already-hashed key."""
        node, hops = self.find_successor(key_int, start_node=start_node)
        node.btree.insert(record, key_int)
        return hops

    def insert_title(self, movie_title, record, start_node=None):
        """Insert by title (hash inside for safety/consistency)."""
        key_int = chord_hash(movie_title, self.m)
        return self.insert(key_int, record, start_node=start_node)

    def lookup(self, movie_title, start_node=None):
        key_int = chord_hash(movie_title, self.m)
        node, hops = self.find_successor(key_int, start_node=start_node)
        records = node.btree.search_key(key_int)
        return records, hops

    def delete_key(self, key_int, start_node=None):
        node, hops = self.find_successor(key_int, start_node=start_node)
        node.btree.delete(key_int)
        return hops

    def delete_title(self, movie_title, start_node=None):
        key_int = chord_hash(movie_title, self.m)
        return self.delete_key(key_int, start_node=start_node)

    def update_movie_field(self, title, field, new_value, start_node=None):
        key_int = chord_hash(title, self.m)
        node, hops = self.find_successor(key_int, start_node=start_node)
        records = node.btree.search_key(key_int)
        if not records:
            return False, hops
        record = records[0]
        record[field] = new_value
        return True, hops

    # ------------------------- membership operations -------------------------
    def join_node(self, node_id, start_node=None):
        """
        Node join.
        Returns:
        - routing hops to locate successor of node_id
        - moved_cnt
        - migrate_hops: routing hops used to move keys (sum over moved keys)
        """
        locate_hops = 0

        # routing cost to find where the node would attach before insertion
        if self.nodes:
            _, locate_hops = self.find_successor(node_id, start_node=start_node or self.nodes[0])

        new_node = ChordNode(node_id, self.m, btree_size=self.btree_size)
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda n: n.node_id)
        self._update_links()

        migrate_hops = 0
        moved_cnt = 0
        if len(self.nodes) > 1:
            migrate_hops, moved_cnt = self._redistribute_keys(new_node)

        self.fix_all_fingers()

        return new_node, locate_hops, moved_cnt

    def _redistribute_keys(self, new_node):
        """
        Move keys from new_node.successor to new_node when they fall in (pred, new_node].
        Returns (migrate_hops, moved_count).
        """
        pred = new_node.predecessor
        succ = new_node.successor

        items = succ.btree.get_all_items()
        total_hops = 0
        moved = 0

        for key_int, record in items:
            if self._in_interval(key_int, pred.node_id, new_node.node_id):
                # count routing cost 
                _, hops = self.find_successor(key_int, start_node=succ)
                total_hops += hops
                new_node.btree.insert(record, key_int)
                succ.btree.delete(key_int)
                moved += 1

        return total_hops, moved

    def leave_node(self, node_id, start_node=None):
        """
        Node leave.
        Returns:
        - routing_hops (event): overlay routing cost for the LEAVE request
        - moved_cnt
        """
        node = next((n for n in self.nodes if n.node_id == node_id), None)
        if not node:
            return False, 0, 0

        if len(self.nodes) == 1:
            self.nodes.remove(node)
            return True, 0, 0

        routing_hops = 0
        if self.nodes:
            start = start_node or self.nodes[0]
            _, routing_hops = self.find_successor(node_id, start_node=start)

        succ = node.successor

        # move data for correctness
        moved = 0
        items = node.btree.get_all_items()
        for key_int, record in items:
            succ.btree.insert(record, key_int)
            moved += 1

        self.nodes.remove(node)
        self._update_links()
        self.fix_all_fingers()

        return True, routing_hops, moved

    # ------------------------- debug -------------------------
    def print_nodes_summary(self):
        for n in self.nodes:
            items = n.btree.get_all_items()
            print(f"{n} keys = {len(items)}")

