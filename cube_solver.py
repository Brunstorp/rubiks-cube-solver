import os
import pickle
import time

from cube import Cube

INVERT = {
    "U": "U'", "U'": "U", "U2": "U2",
    "D": "D'", "D'": "D", "D2": "D2",
    "L": "L'", "L'": "L", "L2": "L2",
    "R": "R'", "R'": "R", "R2": "R2",
    "F": "F'", "F'": "F", "F2": "F2",
    "B": "B'", "B'": "B", "B2": "B2",
}


def _invert_path(path):
    return [INVERT[m] for m in reversed(path)]


# Sticker indices for each of the 8 corner slots, in cube position order:
# 0:UFL, 1:UFR, 2:UBL, 3:UBR, 4:DFL, 5:DFR, 6:DBL, 7:DBR
CORNER_SLOTS = [
    (15, 0, 20),    # UFL
    (17, 2, 27),    # UFR
    (9,  18, 47),   # UBL
    (11, 29, 45),   # UBR
    (36, 6, 26),    # DFL
    (38, 8, 33),    # DFR
    (42, 24, 53),   # DBL
    (44, 35, 51),   # DBR
]

# Each corner cubie has a unique 3-color set. Map sorted-color frozenset -> corner ID.
CORNER_ID_BY_COLORS = {
    frozenset("UFL"): 0,
    frozenset("UFR"): 1,
    frozenset("ULB"): 2,
    frozenset("URB"): 3,
    frozenset("DFL"): 4,
    frozenset("DFR"): 5,
    frozenset("DLB"): 6,
    frozenset("DRB"): 7,
}


def corner_perm_of(state):
    # Returns an 8-tuple: corner_perm[i] = ID of the cubie currently at slot i.
    return tuple(
        CORNER_ID_BY_COLORS[frozenset(state[j] for j in slot)]
        for slot in CORNER_SLOTS
    )


class Solver:

    solved_state = list(Cube.solved_state)

    def __init__(self, cube):
        self.cube = cube
        self.perm_table = cube.table
        self.OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"}
        self.FIRST_OF_PAIR = {"U", "R", "F"}
        self.forward_table = None
        self.forward_depth = 0
        self.corner_pdb = None

    def _is_pruned(self, face, last_face):
        if face == last_face:
            return True
        if face == self.OPPOSITE.get(last_face) and last_face not in self.FIRST_OF_PAIR:
            return True
        return False

    def precompute_forward(self, depth, cache_path=None):
        # BFS from solved up to `depth` moves. Stores state -> inverse path.
        self.forward_depth = depth
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.forward_table = pickle.load(f)
            print(f"Loaded forward table from {cache_path}: {len(self.forward_table)} states")
            return self.forward_table

        solved_str = "".join(Solver.solved_state)
        table = {solved_str: []}
        frontier = [(list(Solver.solved_state), [], None)]

        t0 = time.time()
        for d in range(depth):
            next_frontier = []
            for state, fwd_path, last_face in frontier:
                for name, perm in self.perm_table.items():
                    face = name[0]
                    if self._is_pruned(face, last_face):
                        continue
                    new_state = [state[perm[i]] for i in range(54)]
                    key = "".join(new_state)
                    if key in table:
                        continue
                    new_fwd = fwd_path + [name]
                    table[key] = _invert_path(new_fwd)
                    next_frontier.append((new_state, new_fwd, face))
            frontier = next_frontier
            print(f"  depth {d+1}: {len(table)} total states ({time.time()-t0:.2f}s)")

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(table, f)
            print(f"Saved forward table to {cache_path}")

        self.forward_table = table
        return table

    def precompute_corner_pdb(self, cache_path=None):
        # BFS over the 8! = 40,320 corner permutations. Pure HTM (all 18 moves).
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.corner_pdb = pickle.load(f)
            print(f"Loaded corner PDB from {cache_path}: {len(self.corner_pdb)} states")
            return self.corner_pdb

        # Compute the corner-permutation effect of each move (extracted from solved cube)
        solved = list(Solver.solved_state)
        move_cps = {}
        for name, perm in self.perm_table.items():
            new_state = [solved[perm[i]] for i in range(54)]
            move_cps[name] = corner_perm_of(new_state)

        # BFS on corner permutations
        start = (0, 1, 2, 3, 4, 5, 6, 7)
        pdb = {start: 0}
        frontier = [start]
        depth = 0
        t0 = time.time()
        while frontier:
            depth += 1
            next_frontier = []
            for cp in frontier:
                for mcp in move_cps.values():
                    new_cp = (cp[mcp[0]], cp[mcp[1]], cp[mcp[2]], cp[mcp[3]],
                              cp[mcp[4]], cp[mcp[5]], cp[mcp[6]], cp[mcp[7]])
                    if new_cp not in pdb:
                        pdb[new_cp] = depth
                        next_frontier.append(new_cp)
            frontier = next_frontier
            if frontier:
                print(f"  corner depth {depth}: {len(pdb)} total ({time.time()-t0:.2f}s)")

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(pdb, f)
            print(f"Saved corner PDB to {cache_path}")

        self.corner_pdb = pdb
        return pdb

    def heuristic(self, state):
        # Lower bound on moves to solved (admissible). Uses corner-perm PDB if loaded; else 0.
        if self.corner_pdb is None:
            return 0
        return self.corner_pdb[corner_perm_of(state)]

    def solve_mitm(self, max_back_depth=8):
        if self.forward_table is None:
            raise RuntimeError("Call precompute_forward(depth) first")

        key = "".join(self.cube.state)
        if key in self.forward_table:
            return self.forward_table[key]

        for back_depth in range(1, max_back_depth + 1):
            path = []
            tail = self._dfs_mitm(self.cube.state, back_depth, None, path)
            if tail is not None:
                return path + tail
        return None

    def _dfs_mitm(self, state, depth_left, last_face, path):
        if depth_left == 0:
            return self.forward_table.get("".join(state))
        # Heuristic prune: if h(state) > depth_left + forward_depth, no path can reach
        # the forward table in `depth_left` moves (triangle inequality).
        if self.heuristic(state) > depth_left + self.forward_depth:
            return None
        for name, perm in self.perm_table.items():
            face = name[0]
            if self._is_pruned(face, last_face):
                continue
            new_state = [state[perm[i]] for i in range(54)]
            path.append(name)
            tail = self._dfs_mitm(new_state, depth_left - 1, face, path)
            if tail is not None:
                return tail
            path.pop()
        return None


def _verify(scramble, solution):
    c = Cube()
    c.move(scramble)
    c.move(" ".join(solution))
    return c.state == Cube.solved_state


if __name__ == "__main__":
    FORWARD_DEPTH = 6
    FWD_CACHE = f"forward_d{FORWARD_DEPTH}.pkl"
    PDB_CACHE = "corner_pdb.pkl"

    builder = Solver(Cube())
    print(f"Building/loading forward table to depth {FORWARD_DEPTH}...")
    builder.precompute_forward(FORWARD_DEPTH, cache_path=FWD_CACHE)
    print("Building/loading corner-permutation PDB...")
    builder.precompute_corner_pdb(cache_path=PDB_CACHE)

    scrambles = [
        "R U R' U'",                                       # 4
        "R U R' U' F R F'",                                # 7
        "R U2 R' U' R U' R'",                              # 7
        "R U R' U' R' F R2 U' R'",                         # 9
        "R U2 R2 F R F' U2 R' F R F'",                     # 11
    ]

    print("\nWith heuristic ON:")
    for scramble in scrambles:
        cube = Cube()
        cube.move(scramble)
        solver = Solver(cube)
        solver.forward_table = builder.forward_table
        solver.forward_depth = builder.forward_depth
        solver.corner_pdb = builder.corner_pdb

        t0 = time.time()
        solution = solver.solve_mitm(max_back_depth=8)
        elapsed = time.time() - t0
        ok = _verify(scramble, solution) if solution else False
        n = len(solution) if solution else "-"
        print(f"  {scramble!r:50s} -> len={n}, {elapsed:.2f}s, verified={ok}")
