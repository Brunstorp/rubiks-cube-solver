"""
cube_solver.py — Kociemba's two-phase algorithm.

References: https://kociemba.org/math/twophase.htm

Operates on the Cube class from cube.py.

Strategy
--------
Phase 1: reduce the cube to the subgroup G1 = <U, D, L2, R2, F2, B2>.
A cube is in G1 iff every corner is correctly oriented, every edge is correctly
oriented, and the four E-slice edges (FR, FL, BL, BR) are located in the
E-slice (regardless of which one sits where).

Phase 2: starting in G1, finish the solve using only the 10 G1-moves.

Each phase is an IDA* search over Kociemba's coordinates, with a pruning table
built once at startup by BFS over the coordinate space.

Coordinates
-----------
Phase 1:
    eo_coord     edge orientations              2^11   = 2048
    co_coord     corner orientations            3^7    = 2187
    slice_coord  positions of the four          C(12,4)=  495
                 E-slice edges
Phase 2:
    cperm_coord  corner permutation             8!     = 40320
    edge8_coord  permutation of UD-layer edges  8!     = 40320
    eslice_coord permutation of E-slice edges   4!     =    24
"""

import math
import multiprocessing as mp
import os
import pickle
import time

from cube import Cube

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "kociemba_tables.pkl")


# ─── Cubie-level representation ────────────────────────────────────────────────
# Standard Kociemba indexing.
# Corners: URF=0  UFL=1  ULB=2  UBR=3  DFR=4  DLF=5  DBL=6  DRB=7
# Edges:   UR=0   UF=1   UL=2   UB=3   DR=4   DF=5   DL=6   DB=7
#          FR=8   FL=9   BL=10  BR=11

class CubieCube:
    """Cubie-level cube state: corner/edge permutations and orientations."""
    __slots__ = ('cp', 'co', 'ep', 'eo')

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.cp = list(cp) if cp is not None else list(range(8))
        self.co = list(co) if co is not None else [0] * 8
        self.ep = list(ep) if ep is not None else list(range(12))
        self.eo = list(eo) if eo is not None else [0] * 12

    def copy(self):
        return CubieCube(self.cp, self.co, self.ep, self.eo)

    def multiply(self, m):
        """In place: self ← self * m  (apply move m to the current state)."""
        cp, co, ep, eo = self.cp, self.co, self.ep, self.eo
        mcp, mco, mep, meo = m.cp, m.co, m.ep, m.eo
        self.cp = [cp[mcp[i]] for i in range(8)]
        self.co = [(co[mcp[i]] + mco[i]) % 3 for i in range(8)]
        self.ep = [ep[mep[i]] for i in range(12)]
        self.eo = [(eo[mep[i]] + meo[i]) % 2 for i in range(12)]
        return self

    def is_solved(self):
        return (self.cp == list(range(8)) and self.co == [0]*8
                and self.ep == list(range(12)) and self.eo == [0]*12)


def _compose(a, b):
    """Return a fresh CubieCube equal to a * b."""
    r = a.copy()
    r.multiply(b)
    return r


# ─── Move definitions (standard Kociemba conventions) ─────────────────────────
# These are the textbook permutations; see kociemba.org/math/CubeDefs.html.

_BASE = {
    'U': CubieCube(
        cp=[3, 0, 1, 2, 4, 5, 6, 7], co=[0]*8,
        ep=[3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], eo=[0]*12,
    ),
    'R': CubieCube(
        cp=[4, 1, 2, 0, 7, 5, 6, 3], co=[2, 0, 0, 1, 1, 0, 0, 2],
        ep=[8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0], eo=[0]*12,
    ),
    'F': CubieCube(
        cp=[1, 5, 2, 3, 0, 4, 6, 7], co=[1, 2, 0, 0, 2, 1, 0, 0],
        ep=[0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
        eo=[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    ),
    'D': CubieCube(
        cp=[0, 1, 2, 3, 5, 6, 7, 4], co=[0]*8,
        ep=[0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11], eo=[0]*12,
    ),
    'L': CubieCube(
        cp=[0, 2, 6, 3, 4, 1, 5, 7], co=[0, 1, 2, 0, 0, 2, 1, 0],
        ep=[0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11], eo=[0]*12,
    ),
    'B': CubieCube(
        cp=[0, 1, 3, 7, 4, 5, 2, 6], co=[0, 0, 1, 2, 0, 0, 2, 1],
        ep=[0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
        eo=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    ),
}

# Build all 18 face moves: M, M2, M'.
MOVE_NAMES = []
MOVES = []
MOVE_FACE = []  # face index 0..5 (U R F D L B) for each move, used to skip same-face seqs
for face_idx, face in enumerate('URFDLB'):
    m1 = _BASE[face]
    m2 = _compose(m1, m1)
    m3 = _compose(m2, m1)
    for suf, m in (('', m1), ('2', m2), ("'", m3)):
        MOVE_NAMES.append(face + suf)
        MOVES.append(m)
        MOVE_FACE.append(face_idx)

N_MOVES = len(MOVES)            # 18
NAME_TO_IDX = {n: i for i, n in enumerate(MOVE_NAMES)}

# Phase 2: only U, U2, U', D, D2, D', L2, R2, F2, B2
PHASE2_NAMES = ['U', 'U2', "U'", 'D', 'D2', "D'", 'L2', 'R2', 'F2', 'B2']
PHASE2_MOVES = [NAME_TO_IDX[n] for n in PHASE2_NAMES]
N_PHASE2 = len(PHASE2_MOVES)    # 10

# Opposite-face pairs (U/D, R/L, F/B) — moves on opposite faces commute, so we
# canonicalise their order during search to prune duplicates.
OPP_FACE = {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2}


# ─── Sticker (54-list) → CubieCube conversion ────────────────────────────────
# These mappings are tied to the Cube class's sticker indexing in cube.py.
#
# Corner facelets: each tuple is ordered as (U/D facelet, then the next two
# clockwise viewed from outside the corner). The orientation of a corner is the
# index in this tuple where the U/D sticker is found.
CORNER_FACELETS = (
    (17, 27,  2),  # URF: U-facelet, R-facelet, F-facelet
    (15,  0, 20),  # UFL
    ( 9, 18, 47),  # ULB
    (11, 45, 29),  # UBR
    (38,  8, 33),  # DFR: D-facelet, F-facelet, R-facelet
    (36, 26,  6),  # DLF
    (42, 53, 24),  # DBL
    (44, 35, 51),  # DRB
)
CORNER_COLORS = (
    ('U', 'R', 'F'), ('U', 'F', 'L'), ('U', 'L', 'B'), ('U', 'B', 'R'),
    ('D', 'F', 'R'), ('D', 'L', 'F'), ('D', 'B', 'L'), ('D', 'R', 'B'),
)

# Edge facelets: each tuple is (marker facelet, other facelet). The marker
# facelet is on the U/D axis when one is available, otherwise on the F/B axis.
# Edge orientation = 0 iff the marker color of the piece is on the marker
# facelet of the slot. With this convention U, D, L, R turns preserve EO and
# F, B turns flip EO (the standard Kociemba convention).
EDGE_FACELETS = (
    (14, 28),  # UR
    (16,  1),  # UF
    (12, 19),  # UL
    (10, 46),  # UB
    (41, 34),  # DR
    (37,  7),  # DF
    (39, 25),  # DL
    (43, 52),  # DB
    ( 5, 30),  # FR  (F is marker facelet)
    ( 3, 23),  # FL
    (50, 21),  # BL
    (48, 32),  # BR
)
EDGE_COLORS = (
    ('U', 'R'), ('U', 'F'), ('U', 'L'), ('U', 'B'),
    ('D', 'R'), ('D', 'F'), ('D', 'L'), ('D', 'B'),
    ('F', 'R'), ('F', 'L'), ('B', 'L'), ('B', 'R'),
)
# Per piece, the color of its marker sticker.
EDGE_MARKER = ('U', 'U', 'U', 'U', 'D', 'D', 'D', 'D', 'F', 'F', 'B', 'B')


def sticker_to_cubie(state):
    """Convert a 54-element sticker list (Cube.state) into a CubieCube."""
    cp, co = [0]*8, [0]*8
    ep, eo = [0]*12, [0]*12

    for slot in range(8):
        a, b, c = CORNER_FACELETS[slot]
        cs = (state[a], state[b], state[c])
        cs_set = frozenset(cs)
        for piece in range(8):
            if cs_set == frozenset(CORNER_COLORS[piece]):
                cp[slot] = piece
                break
        for k in range(3):
            if cs[k] in ('U', 'D'):
                co[slot] = k
                break

    for slot in range(12):
        a, b = EDGE_FACELETS[slot]
        cs = (state[a], state[b])
        cs_set = frozenset(cs)
        for piece in range(12):
            if cs_set == frozenset(EDGE_COLORS[piece]):
                ep[slot] = piece
                break
        eo[slot] = 0 if state[a] == EDGE_MARKER[ep[slot]] else 1

    return CubieCube(cp, co, ep, eo)


# ─── Coordinate encodings ─────────────────────────────────────────────────────

def encode_eo(eo):
    """Edge orientations → integer in [0, 2048).  eo[11] is parity-determined."""
    n = 0
    for i in range(11):
        n = (n << 1) | eo[i]
    return n


def decode_eo(n):
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = n & 1
        s ^= eo[i]
        n >>= 1
    eo[11] = s
    return eo


def encode_co(co):
    """Corner orientations → integer in [0, 2187).  co[7] is parity-determined."""
    n = 0
    for i in range(7):
        n = n * 3 + co[i]
    return n


def decode_co(n):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = n % 3
        s += co[i]
        n //= 3
    co[7] = (-s) % 3
    return co


# Combinatorial number system for the slice coordinate (which 4 of 12 slots
# hold E-slice edges, ignoring which is which).  We use the standard ascending
# encoding: for E-slice positions p_0 < p_1 < p_2 < p_3 the index is
# C(p_0,1)+C(p_1,2)+C(p_2,3)+C(p_3,4).  Solved (slice edges at 8,9,10,11) maps
# to 8+36+120+330 = 494, not 0; the BFS handles arbitrary goal coords.
SLICE_GOAL = 8 + 36 + 120 + 330  # 494

def encode_slice(ep):
    """Locations of the four E-slice edges (pieces 8-11) → integer in [0, 495)."""
    positions = [i for i in range(12) if ep[i] >= 8]  # ascending
    return sum(math.comb(positions[k], k + 1) for k in range(4))


def decode_slice(n):
    """Return a 12-element list whose entries are 1 where an E-slice edge sits."""
    flag = [0] * 12
    rem = n
    for k in range(3, -1, -1):
        p = k
        while math.comb(p + 1, k + 1) <= rem:
            p += 1
        flag[p] = 1
        rem -= math.comb(p, k + 1)
    return flag


# Phase 2 coordinates.  Lehmer code → factorial number system.

def encode_perm(perm, n):
    """Permutation of n elements (values 0..n-1) → integer in [0, n!)."""
    code = 0
    for i in range(n - 1):
        c = 0
        for j in range(i + 1, n):
            if perm[j] < perm[i]:
                c += 1
        code = code * (n - i) + c
    return code


def decode_perm(code, n):
    perm = [0] * n
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    avail = list(range(n))
    for i in range(n):
        f = fact[n - 1 - i]
        idx = code // f
        code %= f
        perm[i] = avail.pop(idx)
    return perm


def encode_cperm(cp):
    return encode_perm(cp, 8)


def decode_cperm(code):
    return decode_perm(code, 8)


def encode_edge8(ep):
    """Permutation of UD-layer edges (the first eight entries of ep). In G1
    these always contain pieces 0..7, so encode them as a permutation of 8."""
    return encode_perm(ep[:8], 8)


def decode_edge8(code):
    p = decode_perm(code, 8)
    return p + [8, 9, 10, 11]


def encode_eslice(ep):
    """Permutation of the four E-slice edges (ep[8..11] in G1). Maps to [0, 24)."""
    sub = [ep[i] - 8 for i in range(8, 12)]
    return encode_perm(sub, 4)


def decode_eslice(code):
    p = decode_perm(code, 4)
    return [x + 8 for x in p]


# ─── Move tables for each coordinate ──────────────────────────────────────────
# Each table[coord][m] gives the new coord after applying move m.

def _apply_move_to_eo(eo, m):
    return [(eo[m.ep[i]] + m.eo[i]) % 2 for i in range(12)]


def _apply_move_to_co(co, m):
    return [(co[m.cp[i]] + m.co[i]) % 3 for i in range(8)]


def _apply_move_to_slice(flag, m):
    # flag[i] = 1 iff slot i holds an E-slice edge.  After move m, the new
    # slot i pulls its piece from old slot m.ep[i].
    return [flag[m.ep[i]] for i in range(12)]


def _apply_move_to_perm(perm, m_perm):
    return [perm[m_perm[i]] for i in range(len(perm))]


def _build_eo_table():
    print("  building EO move table...", end=" ", flush=True)
    t = [[0]*N_MOVES for _ in range(2048)]
    for c in range(2048):
        eo = decode_eo(c)
        for m in range(N_MOVES):
            t[c][m] = encode_eo(_apply_move_to_eo(eo, MOVES[m]))
    print("done")
    return t


def _build_co_table():
    print("  building CO move table...", end=" ", flush=True)
    t = [[0]*N_MOVES for _ in range(2187)]
    for c in range(2187):
        co = decode_co(c)
        for m in range(N_MOVES):
            t[c][m] = encode_co(_apply_move_to_co(co, MOVES[m]))
    print("done")
    return t


def _build_slice_table():
    print("  building slice move table...", end=" ", flush=True)
    t = [[0]*N_MOVES for _ in range(495)]
    for c in range(495):
        flag = decode_slice(c)
        for m in range(N_MOVES):
            new_flag = _apply_move_to_slice(flag, MOVES[m])
            positions = [i for i in range(12) if new_flag[i]]
            t[c][m] = sum(math.comb(positions[k], k + 1) for k in range(4))
    print("done")
    return t


def _build_cperm_table():
    print("  building corner-perm move table...", end=" ", flush=True)
    t = [[0]*N_PHASE2 for _ in range(40320)]
    for c in range(40320):
        cp = decode_cperm(c)
        for j, m in enumerate(PHASE2_MOVES):
            new_cp = _apply_move_to_perm(cp, MOVES[m].cp)
            t[c][j] = encode_cperm(new_cp)
    print("done")
    return t


def _build_edge8_table():
    print("  building UD-edge perm move table...", end=" ", flush=True)
    t = [[0]*N_PHASE2 for _ in range(40320)]
    for c in range(40320):
        ep = decode_edge8(c)
        for j, m in enumerate(PHASE2_MOVES):
            new_ep = _apply_move_to_perm(ep, MOVES[m].ep)
            t[c][j] = encode_edge8(new_ep)
    print("done")
    return t


def _build_eslice_table():
    print("  building E-slice perm move table...", end=" ", flush=True)
    t = [[0]*N_PHASE2 for _ in range(24)]
    for c in range(24):
        ep_slice = decode_eslice(c)
        # Pad with UD edges so we can apply the full edge permutation.
        ep = [0, 1, 2, 3, 4, 5, 6, 7] + ep_slice
        for j, m in enumerate(PHASE2_MOVES):
            new_ep = _apply_move_to_perm(ep, MOVES[m].ep)
            t[c][j] = encode_eslice(new_ep)
    print("done")
    return t


# ─── Pruning tables (BFS over coordinate pairs) ───────────────────────────────

def _bfs_prune(size_a, size_b, table_a, table_b, n_moves, name,
               goal_a=0, goal_b=0):
    """BFS from the goal state.  Stores min #moves per state."""
    print(f"  building {name} pruning table ({size_a*size_b:,} entries)...",
          end=" ", flush=True)
    t0 = time.time()
    N = size_a * size_b
    dist = bytearray([255]) * N
    goal = goal_a * size_b + goal_b
    dist[goal] = 0
    frontier = [goal]
    depth = 0
    visited = 1
    while frontier:
        depth += 1
        next_frontier = []
        for s in frontier:
            a, b = divmod(s, size_b)
            for m in range(n_moves):
                na = table_a[a][m]
                nb = table_b[b][m]
                ns = na * size_b + nb
                if dist[ns] == 255:
                    dist[ns] = depth
                    next_frontier.append(ns)
        visited += len(next_frontier)
        frontier = next_frontier
        if visited == N:
            break
    print(f"depth {depth}, {time.time()-t0:.1f}s")
    return dist


# ─── Worker plumbing for parallel IDA* ───────────────────────────────────────
# At Solver construction, the parent populates _W_TABLES and forks a worker
# Pool. Forked children inherit _W_TABLES via copy-on-write — no pickling of
# the multi-megabyte move/pruning tables. _W_ABORT is a shared lock-free byte
# flag: when one worker finds a solution, the parent flips it to 1 so the
# other workers' searches can exit early.

_W_TABLES = None
_W_ABORT = None


def _worker_init(abort_value):
    global _W_ABORT
    _W_ABORT = abort_value


# ─── Recursive IDA* (shared by serial and parallel paths) ─────────────────────
# The tables are passed in as locals — Python local-variable access is the
# fastest form of name lookup, which matters in this hot loop.

def _phase1_search(eo, co, sl, depth, last_face, path,
                    eo_t, co_t, slice_t, p1_eo_slice, p1_co_slice,
                    abort=None, counter=None):
    if abort is not None:
        counter[0] += 1
        if counter[0] >= 8192:
            counter[0] = 0
            if abort.value:
                return False
    if depth == 0:
        return eo == 0 and co == 0 and sl == SLICE_GOAL
    h = p1_eo_slice[eo * 495 + sl]
    h2 = p1_co_slice[co * 495 + sl]
    if h2 > h:
        h = h2
    if h > depth:
        return False
    for m in range(N_MOVES):
        f = MOVE_FACE[m]
        if f == last_face:
            continue
        if last_face is not None and OPP_FACE.get(last_face) == f and f < last_face:
            continue
        n_eo = eo_t[eo][m]
        n_co = co_t[co][m]
        n_sl = slice_t[sl][m]
        path.append(m)
        if _phase1_search(n_eo, n_co, n_sl, depth - 1, f, path,
                          eo_t, co_t, slice_t, p1_eo_slice, p1_co_slice,
                          abort, counter):
            return True
        path.pop()
    return False


def _phase2_search(cp, e8, es, depth, last_face, path,
                    cperm_t, edge8_t, eslice_t, p2_cp_es, p2_e8_es,
                    abort=None, counter=None):
    if abort is not None:
        counter[0] += 1
        if counter[0] >= 8192:
            counter[0] = 0
            if abort.value:
                return False
    if depth == 0:
        return cp == 0 and e8 == 0 and es == 0
    h = p2_cp_es[cp * 24 + es]
    h2 = p2_e8_es[e8 * 24 + es]
    if h2 > h:
        h = h2
    if h > depth:
        return False
    for j, m in enumerate(PHASE2_MOVES):
        f = MOVE_FACE[m]
        if f == last_face:
            continue
        if last_face is not None and OPP_FACE.get(last_face) == f and f < last_face:
            continue
        n_cp = cperm_t[cp][j]
        n_e8 = edge8_t[e8][j]
        n_es = eslice_t[es][j]
        path.append(m)
        if _phase2_search(n_cp, n_e8, n_es, depth - 1, f, path,
                          cperm_t, edge8_t, eslice_t, p2_cp_es, p2_e8_es,
                          abort, counter):
            return True
        path.pop()
    return False


# ─── Pool worker entry points ─────────────────────────────────────────────────
# A task is one root branch: the resulting coords after a single first move,
# plus the remaining depth budget. The worker runs IDA* on its branch and
# returns the move list if it found a solution, else None.

def _phase1_worker(task):
    eo, co, sl, depth, last_face, first_move = task
    t = _W_TABLES
    path = [first_move]
    counter = [0]
    if _phase1_search(eo, co, sl, depth, last_face, path,
                      t['eo_t'], t['co_t'], t['slice_t'],
                      t['p1_eo_slice'], t['p1_co_slice'],
                      _W_ABORT, counter):
        return path
    return None


def _phase2_worker(task):
    cp, e8, es, depth, last_face, first_move = task
    t = _W_TABLES
    path = [first_move]
    counter = [0]
    if _phase2_search(cp, e8, es, depth, last_face, path,
                      t['cperm_t'], t['edge8_t'], t['eslice_t'],
                      t['p2_cp_es'], t['p2_e8_es'],
                      _W_ABORT, counter):
        return path
    return None


# ─── Solver ───────────────────────────────────────────────────────────────────

class Solver:
    """Two-phase Kociemba solver. Tables are built once at construction.

    With ``n_workers`` > 1, top-level IDA* branches are searched in parallel
    using a fork-based multiprocessing Pool. Workers inherit the table memory
    via fork's copy-on-write, so there is no per-call serialisation cost.
    """

    _TABLE_KEYS = ('eo_t', 'co_t', 'slice_t', 'cperm_t', 'edge8_t', 'eslice_t',
                   'p1_eo_slice', 'p1_co_slice', 'p2_cp_es', 'p2_e8_es')

    def __init__(self, verbose=True, cache_path=_CACHE_PATH, n_workers=None):
        self.verbose = verbose
        self._pool = None
        self._abort = None

        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f"Loading Kociemba solver tables from {cache_path}...")
            t0 = time.time()
            with open(cache_path, "rb") as f:
                tables = pickle.load(f)
            for k in self._TABLE_KEYS:
                setattr(self, k, tables[k])
            if verbose:
                print(f"Tables loaded in {time.time()-t0:.1f}s.\n")
        else:
            if verbose:
                print("Building Kociemba solver tables...")
            t0 = time.time()
            # Phase 1 move tables
            self.eo_t     = _build_eo_table()
            self.co_t     = _build_co_table()
            self.slice_t  = _build_slice_table()
            # Phase 2 move tables
            self.cperm_t  = _build_cperm_table()
            self.edge8_t  = _build_edge8_table()
            self.eslice_t = _build_eslice_table()
            # Pruning tables
            self.p1_eo_slice = _bfs_prune(2048, 495, self.eo_t, self.slice_t,
                                           N_MOVES, "phase-1 EO+slice",
                                           goal_a=0, goal_b=SLICE_GOAL)
            self.p1_co_slice = _bfs_prune(2187, 495, self.co_t, self.slice_t,
                                           N_MOVES, "phase-1 CO+slice",
                                           goal_a=0, goal_b=SLICE_GOAL)
            self.p2_cp_es    = _bfs_prune(40320, 24, self.cperm_t, self.eslice_t,
                                           N_PHASE2, "phase-2 cperm+eslice")
            self.p2_e8_es    = _bfs_prune(40320, 24, self.edge8_t, self.eslice_t,
                                           N_PHASE2, "phase-2 edge8+eslice")
            if verbose:
                print(f"All tables built in {time.time()-t0:.1f}s.")

            if cache_path:
                if verbose:
                    print(f"Saving tables to {cache_path}...", end=" ", flush=True)
                t0 = time.time()
                tmp = cache_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump({k: getattr(self, k) for k in self._TABLE_KEYS},
                                f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache_path)
                if verbose:
                    print(f"done ({time.time()-t0:.1f}s)\n")

        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers = max(1, int(n_workers))
        if self.n_workers > 1:
            self._setup_pool()

    # ─── Worker pool lifecycle ────────────────────────────────────────────────
    def _setup_pool(self):
        # Make tables visible to forked children before the fork happens.
        global _W_TABLES
        _W_TABLES = {k: getattr(self, k) for k in self._TABLE_KEYS}

        try:
            ctx = mp.get_context('fork')
        except ValueError as e:
            if self.verbose:
                print(f"Fork start method unavailable ({e}); running serial.")
            return

        try:
            # lock=False → RawValue: lock-free reads/writes. We have a single
            # writer (parent) and many readers (workers), and we tolerate a
            # few-microsecond delay before workers see the flip.
            self._abort = ctx.Value('b', 0, lock=False)
            self._pool = ctx.Pool(
                self.n_workers,
                initializer=_worker_init,
                initargs=(self._abort,),
            )
        except Exception as e:
            if self.verbose:
                print(f"Failed to create worker pool ({e}); running serial.")
            self._pool = None
            self._abort = None
            return

        if self.verbose:
            print(f"Parallel solver: {self.n_workers} worker processes.\n")

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        pool = getattr(self, '_pool', None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    # ─── Phase 1 ──────────────────────────────────────────────────────────────
    def _solve_phase1(self, cubie):
        eo = encode_eo(cubie.eo)
        co = encode_co(cubie.co)
        sl = encode_slice(cubie.ep)
        if eo == 0 and co == 0 and sl == SLICE_GOAL:
            return []
        if self._pool is not None:
            return self._phase1_parallel(eo, co, sl)
        for depth in range(1, 13):  # phase-1 optimum is ≤ 12
            path = []
            if _phase1_search(eo, co, sl, depth, None, path,
                              self.eo_t, self.co_t, self.slice_t,
                              self.p1_eo_slice, self.p1_co_slice):
                return path
        raise RuntimeError("phase 1 failed (unsolvable / bad input)")

    def _phase1_parallel(self, eo, co, sl):
        eo_t, co_t, slice_t = self.eo_t, self.co_t, self.slice_t
        for depth in range(1, 13):
            tasks = [
                (eo_t[eo][m], co_t[co][m], slice_t[sl][m],
                 depth - 1, MOVE_FACE[m], m)
                for m in range(N_MOVES)
            ]
            self._abort.value = 0
            result = None
            for r in self._pool.imap_unordered(_phase1_worker, tasks):
                if r is not None and result is None:
                    result = r
                    self._abort.value = 1  # tell siblings to bail
            if result is not None:
                return result
        raise RuntimeError("phase 1 failed (unsolvable / bad input)")

    # ─── Phase 2 ──────────────────────────────────────────────────────────────
    def _solve_phase2(self, cubie):
        cp = encode_cperm(cubie.cp)
        e8 = encode_edge8(cubie.ep)
        es = encode_eslice(cubie.ep)
        if cp == 0 and e8 == 0 and es == 0:
            return []
        if self._pool is not None:
            return self._phase2_parallel(cp, e8, es)
        for depth in range(1, 19):  # phase-2 optimum is ≤ 18
            path = []
            if _phase2_search(cp, e8, es, depth, None, path,
                              self.cperm_t, self.edge8_t, self.eslice_t,
                              self.p2_cp_es, self.p2_e8_es):
                return path
        raise RuntimeError("phase 2 failed (bad input)")

    def _phase2_parallel(self, cp, e8, es):
        cperm_t, edge8_t, eslice_t = self.cperm_t, self.edge8_t, self.eslice_t
        for depth in range(1, 19):
            tasks = [
                (cperm_t[cp][j], edge8_t[e8][j], eslice_t[es][j],
                 depth - 1, MOVE_FACE[m], m)
                for j, m in enumerate(PHASE2_MOVES)
            ]
            self._abort.value = 0
            result = None
            for r in self._pool.imap_unordered(_phase2_worker, tasks):
                if r is not None and result is None:
                    result = r
                    self._abort.value = 1
            if result is not None:
                return result
        raise RuntimeError("phase 2 failed (bad input)")

    # ─── Public API ───────────────────────────────────────────────────────────
    def solve(self, cube):
        """
        Solve a Cube and return the move sequence as a space-separated string.

        The Cube argument is left unchanged; apply the returned string with
        cube.move(...) to verify.
        """
        cubie = sticker_to_cubie(cube.state)

        p1 = self._solve_phase1(cubie)
        # Apply phase-1 moves to cubie state before starting phase 2.
        for m in p1:
            cubie.multiply(MOVES[m])
        p2 = self._solve_phase2(cubie)

        full = p1 + p2
        return " ".join(MOVE_NAMES[m] for m in full)


# ─── Convenience function and self-tests ──────────────────────────────────────

_default_solver = None
def solve(cube):
    """Lazy-init a global Solver and return the move string for the given Cube."""
    global _default_solver
    if _default_solver is None:
        _default_solver = Solver()
    return _default_solver.solve(cube)


if __name__ == "__main__":
    # 1) Sanity-check sticker_to_cubie on the solved state.
    cube = Cube()
    cube.random_scramble()
    # 3) Solve the user's example scramble.
    solver = Solver()

    t0 = time.time()
    solution = solver.solve(cube)
    dt = time.time() - t0
    n_moves = len(solution.split())
    print(f"Solution ({n_moves} moves, {dt:.2f}s): {solution}")

    cube.move(solution)
    final = "".join(cube.state)
    expected = "".join(Cube.solved_state)
    if final == expected:
        print("Cube is solved. ✓")
    else:
        print("Cube is NOT solved!")
        print(cube)