"""Validate and convert cube states between the wire format and Cube.state."""
from collections import Counter

from cube import Cube

VALID_LETTERS = set("FULRDB")

# Centers are immovable on a real cube, so each face's center sticker is
# always the face's color. The indices below are taken from cube.py's layout.
CENTER_INDEX = {
    1: 4,    # F center  (face F, sticker index 4 within F)
    2: 13,   # U center  (9 + 4)
    3: 22,   # L center  (18 + 4)
    4: 31,   # R center  (27 + 4)
    5: 40,   # D center  (36 + 4)
    6: 49,   # B center  (45 + 4)
}
CENTER_COLOR = {4: 'F', 13: 'U', 22: 'L', 31: 'R', 40: 'D', 49: 'B'}


class StateError(ValueError):
    """Raised when an incoming cube state fails validation."""


def parse_state(payload):
    """Coerce a JSON payload into a 54-element list of single-letter colors."""
    if not isinstance(payload, list) or len(payload) != 54:
        raise StateError("state must be a list of exactly 54 entries")
    out = []
    for i, x in enumerate(payload):
        if not isinstance(x, str) or x not in VALID_LETTERS:
            raise StateError(f"sticker {i}: invalid value {x!r}")
        out.append(x)
    return out


def validate_state(state):
    """Sanity-check counts and centers. Does not check cube solvability."""
    counts = Counter(state)
    for c in "FULRDB":
        if counts.get(c, 0) != 9:
            raise StateError(
                f"colour {c}: expected 9 stickers, got {counts.get(c, 0)}"
            )
    for idx, expected in CENTER_COLOR.items():
        if state[idx] != expected:
            raise StateError(
                f"centre at index {idx} must be {expected}, got {state[idx]}"
            )


def trace_animation_states(initial_state, moves):
    """Apply moves one at a time, returning the cube state after each step.

    The return value has length len(moves)+1: the initial state followed by
    the state after every applied move. The frontend uses this to step
    through the solution.
    """
    states = [list(initial_state)]
    cube = Cube(state=list(initial_state))
    for m in moves:
        cube.move(m)
        states.append(list(cube.state))
    return states
