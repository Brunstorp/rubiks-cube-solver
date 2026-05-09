"""HTTP API for the cube solver."""
from flask import Blueprint, current_app, jsonify, request

from cube import Cube

from .cube_state import (
    StateError,
    parse_state,
    trace_animation_states,
    validate_solvability,
    validate_state,
)

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.post("/solve")
def solve():
    """Take a 54-letter cube state, return the solution + animation frames."""
    data = request.get_json(silent=True) or {}
    try:
        state = parse_state(data.get("state"))
        validate_state(state)
        validate_solvability(state)
    except StateError as e:
        return jsonify({"error": str(e)}), 400

    cube = Cube(state=list(state))
    try:
        solution = current_app.config["SOLVER"].solve(cube)
    except RuntimeError as e:
        # The solver raises RuntimeError for unsolvable inputs (parity errors,
        # impossible orientations) — these pass count-and-centre checks but
        # don't correspond to any real scramble.
        return jsonify({"error": f"unsolvable cube: {e}"}), 400

    moves = solution.split() if solution else []
    states = trace_animation_states(state, moves)
    return jsonify({
        "solution": solution,
        "moves": moves,
        "states": states,
        "n_moves": len(moves),
    })


@bp.post("/scramble")
def scramble():
    """Return a random valid scrambled state."""
    data = request.get_json(silent=True) or {}
    length = int(data.get("length", 25))
    length = max(1, min(50, length))
    cube = Cube()
    moves = cube.random_scramble(length=length)
    return jsonify({"state": list(cube.state), "scramble": moves})


@bp.get("/solved")
def solved():
    """Return the solved-cube state."""
    return jsonify({"state": list(Cube.solved_state)})
