"""HTTP API for the cube solver."""
import numpy as np
import cv2
from flask import Blueprint, current_app, jsonify, request

from cube import Cube
from cube_classifier import (
    FACE_ORDER,
    classify_cube_from_images_with_uncertain,
    detect_stickers,
)

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


@bp.post("/classify")
def classify():
    """Take 6 face photos (multipart fields F, U, L, R, D, B) and return the
    inferred 54-letter cube state."""
    images = {}
    for face in FACE_ORDER:
        f = request.files.get(face)
        if f is None:
            return jsonify({"error": f"missing photo for face {face}"}), 400
        data = f.read()
        if not data:
            return jsonify({"error": f"empty photo for face {face}"}), 400
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": f"could not decode photo for face {face}"}), 400
        images[face] = bgr

    try:
        state, uncertain, warning = classify_cube_from_images_with_uncertain(images)
    except Exception as e:
        # Hard failure (≥3 photos failed cube detection). Best-effort isn't
        # useful; the user has to retake.
        return jsonify({"error": f"classification failed: {e}"}), 400

    return jsonify({
        "state": list(state),
        "uncertain": uncertain,
        "warning": warning,  # null when everything looked clean
    })


@bp.post("/detect_face")
def detect_face():
    """Run per-sticker contour detection on a single frame and return the 9
    detected stickers (or detected=False if the frame isn't usable yet).

    Used by the live-scan wizard to give the user real-time feedback while
    they line up each face. Expects a multipart "frame" field carrying a
    JPEG/PNG image, and an optional "image_width" / "image_height" so the
    frontend can map detection coords back to the displayed video size."""
    f = request.files.get("frame")
    if f is None:
        return jsonify({"error": "missing frame"}), 400
    data = f.read()
    if not data:
        return jsonify({"error": "empty frame"}), 400
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "could not decode frame"}), 400

    stickers = detect_stickers(bgr)
    if stickers is None:
        return jsonify({
            "detected": False,
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
        })
    return jsonify({
        "detected": True,
        "stickers": stickers,
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
    })
