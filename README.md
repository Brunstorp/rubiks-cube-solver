# Rubik's Cube Solver

A personal project to build a Rubik's Cube solver from scratch.

## Status

Working: Kociemba two-phase solver with cached pruning tables and parallel
IDA* search. Web UI with a 3D editable cube and step-by-step solution
playback.

## Run the web app

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open http://127.0.0.1:5000.

## Layout

- `cube.py` — sticker-level cube model
- `cube_solver.py` — Kociemba two-phase solver with parallel IDA*
- `backend/` — Flask app (routes, validation, state conversion)
- `app.py` — server entry point
- `templates/index.html`, `static/` — Three.js front-end

## Author

[Brunstorp](https://github.com/Brunstorp)
