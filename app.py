"""Entry point for the Rubik's cube solver web app.

Run with:
    .venv/bin/python app.py

Then open http://127.0.0.1:5000 in a browser.
"""
import argparse

from backend import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the cube solver web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Solver worker processes (default: all CPU cores). "
             "Set to 1 to disable parallelism.",
    )
    args = parser.parse_args()

    app = create_app(n_workers=args.workers)
    # debug=False so Flask's auto-reloader doesn't fork the solver pool twice.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
