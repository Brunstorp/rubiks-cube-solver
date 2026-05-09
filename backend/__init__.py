"""Flask backend for the Rubik's cube solver web app."""
import os

from flask import Flask, render_template

from cube_solver import Solver


def create_app(n_workers=None):
    """Build the Flask app and the singleton Solver."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(root, "templates"),
        static_folder=os.path.join(root, "static"),
    )

    # Build the solver once, at startup. Tables load from the cached pickle.
    app.config["SOLVER"] = Solver(verbose=False, n_workers=n_workers)

    @app.get("/")
    def index():
        return render_template("index.html")

    from . import routes
    app.register_blueprint(routes.bp)

    return app
