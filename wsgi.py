"""WSGI entry point — catches import errors and shows diagnostics."""
import sys
import traceback

try:
    from app import app
except Exception:
    # App failed to import — serve a diagnostic page instead
    _tb = traceback.format_exc()
    print(f"FATAL: app import failed:\n{_tb}", file=sys.stderr)

    from flask import Flask
    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _diag(path):
        return (
            f"<html><body style='background:#1a0f0a;color:#f5deb3;font-family:monospace;padding:40px;'>"
            f"<h1>Big Island — Startup Error</h1>"
            f"<pre style='white-space:pre-wrap;color:#f87171;'>{_tb}</pre>"
            f"<p>Python {sys.version}</p>"
            f"</body></html>"
        ), 500
