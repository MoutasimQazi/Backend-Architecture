"""WSGI entrypoint for cPanel / Phusion Passenger.

Passenger speaks WSGI; FastAPI is ASGI. Handing Passenger the FastAPI object
directly does not work — it is not a WSGI callable — so it is wrapped with
`a2wsgi.ASGIMiddleware`, which runs the ASGI app on an event loop inside the
worker thread.

Every request path in this app is synchronous underneath (SQLAlchemy with
PyMySQL), so nothing is lost by the bridge. The one thing it cannot do is
server-sent events: WSGI has no streaming contract Passenger honours. When
streaming is needed, run uvicorn on a port and reverse-proxy to it instead.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# cPanel's "Setup Python App" activates a virtualenv for the Passenger process,
# but a manually created one needs to be put on the path here.
_VENV = os.environ.get("APP_VENV") or os.path.join(ROOT, "venv")
_SITE_PACKAGES = os.path.join(
    _VENV, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
)
if os.path.isdir(_SITE_PACKAGES) and _SITE_PACKAGES not in sys.path:
    sys.path.insert(0, _SITE_PACKAGES)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from apps.api.main import app as _asgi_app  # noqa: E402

try:
    from a2wsgi import ASGIMiddleware
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "a2wsgi is required to serve the ASGI app under Passenger. "
        "Install it with: pip install a2wsgi"
    ) from exc

application = ASGIMiddleware(_asgi_app)
