import os
import sys


def pytest_configure():
    # Ensure `src/` is on sys.path for src-layout imports.
    root = os.path.dirname(os.path.dirname(__file__))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
