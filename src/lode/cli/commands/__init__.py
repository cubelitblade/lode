"""lode CLI command modules.

Each command lives in its own module (``survey.py``, ``mine.py``, ...) and
registers itself on the shared ``app`` via a ``register`` function. Shared
command-layer plumbing (store opening, error exits, model gate, progress)
lives in ``_common``.
"""
