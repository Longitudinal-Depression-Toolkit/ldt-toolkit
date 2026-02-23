from __future__ import annotations

import importlib
from typing import Any

import sklearn


def import_skrub_symbol(symbol_name: str) -> Any:
    """Import one symbol from `skrub` with temporary sklearn-version patching.

    Some environments use a development sklearn build string that causes
    `skrub` import guards to fail. This helper temporarily rewrites the version
    string only during import, then restores the original value.

    Args:
        symbol_name (str): Name of the attribute to fetch from `skrub`.

    Returns:
        Any: Imported symbol from the `skrub` module.
    """
    original_version = getattr(sklearn, "__version__", "")
    needs_patch = original_version.startswith("1.6.dev")
    if needs_patch:
        sklearn.__version__ = "1.5.2"
    try:
        skrub_module = importlib.import_module("skrub")
    finally:
        if needs_patch:
            sklearn.__version__ = original_version
    return getattr(skrub_module, symbol_name)
