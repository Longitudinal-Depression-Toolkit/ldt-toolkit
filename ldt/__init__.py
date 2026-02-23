from __future__ import annotations

from pathlib import Path

# Expose modules that currently live under `src/` through the public `ldt` package.
_package_root = Path(__file__).resolve().parent
_src_root = _package_root.parent / "src"
if _src_root.is_dir():
    __path__.append(str(_src_root))

import ldt.utils.runtime_warning_filters as _runtime_warning_filters  # noqa: E402,F401
