from __future__ import annotations

import io
import json
import sys
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ldt.operations import execute_operation
from ldt.utils.errors import LibraryError


def _fail(*, code: int, error_type: str, message: str) -> int:
    payload = {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    print(json.dumps(payload), file=sys.stdout)
    return code


def _read_payload() -> Mapping[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("Missing JSON input payload on stdin.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")
    return payload


def main() -> int:
    """Run one bridge operation and print JSON result."""

    try:
        payload = _read_payload()
    except json.JSONDecodeError as exc:
        return _fail(code=2, error_type="json_decode_error", message=str(exc))
    except ValueError as exc:
        return _fail(code=2, error_type="payload_error", message=str(exc))

    operation = payload.get("operation")
    params = payload.get("params", {})
    if not isinstance(operation, str) or not operation.strip():
        return _fail(
            code=2,
            error_type="payload_error",
            message="`operation` must be a non-empty string.",
        )
    if not isinstance(params, dict):
        return _fail(
            code=2,
            error_type="payload_error",
            message="`params` must be an object.",
        )

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = execute_operation(operation.strip(), params)
    except LibraryError as exc:
        return _fail(code=3, error_type="library_error", message=str(exc))
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _fail(code=1, error_type="internal_error", message=str(exc))

    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
