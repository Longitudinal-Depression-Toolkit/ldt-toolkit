from __future__ import annotations

import io
import json
import sys
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import ldt.bridge.runtime_warning_filters as _runtime_warning_filters  # noqa: F401
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


def _is_ping_request() -> bool:
    if len(sys.argv) <= 1:
        return False
    return sys.argv[1].strip().lower() in {"--ping", "ping", "--health"}


def _emit_ping() -> int:
    payload = {
        "ok": True,
        "result": {
            "bridge": "ldt-bridge",
            "status": "ok",
        },
    }
    print(json.dumps(payload), file=sys.stdout)
    return 0


def main() -> int:
    """Run one bridge operation and print JSON result."""

    if _is_ping_request():
        return _emit_ping()

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
        from ldt.bridge.operations import execute_operation
    except Exception as exc:  # pragma: no cover - defensive import boundary
        return _fail(
            code=3,
            error_type="library_error",
            message=f"Failed to import bridge operations: {exc}",
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
