"""Minimal MCP stdio protocol support for tools/list and tools/call.

The MCP stdio transport sends one JSON-RPC message per line
(newline-delimited JSON).  For compatibility with older LSP-style clients,
messages framed with a ``Content-Length`` header are also accepted; replies
use the same framing as the most recent request.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


class McpProtocolError(RuntimeError):
    """Raised for malformed incoming MCP frames."""


@dataclass
class JsonRpcRequest:
    raw: Dict[str, Any]

    @property
    def id(self) -> Any:
        return self.raw.get("id")

    @property
    def method(self) -> str:
        method = self.raw.get("method")
        if not isinstance(method, str):
            raise McpProtocolError("Request is missing a string method")
        return method

    @property
    def params(self) -> Dict[str, Any]:
        params = self.raw.get("params", {})
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise McpProtocolError("Request params must be an object")
        return params


_FRAMING = {"mode": "ndjson"}   # "ndjson" (MCP standard) or "header"


def read_message(stdin_buffer) -> Optional[Dict[str, Any]]:
    """Read one message from stdin (newline-delimited JSON or Content-Length framed)."""
    headers: Dict[str, str] = {}
    while True:
        line = stdin_buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            if headers:
                break
            continue            # skip blank lines between messages
        if not headers and line.lstrip().startswith(b"{"):
            _FRAMING["mode"] = "ndjson"
            return json.loads(line.decode("utf-8"))
        key, _, value = line.decode("utf-8").partition(":")
        headers[key.strip().lower()] = value.strip()
    _FRAMING["mode"] = "header"
    if "content-length" not in headers:
        raise McpProtocolError("Missing Content-Length header")
    length = int(headers["content-length"])
    body = stdin_buffer.read(length)
    if len(body) != length:
        raise McpProtocolError("Unexpected EOF while reading message body")
    return json.loads(body.decode("utf-8"))


def write_message(payload: Dict[str, Any], stdout_buffer) -> None:
    """Write one message to stdout using the framing of the last request."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if _FRAMING["mode"] == "header":
        stdout_buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        stdout_buffer.write(body)
    else:
        stdout_buffer.write(body + b"\n")
    stdout_buffer.flush()


def make_result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def send_notification(method: str, params: Dict[str, Any]) -> None:
    write_message({"jsonrpc": "2.0", "method": method, "params": params}, sys.stdout.buffer)
