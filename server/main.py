"""Entry point for the EB contact-resonance AFM MCP server."""

from __future__ import annotations

import traceback
from typing import Any, Dict

from server.schemas.models import ValidationError
from server.tool_registry import ToolRegistry
from server.utils.logging_utils import configure_logging
from server.utils.mcp_protocol import (
    JsonRpcRequest,
    make_error,
    make_result,
    read_message,
    write_message,
)


SERVER_INFO = {"name": "eb-cr-afm-mcp", "version": "0.1.0"}


def _tool_content(payload: Dict[str, Any], is_error: bool = False) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": "Tool execution completed." if not is_error else "Tool execution failed.",
            }
        ],
        "structuredContent": payload,
        "isError": is_error,
    }


def handle_request(registry: ToolRegistry, request_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    request = JsonRpcRequest(request_payload)
    params = request.params

    if request.method == "initialize":
        return make_result(
            request.id,
            {
                "protocolVersion": params.get("protocolVersion", "2025-03-26"),
                "serverInfo": SERVER_INFO,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "logging": {},
                },
            },
        )

    if request.method == "notifications/initialized":
        return None

    if request.method == "ping":
        return make_result(request.id, {})

    if request.method == "tools/list":
        return make_result(request.id, {"tools": registry.list_tools()})

    if request.method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str):
            return make_error(request.id, -32602, "tools/call requires a string tool name")
        try:
            result = registry.call_tool(name, arguments)
            return make_result(request.id, _tool_content(result))
        except ValidationError as exc:
            return make_result(request.id, _tool_content({"error": str(exc)}, is_error=True))
        except KeyError as exc:
            return make_error(request.id, -32601, str(exc))
        except Exception as exc:
            return make_result(request.id, _tool_content({"error": str(exc)}, is_error=True))

    return make_error(request.id, -32601, f"Method not found: {request.method}")


def main() -> None:
    logger = configure_logging()
    registry = ToolRegistry.create()
    stdin = __import__("sys").stdin.buffer
    stdout = __import__("sys").stdout.buffer

    while True:
        try:
            payload = read_message(stdin)
            if payload is None:
                break
            response = handle_request(registry, payload)
            if response is not None:
                write_message(response, stdout)
        except Exception as exc:
            logger.error("Unhandled MCP server error: %s", exc)
            logger.debug("%s", traceback.format_exc())
            fallback = make_error(None, -32603, str(exc))
            write_message(fallback, stdout)


if __name__ == "__main__":
    main()
