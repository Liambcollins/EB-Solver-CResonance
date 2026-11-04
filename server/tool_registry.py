"""Tool registry and MCP-facing tool execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from server.schemas.models import ValidationError
from server.wrappers import build_tools


@dataclass
class ToolRegistry:
    tools: Dict[str, Any]

    @classmethod
    def create(cls) -> "ToolRegistry":
        return cls({tool.name: tool for tool in build_tools()})

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema,
            }
            for tool in self.tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any] | None) -> Dict[str, Any]:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        tool = self.tools[name]
        try:
            return tool.handler(arguments or {})
        except ValidationError:
            raise
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

