"""Helpers to inspect the public EB solver API and curate MCP exposure."""

from __future__ import annotations

import importlib
import inspect
from typing import Dict, Iterable, List

from server.schemas.models import PublicApiEntry


def discover_public_api(module_name: str, wrapped_tool_names: Iterable[str]) -> List[PublicApiEntry]:
    """Inspect public functions and classes from a module."""
    module = importlib.import_module(module_name)
    wrapped = set(wrapped_tool_names)
    exported = getattr(module, "__all__", None)
    names = exported if exported is not None else [name for name in dir(module) if not name.startswith("_")]

    entries: List[PublicApiEntry] = []
    for name in sorted(set(names)):
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue
        if inspect.isfunction(obj):
            kind = "function"
        elif inspect.isclass(obj):
            kind = "class"
        else:
            continue
        doc = inspect.getdoc(obj) or ""
        first_line = doc.strip().splitlines()[0] if doc.strip() else ""
        entries.append(
            PublicApiEntry(
                name=name,
                kind=kind,
                description=first_line,
                module=getattr(obj, "__module__", module_name),
                wrapped_as_tool=name in wrapped,
            )
        )
    return entries


def summarize_api(entries: List[PublicApiEntry]) -> Dict[str, int]:
    """Return a compact summary for tool discovery responses."""
    return {
        "functions": sum(entry.kind == "function" for entry in entries),
        "classes": sum(entry.kind == "class" for entry in entries),
        "wrapped_tools": sum(entry.wrapped_as_tool for entry in entries),
    }

