"""Tests for the MCP wrapper layer around eb_cr_afm."""

from __future__ import annotations

import shutil
from pathlib import Path
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from server.main import handle_request
from server.schemas.models import SingleFrequencyRequest, ValidationError
from server.tool_registry import ToolRegistry


def test_single_frequency_schema_rejects_missing_frequency():
    try:
        SingleFrequencyRequest.from_dict({})
    except ValidationError as exc:
        assert "arguments.frequency_hz" in str(exc)
    else:
        raise AssertionError("ValidationError was not raised")


def test_registry_lists_curated_tools():
    registry = ToolRegistry.create()
    tool_names = {tool["name"] for tool in registry.list_tools()}
    assert {
        "list_public_api",
        "solve_frequency_response",
        "analyze_frequency_sweep",
        "compute_spatial_spectra_map",
    }.issubset(tool_names)


def test_tools_call_returns_structured_content():
    registry = ToolRegistry.create()
    response = handle_request(
        registry,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_frequency_sweep",
                "arguments": {
                    "solver_options": {
                        "n_points": 64,
                        "f_start": 330000.0,
                        "f_stop": 360000.0,
                        "f_step": 10000.0,
                        "method": "analytical",
                    }
                },
            },
        },
    )
    assert response is not None
    result = response["result"]
    assert result["structuredContent"]["dominant_mode_count"] >= 1
    assert result["isError"] is False


def test_spatial_spectra_map_can_write_npz():
    registry = ToolRegistry.create()
    tmpdir = Path("tests") / "_tmp_mcp_outputs"
    tmpdir.mkdir(exist_ok=True)
    target = tmpdir / "spectra_map.npz"
    try:
        result = registry.call_tool(
            "compute_spatial_spectra_map",
            {
                "solver_options": {
                    "n_points": 48,
                    "f_start": 330000.0,
                    "f_stop": 340000.0,
                    "f_step": 10000.0,
                    "method": "analytical",
                },
                "n_x_points": 16,
                "output": {
                    "mode": "file",
                    "file_path": str(target),
                    "include_values": False,
                },
            },
        )
        saved = Path(result["output_file"])
        assert saved.exists()
        with np.load(saved) as payload:
            assert "z_amp" in payload
            assert payload["z_amp"].shape[0] == 16
    finally:
        if target.exists():
            target.unlink()
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
