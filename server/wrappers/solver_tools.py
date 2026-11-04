"""Curated MCP tools wrapping high-value EB AFM solver workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from eb_cr_afm import (
    compute_2d_spectra,
    default_params,
    default_params_multi75,
    frequency_sweep,
    sho_analysis,
    solve_single_frequency,
)

from server.schemas.models import (
    DiscoveryRequest,
    FrequencySweepAnalysisRequest,
    SingleFrequencyRequest,
    SpectraMapRequest,
)
from server.utils.discovery import discover_public_api, summarize_api
from server.utils.serialization import array_bundle, json_safe, write_npz_payload


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


def _preset_payload() -> Dict[str, Any]:
    presets = {}
    for name, factory in {
        "bradler_table_i": default_params,
        "multi75_nominal": default_params_multi75,
    }.items():
        cant, contact, exc, opts = factory()
        presets[name] = {
            "cantilever": json_safe(cant.__dict__),
            "contact": json_safe(contact.__dict__),
            "excitation": json_safe({
                **exc.__dict__,
                "f0": exc.f0,
            }),
            "solver_options": json_safe(opts.__dict__),
        }
    return presets


def _compute_shape_factor_safe(result: Dict[str, Any], cantilever) -> float | None:
    """Compute the dynamic shape factor with NumPy 1.x/2.x compatibility."""
    x = np.asarray(result["x"])
    z = np.asarray(result["z"])
    zp = np.asarray(result["zp"])
    if x.size == 0:
        return None
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    z_sq_mean = integrate(np.abs(z) ** 2, x) / cantilever.L
    if np.abs(z_sq_mean) < 1e-60:
        return None
    s_static = 3.0 / (2.0 * cantilever.L * np.cos(cantilever.phi))
    return float((1.0 / s_static) * np.abs(z[-1]) * np.abs(zp[-1]) / z_sq_mean)


def _handle_list_public_api(arguments: Dict[str, Any]) -> Dict[str, Any]:
    request = DiscoveryRequest.from_dict(arguments)
    wrapped_names = {
        "list_public_api",
        "solve_frequency_response",
        "analyze_frequency_sweep",
        "compute_spatial_spectra_map",
    }
    entries = discover_public_api(request.module, wrapped_names)
    return {
        "module": request.module,
        "summary": summarize_api(entries),
        "presets": _preset_payload(),
        "entries": [entry.__dict__ for entry in entries],
    }


def _handle_single_frequency(arguments: Dict[str, Any]) -> Dict[str, Any]:
    request = SingleFrequencyRequest.from_dict(arguments)
    result = solve_single_frequency(
        request.frequency_hz,
        request.cantilever.to_package(),
        request.contact.to_package(),
        request.excitation.to_package(),
        request.solver_options.to_package(),
        request.observation_position_m,
    )
    return {
        "frequency_hz": request.frequency_hz,
        "observation_position_m": request.observation_position_m,
        "observables": {
            "amplitude_displacement_m": result["A_disp"],
            "phase_displacement_rad": result["phase_disp"],
            "amplitude_slope_rad": result["A_slope"],
            "phase_slope_rad": result["phase_slope"],
        },
        "tip_state": {
            "z_tip": json_safe(result["z"][-1]),
            "zp_tip": json_safe(result["zp"][-1]),
        },
        "profiles": {
            "x_m": array_bundle("x_m", result["x"], request_to_output_inline()),
            "z_complex": array_bundle("z_complex", result["z"], request_to_output_inline()),
            "zp_complex": array_bundle("zp_complex", result["zp"], request_to_output_inline()),
        },
        "diagnostics": json_safe(result.get("analytical_bc_diagnostics", {})),
    }


def request_to_output_inline():
    from server.schemas.models import ArrayOutputConfig

    return ArrayOutputConfig(mode="inline", include_values=True)


def _handle_frequency_sweep(arguments: Dict[str, Any]) -> Dict[str, Any]:
    request = FrequencySweepAnalysisRequest.from_dict(arguments)
    cantilever = request.cantilever.to_package()
    sweep = frequency_sweep(
        cantilever,
        request.contact.to_package(),
        request.excitation.to_package(),
        request.solver_options.to_package(),
        request.observation_position_m,
    )
    modes = sho_analysis(
        sweep,
        prominence_frac=request.prominence_fraction,
        use_phase=request.use_phase,
    )
    dominant_shape_factor = None
    if sweep["results"]:
        dominant_index = int(np.argmax(sweep["amplitudes"]))
        dominant_shape_factor = _compute_shape_factor_safe(sweep["results"][dominant_index], cantilever)
    return {
        "frequency_window_hz": {
            "start": request.solver_options.f_start,
            "stop": request.solver_options.f_stop,
            "step": request.solver_options.f_step,
        },
        "dominant_mode_count": len(modes),
        "modes": json_safe(modes),
        "shape_factor_at_peak": dominant_shape_factor,
        "sweep": {
            "frequencies_hz": array_bundle("frequencies_hz", sweep["freqs"], request_to_output_inline()),
            "amplitudes_slope_rad": array_bundle("amplitudes_slope_rad", sweep["amplitudes_slope"], request_to_output_inline()),
            "phases_slope_deg": array_bundle("phases_slope_deg", sweep["phases_slope"], request_to_output_inline()),
            "amplitudes_displacement_m": array_bundle("amplitudes_displacement_m", sweep["amplitudes_disp"], request_to_output_inline()),
            "phases_displacement_deg": array_bundle("phases_displacement_deg", sweep["phases_disp"], request_to_output_inline()),
        },
        "workflow_steps": [
            "Validated nested parameter objects.",
            "Computed the driven frequency sweep.",
            "Detected and fit resonance peaks with the SHO model.",
            "Computed a dynamic shape factor at the strongest response point.",
        ],
    }


def _handle_spatial_spectra(arguments: Dict[str, Any]) -> Dict[str, Any]:
    request = SpectraMapRequest.from_dict(arguments)
    spectra = compute_2d_spectra(
        request.cantilever.to_package(),
        request.contact.to_package(),
        request.excitation.to_package(),
        request.solver_options.to_package(),
        n_x_points=request.n_x_points,
    )
    arrays = {
        "freqs_hz": np.asarray(spectra["freqs"]),
        "x_m": np.asarray(spectra["x_grid"]),
        "z_amp": np.asarray(spectra["z_amp"]),
        "z_phase_deg": np.asarray(spectra["z_phase"]),
        "zp_amp": np.asarray(spectra["zp_amp"]),
        "zp_phase_deg": np.asarray(spectra["zp_phase"]),
        "tip_amp": np.asarray(spectra["tip_amp"]),
        "tip_phase_deg": np.asarray(spectra["tip_phase"]),
    }

    response = {
        "grid": {
            "n_x_points": request.n_x_points,
            "n_frequency_points": int(arrays["freqs_hz"].shape[0]),
        },
        "arrays": {
            key: array_bundle(key, value, request.output)
            for key, value in arrays.items()
        },
    }
    if request.output.mode == "file":
        if not request.output.file_path:
            raise ValueError("output.file_path is required when output.mode='file'")
        response["output_file"] = write_npz_payload(request.output.file_path, arrays)
    return response


def build_tools() -> List[ToolSpec]:
    """Return the curated MCP tool list."""
    return [
        ToolSpec(
            name="list_public_api",
            description="Inspect the public eb_cr_afm API and return curated defaults for scientific workflows.",
            input_schema=DiscoveryRequest.json_schema(),
            output_schema={
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "summary": {"type": "object"},
                    "presets": {"type": "object"},
                    "entries": {"type": "array"},
                },
            },
            handler=_handle_list_public_api,
        ),
        ToolSpec(
            name="solve_frequency_response",
            description="Solve the cantilever response at one drive frequency and return tip observables plus spatial profiles.",
            input_schema=SingleFrequencyRequest.json_schema(),
            output_schema={
                "type": "object",
                "properties": {
                    "frequency_hz": {"type": "number"},
                    "observables": {"type": "object"},
                    "tip_state": {"type": "object"},
                    "profiles": {"type": "object"},
                    "diagnostics": {"type": "object"},
                },
            },
            handler=_handle_single_frequency,
        ),
        ToolSpec(
            name="analyze_frequency_sweep",
            description="Run a frequency sweep, fit resonance peaks, and summarize the contact-resonance workflow.",
            input_schema=FrequencySweepAnalysisRequest.json_schema(),
            output_schema={
                "type": "object",
                "properties": {
                    "frequency_window_hz": {"type": "object"},
                    "dominant_mode_count": {"type": "integer"},
                    "modes": {"type": "array"},
                    "shape_factor_at_peak": {"type": ["number", "null"]},
                    "sweep": {"type": "object"},
                    "workflow_steps": {"type": "array", "items": {"type": "string"}},
                },
            },
            handler=_handle_frequency_sweep,
        ),
        ToolSpec(
            name="compute_spatial_spectra_map",
            description="Compute 2D displacement and slope spectra across beam position and frequency, optionally writing arrays to an NPZ file.",
            input_schema=SpectraMapRequest.json_schema(),
            output_schema={
                "type": "object",
                "properties": {
                    "grid": {"type": "object"},
                    "arrays": {"type": "object"},
                    "output_file": {"type": "string"},
                },
            },
            handler=_handle_spatial_spectra,
        ),
    ]
