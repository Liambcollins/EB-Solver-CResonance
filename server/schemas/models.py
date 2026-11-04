"""Lightweight validation and schema models for MCP-exposed tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from eb_cr_afm import (
    CantileverParams,
    ContactParams,
    ExcitationParams,
    SolverOptions,
)


class ValidationError(ValueError):
    """Raised when tool inputs do not satisfy the declared schema."""


def _expect_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def _expect_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{path} must be a string")
    return value


def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{path} must be a boolean")
    return value


def _expect_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path} must be a number")
    return float(value)


def _expect_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{path} must be an integer")
    return value


def _expect_optional_number(payload: Dict[str, Any], key: str, default: Optional[float], path: str) -> Optional[float]:
    if key not in payload:
        return default
    if payload[key] is None:
        return None
    return _expect_number(payload[key], f"{path}.{key}")


def _expect_optional_string(payload: Dict[str, Any], key: str, default: Optional[str], path: str) -> Optional[str]:
    if key not in payload:
        return default
    if payload[key] is None:
        return None
    return _expect_string(payload[key], f"{path}.{key}")


def _expect_literal(payload: Dict[str, Any], key: str, allowed: List[str], default: str, path: str) -> str:
    value = _expect_optional_string(payload, key, default, path)
    assert value is not None
    if value not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValidationError(f"{path}.{key} must be one of {allowed_text}")
    return value


@dataclass(frozen=True)
class ComplexNumberModel:
    """JSON-safe complex number representation."""

    real: float = 0.0
    imag: float = 0.0

    @classmethod
    def from_any(cls, value: Any, path: str) -> "ComplexNumberModel":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(real=float(value), imag=0.0)
        if isinstance(value, dict):
            return cls(
                real=_expect_number(value.get("real", 0.0), f"{path}.real"),
                imag=_expect_number(value.get("imag", 0.0), f"{path}.imag"),
            )
        raise ValidationError(f"{path} must be a number or an object with real/imag fields")

    def to_complex(self) -> complex:
        return complex(self.real, self.imag)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "oneOf": [
                {"type": "number"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "real": {"type": "number", "default": 0.0},
                        "imag": {"type": "number", "default": 0.0},
                    },
                },
            ]
        }


@dataclass(frozen=True)
class CantileverModel:
    L: float = 240e-6
    w: float = 40e-6
    H: float = 6e-6
    k: float = 3.0
    m: float = 5e-11
    mT: float = 5e-12
    phi: float = 0.2

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], path: str = "cantilever") -> "CantileverModel":
        payload = _expect_mapping(payload, path)
        return cls(
            L=_expect_number(payload.get("L", cls.L), f"{path}.L"),
            w=_expect_number(payload.get("w", cls.w), f"{path}.w"),
            H=_expect_number(payload.get("H", cls.H), f"{path}.H"),
            k=_expect_number(payload.get("k", cls.k), f"{path}.k"),
            m=_expect_number(payload.get("m", cls.m), f"{path}.m"),
            mT=_expect_number(payload.get("mT", cls.mT), f"{path}.mT"),
            phi=_expect_number(payload.get("phi", cls.phi), f"{path}.phi"),
        )

    def to_package(self) -> CantileverParams:
        return CantileverParams(**self.__dict__)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "L": {"type": "number", "default": 240e-6},
                "w": {"type": "number", "default": 40e-6},
                "H": {"type": "number", "default": 6e-6},
                "k": {"type": "number", "default": 3.0},
                "m": {"type": "number", "default": 5e-11},
                "mT": {"type": "number", "default": 5e-12},
                "phi": {"type": "number", "default": 0.2},
            },
        }


@dataclass(frozen=True)
class ContactModel:
    k1: float = 1000.0
    k2: float = 850.0
    c1: float = 20e-6
    c2: float = 17e-6
    psi: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], path: str = "contact") -> "ContactModel":
        payload = _expect_mapping(payload, path)
        return cls(
            k1=_expect_number(payload.get("k1", cls.k1), f"{path}.k1"),
            k2=_expect_number(payload.get("k2", cls.k2), f"{path}.k2"),
            c1=_expect_number(payload.get("c1", cls.c1), f"{path}.c1"),
            c2=_expect_number(payload.get("c2", cls.c2), f"{path}.c2"),
            psi=_expect_number(payload.get("psi", cls.psi), f"{path}.psi"),
        )

    def to_package(self) -> ContactParams:
        return ContactParams(**self.__dict__)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "k1": {"type": "number", "default": 1000.0},
                "k2": {"type": "number", "default": 850.0},
                "c1": {"type": "number", "default": 20e-6},
                "c2": {"type": "number", "default": 17e-6},
                "psi": {"type": "number", "default": 0.0},
            },
        }


@dataclass(frozen=True)
class ExcitationModel:
    d1: float = 1e-12
    d2: float = 0.0
    f0: ComplexNumberModel = ComplexNumberModel(real=100e-12, imag=0.0)
    delta_V: float = 1.0
    V_AC: float = 1.0
    eps_r: float = 1.0
    use_distributed_load: bool = True
    use_f0: bool = True
    bias_coupled_f0: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], path: str = "excitation") -> "ExcitationModel":
        payload = _expect_mapping(payload, path)
        f0_raw = payload.get("f0", {"real": cls.f0.real, "imag": cls.f0.imag})
        return cls(
            d1=_expect_number(payload.get("d1", cls.d1), f"{path}.d1"),
            d2=_expect_number(payload.get("d2", cls.d2), f"{path}.d2"),
            f0=ComplexNumberModel.from_any(f0_raw, f"{path}.f0"),
            delta_V=_expect_number(payload.get("delta_V", cls.delta_V), f"{path}.delta_V"),
            V_AC=_expect_number(payload.get("V_AC", cls.V_AC), f"{path}.V_AC"),
            eps_r=_expect_number(payload.get("eps_r", cls.eps_r), f"{path}.eps_r"),
            use_distributed_load=_expect_bool(payload.get("use_distributed_load", cls.use_distributed_load), f"{path}.use_distributed_load"),
            use_f0=_expect_bool(payload.get("use_f0", cls.use_f0), f"{path}.use_f0"),
            bias_coupled_f0=_expect_bool(payload.get("bias_coupled_f0", cls.bias_coupled_f0), f"{path}.bias_coupled_f0"),
        )

    def to_package(self) -> ExcitationParams:
        payload = dict(self.__dict__)
        payload["f0"] = self.f0.to_complex()
        return ExcitationParams(**payload)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "d1": {"type": "number", "default": 1e-12},
                "d2": {"type": "number", "default": 0.0},
                "f0": ComplexNumberModel.json_schema(),
                "delta_V": {"type": "number", "default": 1.0},
                "V_AC": {"type": "number", "default": 1.0},
                "eps_r": {"type": "number", "default": 1.0},
                "use_distributed_load": {"type": "boolean", "default": True},
                "use_f0": {"type": "boolean", "default": True},
                "bias_coupled_f0": {"type": "boolean", "default": False},
            },
        }


@dataclass(frozen=True)
class SolverOptionsModel:
    n_points: int = 300
    f_start: float = 300e3
    f_stop: float = 400e3
    f_step: float = 500.0
    g: float = 20.0
    method: str = "analytical"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], path: str = "solver_options") -> "SolverOptionsModel":
        payload = _expect_mapping(payload, path)
        return cls(
            n_points=_expect_int(payload.get("n_points", cls.n_points), f"{path}.n_points"),
            f_start=_expect_number(payload.get("f_start", cls.f_start), f"{path}.f_start"),
            f_stop=_expect_number(payload.get("f_stop", cls.f_stop), f"{path}.f_stop"),
            f_step=_expect_number(payload.get("f_step", cls.f_step), f"{path}.f_step"),
            g=_expect_number(payload.get("g", cls.g), f"{path}.g"),
            method=_expect_literal(payload, "method", ["analytical", "bvp"], cls.method, path),
        )

    def to_package(self) -> SolverOptions:
        return SolverOptions(**self.__dict__)

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "n_points": {"type": "integer", "default": 300, "minimum": 16},
                "f_start": {"type": "number", "default": 300e3},
                "f_stop": {"type": "number", "default": 400e3},
                "f_step": {"type": "number", "default": 500.0, "exclusiveMinimum": 0.0},
                "g": {"type": "number", "default": 20.0},
                "method": {"type": "string", "enum": ["analytical", "bvp"], "default": "analytical"},
            },
        }


@dataclass(frozen=True)
class ArrayOutputConfig:
    mode: str = "inline"
    file_path: Optional[str] = None
    include_values: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]], path: str = "output") -> "ArrayOutputConfig":
        if payload is None:
            return cls()
        payload = _expect_mapping(payload, path)
        return cls(
            mode=_expect_literal(payload, "mode", ["inline", "file"], "inline", path),
            file_path=_expect_optional_string(payload, "file_path", None, path),
            include_values=_expect_bool(payload.get("include_values", True), f"{path}.include_values"),
        )

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["inline", "file"], "default": "inline"},
                "file_path": {"type": ["string", "null"], "default": None},
                "include_values": {"type": "boolean", "default": True},
            },
        }


@dataclass(frozen=True)
class SingleFrequencyRequest:
    frequency_hz: float
    cantilever: CantileverModel
    contact: ContactModel
    excitation: ExcitationModel
    solver_options: SolverOptionsModel
    observation_position_m: Optional[float]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SingleFrequencyRequest":
        payload = _expect_mapping(payload, "arguments")
        return cls(
            frequency_hz=_expect_number(payload.get("frequency_hz"), "arguments.frequency_hz"),
            cantilever=CantileverModel.from_dict(payload.get("cantilever", {})),
            contact=ContactModel.from_dict(payload.get("contact", {})),
            excitation=ExcitationModel.from_dict(payload.get("excitation", {})),
            solver_options=SolverOptionsModel.from_dict(payload.get("solver_options", {})),
            observation_position_m=_expect_optional_number(payload, "observation_position_m", None, "arguments"),
        )

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["frequency_hz"],
            "properties": {
                "frequency_hz": {"type": "number", "description": "Drive frequency in Hz."},
                "cantilever": CantileverModel.json_schema(),
                "contact": ContactModel.json_schema(),
                "excitation": ExcitationModel.json_schema(),
                "solver_options": SolverOptionsModel.json_schema(),
                "observation_position_m": {"type": ["number", "null"], "default": None},
            },
        }


@dataclass(frozen=True)
class FrequencySweepAnalysisRequest:
    cantilever: CantileverModel
    contact: ContactModel
    excitation: ExcitationModel
    solver_options: SolverOptionsModel
    observation_position_m: Optional[float]
    prominence_fraction: float
    use_phase: bool

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrequencySweepAnalysisRequest":
        payload = _expect_mapping(payload, "arguments")
        return cls(
            cantilever=CantileverModel.from_dict(payload.get("cantilever", {})),
            contact=ContactModel.from_dict(payload.get("contact", {})),
            excitation=ExcitationModel.from_dict(payload.get("excitation", {})),
            solver_options=SolverOptionsModel.from_dict(payload.get("solver_options", {})),
            observation_position_m=_expect_optional_number(payload, "observation_position_m", None, "arguments"),
            prominence_fraction=_expect_number(payload.get("prominence_fraction", 0.01), "arguments.prominence_fraction"),
            use_phase=_expect_bool(payload.get("use_phase", True), "arguments.use_phase"),
        )

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cantilever": CantileverModel.json_schema(),
                "contact": ContactModel.json_schema(),
                "excitation": ExcitationModel.json_schema(),
                "solver_options": SolverOptionsModel.json_schema(),
                "observation_position_m": {"type": ["number", "null"], "default": None},
                "prominence_fraction": {"type": "number", "default": 0.01},
                "use_phase": {"type": "boolean", "default": True},
            },
        }


@dataclass(frozen=True)
class SpectraMapRequest:
    cantilever: CantileverModel
    contact: ContactModel
    excitation: ExcitationModel
    solver_options: SolverOptionsModel
    n_x_points: int
    output: ArrayOutputConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SpectraMapRequest":
        payload = _expect_mapping(payload, "arguments")
        return cls(
            cantilever=CantileverModel.from_dict(payload.get("cantilever", {})),
            contact=ContactModel.from_dict(payload.get("contact", {})),
            excitation=ExcitationModel.from_dict(payload.get("excitation", {})),
            solver_options=SolverOptionsModel.from_dict(payload.get("solver_options", {})),
            n_x_points=_expect_int(payload.get("n_x_points", 100), "arguments.n_x_points"),
            output=ArrayOutputConfig.from_dict(payload.get("output"), "arguments.output"),
        )

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cantilever": CantileverModel.json_schema(),
                "contact": ContactModel.json_schema(),
                "excitation": ExcitationModel.json_schema(),
                "solver_options": SolverOptionsModel.json_schema(),
                "n_x_points": {"type": "integer", "default": 100, "minimum": 8},
                "output": ArrayOutputConfig.json_schema(),
            },
        }


@dataclass(frozen=True)
class DiscoveryRequest:
    module: str = "eb_cr_afm"

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "DiscoveryRequest":
        if payload in (None, {}):
            return cls()
        payload = _expect_mapping(payload, "arguments")
        return cls(module=_expect_string(payload.get("module", "eb_cr_afm"), "arguments.module"))

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "module": {"type": "string", "default": "eb_cr_afm"},
            },
        }


@dataclass(frozen=True)
class PublicApiEntry:
    name: str
    kind: str
    description: str
    module: str
    wrapped_as_tool: bool

