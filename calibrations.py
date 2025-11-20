"""Calibrations and validation helpers for ECU-style configuration.

This module keeps tunable parameters out of the core control loop and lets us
validate ranges before applying them to the runtime objects.  JSON/YAML-like
structures can be loaded by :func:`load_aeb_calibration`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class AEBPlanningCalibration:
    min_aeb_speed: float
    gate_confirm_frames: int
    ttc_confirm_s: float
    ttc_stage_strong: float
    ttc_stage_full: float
    stage_factor_comfort: float
    stage_factor_strong: float
    aeb_ramp_up: float
    aeb_ramp_down: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clamp_and_validate(self) -> None:
        self.min_aeb_speed = max(0.0, float(self.min_aeb_speed))
        self.gate_confirm_frames = max(1, int(self.gate_confirm_frames))
        self.ttc_confirm_s = max(0.5, float(self.ttc_confirm_s))
        self.ttc_stage_strong = max(0.1, min(self.ttc_confirm_s, float(self.ttc_stage_strong)))
        self.ttc_stage_full = max(0.05, min(self.ttc_stage_strong - 0.05, float(self.ttc_stage_full)))
        self.stage_factor_comfort = min(0.95, max(0.1, float(self.stage_factor_comfort)))
        self.stage_factor_strong = min(0.99, max(self.stage_factor_comfort + 0.01, float(self.stage_factor_strong)))
        self.aeb_ramp_up = max(0.5, float(self.aeb_ramp_up))
        self.aeb_ramp_down = max(0.5, float(self.aeb_ramp_down))


def load_aeb_calibration(path: str | None, defaults: Dict[str, Any]) -> AEBPlanningCalibration:
    cfg: Dict[str, Any] = dict(defaults)
    checksum: Tuple[str, str] | None = None
    if path:
        try:
            with open(path, "r") as fp:
                blob = fp.read()
                checksum = ("sha256", hashlib.sha256(blob.encode("utf-8")).hexdigest())
                loaded = json.loads(blob)
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if k in cfg})
                cfg["metadata"] = {
                    "source": path,
                    "provided_keys": sorted(loaded.keys()),
                }
        except Exception as exc:
            cfg["metadata"] = {"source": path, "error": str(exc)}
    if checksum:
        cfg.setdefault("metadata", {})["checksum"] = {checksum[0]: checksum[1]}
    cfg.setdefault("metadata", {})["schema_version"] = "1.0"
    calibration = AEBPlanningCalibration(**cfg)
    calibration.clamp_and_validate()
    return calibration
