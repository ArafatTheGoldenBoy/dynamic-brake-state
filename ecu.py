"""ECU-oriented adapters for the dynamic brake state app.

The existing :class:`App` class already splits perception and control into
private helpers (``_perception_step`` and ``_control_step``).  To make the
architecture more modular for an automotive-style ECU layout, these adapters
delegate the existing helpers behind explicit ECU interfaces.  This keeps
behavior unchanged while clarifying which subsystem owns which responsibility.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


@dataclass
class PerceptionSignal:
    """CAN/Ethernet friendly perception payload.

    This mirrors the detection/fusion products that a dedicated camera/radar ECU
    would broadcast.  The mapping keeps the rest of the app unchanged while
    documenting what gets exchanged between ECUs instead of direct function
    calls.
    """

    bgr: Any
    det_points: Iterable[Any]
    nearest_s_active: Optional[float]
    nearest_kind: Optional[str]
    nearest_thr: Optional[float]
    nearest_box: Optional[Any]
    nearest_conf: Optional[float]
    tl_state: str
    tl_s_active: Optional[float]
    tl_det_box: Optional[Any]
    stop_detected_current: bool
    obstacle_measurements: Iterable[Any] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())
    frame_id: Optional[int] = None
    valid: bool = True
    fault_code: Optional[str] = None
    freshness_s: float = 0.5
    units: Dict[str, str] = field(default_factory=lambda: {
        "nearest_s_active": "m",
        "nearest_conf": "[0,1]",
        "tl_s_active": "m",
    })

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def validate(self, freshness_s: Optional[float] = None) -> None:
        now = time.time()
        freshness = self.freshness_s if freshness_s is None else freshness_s
        if not math.isfinite(self.timestamp) or (now - float(self.timestamp) > freshness):
            self.valid = False
            self.fault_code = self.fault_code or "STALE_PERCEPTION"
        if self.nearest_s_active is not None:
            if not math.isfinite(self.nearest_s_active) or self.nearest_s_active < 0:
                self.valid = False
                self.fault_code = self.fault_code or "NEGATIVE_DISTANCE"
        if self.nearest_conf is not None:
            self.nearest_conf = _clamp(float(self.nearest_conf), (0.0, 1.0))
        if self.tl_s_active is not None and self.tl_s_active < 0:
            self.valid = False
            self.fault_code = self.fault_code or "TL_DISTANCE_INVALID"


@dataclass
class AEBRequest:
    """High-level deceleration request sent from ADAS domain ECU to brake ECU."""

    mode: str
    target_decel: float
    priority: str = "HIGH"
    units: Dict[str, str] = field(default_factory=lambda: {"target_decel": "m/s^2"})

    def validate(self) -> None:
        self.target_decel = float(self.target_decel)
        if self.target_decel < 0.0:
            raise ValueError("target_decel must be non-negative")
        self.priority = self.priority.upper()


@dataclass
class PlanningDecision:
    """Encapsulates planning output plus the high-level AEB request."""

    throttle: float
    brake: float
    ctrl: Optional[Any]
    hold_blocked: bool
    hold_reason: Optional[str]
    stop_armed: bool
    stop_latch_time: float
    stop_release_ignore_until: float
    debug: Dict[str, Any]
    integral_error: float
    aeb_request: Optional[AEBRequest]
    timestamp: float = field(default_factory=lambda: time.time())
    valid: bool = True
    fault_code: Optional[str] = None
    units: Dict[str, str] = field(default_factory=lambda: {
        "throttle": "[0,1]",
        "brake": "[0,1]",
    })

    def validate(self) -> None:
        if not math.isfinite(self.timestamp):
            self.valid = False
            self.fault_code = self.fault_code or "BAD_TIMESTAMP"
        self.throttle = _clamp(float(self.throttle), (0.0, 1.0))
        self.brake = _clamp(float(self.brake), (0.0, 1.0))
        if self.aeb_request is not None:
            try:
                self.aeb_request.validate()
            except Exception as exc:
                self.valid = False
                self.fault_code = self.fault_code or f"AEB_REQ_INVALID:{exc}" 

    def as_tuple(self):
        return (
            self.throttle,
            self.brake,
            self.ctrl,
            self.hold_blocked,
            self.hold_reason,
            self.stop_armed,
            self.stop_latch_time,
            self.stop_release_ignore_until,
            self.debug,
            self.integral_error,
        )


@dataclass
class ActuationResult:
    """Final actuator-facing command plus optional ABS debug signals."""

    brake: float
    abs_dbg: Optional[Dict[str, Any]] = None
    fault_code: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())
    valid: bool = True
    freshness_s: float = 0.5

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def validate(self) -> None:
        if not math.isfinite(self.timestamp):
            self.valid = False
            self.fault_code = self.fault_code or "ACT_TS_INVALID"
        self.brake = _clamp(float(self.brake), (0.0, 1.0))


@dataclass
class PerceptionECU:
    """Wraps the application's perception helper for ECU-style separation."""

    perception_fn: Callable[..., Dict[str, Any]]

    def process(
        self,
        bgr,
        depth_m,
        depth_stereo_m,
        fx,
        fy,
        cx,
        cy,
        sim_time: float,
        sensor_timestamp: Optional[float],
        v: float,
        mu: float,
        log_both: bool,
        csv_writer,
        **kwargs,
    ) -> PerceptionSignal:
        """Run perception for the current frame and return the decoded state."""

        raw = self.perception_fn(
            bgr,
            depth_m,
            depth_stereo_m,
            fx,
            fy,
            cx,
            cy,
            sim_time,
            sensor_timestamp,
            v,
            mu,
            log_both,
            csv_writer,
            **kwargs,
        )
        return PerceptionSignal(
            bgr=raw.get("bgr"),
            det_points=raw.get("det_points", []),
            nearest_s_active=raw.get("nearest_s_active"),
            nearest_kind=raw.get("nearest_kind"),
            nearest_thr=raw.get("nearest_thr"),
            nearest_box=raw.get("nearest_box"),
            nearest_conf=raw.get("nearest_conf"),
            tl_state=raw.get("tl_state", "UNKNOWN"),
            tl_s_active=raw.get("tl_s_active"),
            tl_det_box=raw.get("tl_det_box"),
            stop_detected_current=raw.get("stop_detected_current", False),
            obstacle_measurements=raw.get("obstacle_measurements", []),
        )


@dataclass
class PlanningECU:
    """Delegates control computation to the application's control helper."""

    control_fn: Callable[..., Any]

    def plan(self, *args, **kwargs) -> PlanningDecision:
        """Compute throttle/brake commands based on the current world state."""

        (
            throttle,
            brake,
            ctrl,
            hold_blocked,
            hold_reason,
            stop_armed,
            stop_latch_time,
            stop_release_ignore_until,
            dbg_map,
            I_err,
        ) = self.control_fn(*args, **kwargs)

        a_des = None if dbg_map is None else dbg_map.get("a_des")
        aeb_request = None
        if a_des is not None:
            aeb_request = AEBRequest(mode="AEB", target_decel=float(a_des), priority="HIGH")

        return PlanningDecision(
            throttle=throttle,
            brake=brake,
            ctrl=ctrl,
            hold_blocked=hold_blocked,
            hold_reason=hold_reason,
            stop_armed=stop_armed,
            stop_latch_time=stop_latch_time,
            stop_release_ignore_until=stop_release_ignore_until,
            debug=dbg_map,
            integral_error=I_err,
            aeb_request=aeb_request,
        )


@dataclass
class ActuationECU:
    """Handles actuation-level tasks such as ABS shaping."""

    abs_fn: Optional[Callable[[float, float, Any, float], float]] = None
    abs_debug_fn: Optional[Callable[[], Dict[str, Any]]] = None

    def apply_abs(
        self, brake_cmd: float, v_ego: float, wheel_speeds, a_long: float
    ) -> ActuationResult:
        """Apply ABS shaping and return both command and debug info."""

        if self.abs_fn is None:
            return ActuationResult(brake=brake_cmd, abs_dbg=None)

        brake_out = brake_cmd
        abs_dbg = None
        fault_code = None
        try:
            brake_out = self.abs_fn(brake_cmd, v_ego, wheel_speeds, a_long)
            abs_dbg = self.abs_debug_fn() if self.abs_debug_fn is not None else None
        except Exception as exc:
            fault_code = f"ABS_FAIL:{exc}"

        valid = fault_code is None
        return ActuationResult(brake=brake_out, abs_dbg=abs_dbg, fault_code=fault_code, valid=valid)


class MessageBus:
    """In-process bus that approximates ECU message passing with faults."""

    def __init__(self):
        self._queues: Dict[str, List[Tuple[float, int, Any]]] = {}
        self._topic_config: Dict[str, Dict[str, float]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    def configure_topic(
        self,
        topic: str,
        drop_rate: float = 0.0,
        jitter_s: float = 0.0,
        max_age_s: float = 0.5,
        max_depth: int = 8,
    ) -> None:
        self._topic_config[topic] = {
            "drop_rate": max(0.0, min(1.0, drop_rate)),
            "jitter_s": max(0.0, jitter_s),
            "max_age_s": max(0.05, max_age_s),
            "max_depth": max(1, max_depth),
        }
        self.metrics[topic] = {
            "sent": 0,
            "dropped_tx": 0,
            "delivered": 0,
            "expired": 0,
        }

    def _cfg(self, topic: str) -> Dict[str, float]:
        if topic not in self._topic_config:
            self.configure_topic(topic)
        return self._topic_config[topic]

    def send(
        self,
        topic: str,
        message: Any,
        now: Optional[float] = None,
        latency_s: float = 0.0,
    ) -> None:
        cfg = self._cfg(topic)
        if random.random() < cfg["drop_rate"]:
            self.metrics[topic]["dropped_tx"] += 1
            return
        deliver_at = (now if now is not None else time.time()) + max(0.0, latency_s)
        if cfg["jitter_s"] > 0.0:
            deliver_at += random.uniform(-cfg["jitter_s"], cfg["jitter_s"])
        q = self._queues.setdefault(topic, [])
        q.append((deliver_at, len(q), message))
        if len(q) > cfg["max_depth"]:
            q.pop(0)
        self.metrics[topic]["sent"] += 1

    def receive_latest(
        self, topic: str, now: Optional[float] = None, max_age_s: Optional[float] = None
    ) -> Optional[Any]:
        cfg = self._cfg(topic)
        max_age = cfg["max_age_s"] if max_age_s is None else max_age_s
        now_ts = now if now is not None else time.time()
        queue = self._queues.get(topic, [])
        ready = [(ts, prio, msg) for ts, prio, msg in queue if ts <= now_ts]
        if not ready:
            return None
        self._queues[topic] = [(ts, prio, msg) for ts, prio, msg in queue if ts > now_ts]
        ready.sort(key=lambda t: (t[0], t[1]))
        _, _, latest = ready[-1]
        age = now_ts - latest.timestamp if hasattr(latest, "timestamp") else 0.0
        if age > max_age:
            self.metrics[topic]["expired"] += 1
            return None
        self.metrics[topic]["delivered"] += 1
        return latest


@dataclass
class SafetyDecision:
    throttle: float
    brake: float
    mode: str
    faults: List[str] = field(default_factory=list)


class SafetyManager:
    """Aggregates ECU faults and enforces degraded behavior."""

    def __init__(self, brake_fail_safe: float = 1.0):
        self.brake_fail_safe = _clamp(brake_fail_safe, (0.5, 1.0))
        self.latched_faults: List[str] = []

    def _latch(self, code: Optional[str]) -> None:
        if code and code not in self.latched_faults:
            self.latched_faults.append(code)

    def evaluate(
        self,
        perception: Optional[PerceptionSignal],
        planning: Optional[PlanningDecision],
        actuation: Optional[ActuationResult],
    ) -> SafetyDecision:
        if perception is not None and (not perception.valid or perception.fault_code):
            self._latch(perception.fault_code or "PERCEPTION_INVALID")
        if planning is not None and (not planning.valid or planning.fault_code):
            self._latch(planning.fault_code or "PLANNING_INVALID")
        if actuation is not None and (not actuation.valid or actuation.fault_code):
            self._latch(actuation.fault_code or "ACTUATION_INVALID")
        faults = list(self.latched_faults)

        degrade = any(faults)
        if degrade:
            throttle_cmd = 0.0
            brake_cmd = self.brake_fail_safe
            mode = "DEGRADED_FAIL_SAFE"
        else:
            throttle_cmd = planning.throttle if planning is not None else 0.0
            brake_cmd = planning.brake if planning is not None else 0.0
            mode = "NOMINAL"

        return SafetyDecision(throttle=throttle_cmd, brake=brake_cmd, mode=mode, faults=faults)

