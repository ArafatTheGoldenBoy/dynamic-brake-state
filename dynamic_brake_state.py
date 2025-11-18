from __future__ import print_function
import os, sys, math, argparse, random, queue, glob, csv
import time
from typing import Optional, Tuple, List, Dict, Any
from collections import deque, Counter

import numpy as np
import pygame
import cv2

# Ensure CARLA Python egg is on sys.path before importing carla (supports common layouts)
try:
    import carla  # type: ignore
except Exception:
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        patterns = [
            os.path.join(this_dir, '..', 'carla', 'dist', 'carla-*.egg'),
            os.path.join(this_dir, '..', '..', 'PythonAPI', 'carla', 'dist', 'carla-*.egg'),
            os.path.join(this_dir, '..', 'dist', 'carla-*.egg'),
        ]
        for p in patterns:
            candidates.extend(glob.glob(p))
        # Prefer the first candidate; append to sys.path if found
        if candidates:
            sys.path.append(os.path.normpath(candidates[0]))
        import carla  # retry
    except Exception:
        # Leave import error to be handled at runtime if CARLA egg is missing
        raise

try:
    from ultralytics import YOLO as _ULTRA_YOLO  # type: ignore
except Exception:
    _ULTRA_YOLO = None

import torch 

# ===================== configuration =====================
IMG_W, IMG_H      = 960, 540          # each pane (front + top)
FOV_X_DEG         = 90.0              # front cam horizontal FOV
TELEPHOTO_IMG_W   = 640
TELEPHOTO_IMG_H   = 360
TELEPHOTO_FOV_X_DEG = 25.0
TELEPHOTO_STRIDE_DEFAULT = 3
TELEPHOTO_CACHE_MAX_AGE_S = 0.6
TELEPHOTO_DIGITAL_ZOOM_DEFAULT = 1.5
TELEPHOTO_DIGITAL_ZOOM_MAX = 3.5
TELEPHOTO_ZOOM_TOP_BIAS = 0.35
TL_PRIMARY_CROP_FRAC = 0.20
TL_PRIMARY_SHORT_RANGE_M = 50.0
TL_STATE_SMOOTHING_FRAMES = 5
DT                = 0.02              # 50 Hz (set 0.02 for lower latency)
FX                = (IMG_W / 2.0) / math.tan(math.radians(FOV_X_DEG / 2.0))

# Stereo camera baseline (meters)
STEREO_BASELINE_M = 0.54

# Lane gating (ignore other-lane vehicles)
LANE_HALF_WIDTH   = 1.8
LATERAL_MARGIN    = 0.6
LATERAL_MAX       = LANE_HALF_WIDTH + LATERAL_MARGIN

# Cruise / braking
A_MAX             = 8.0               # hard cap
B_COMFORT         = 3.5               # map a_des -> [0..1] brake
V_TARGET          = 10.0              # m/s default target
KP_THROTTLE       = 0.15
EPS               = 0.5
ALPHA_VBLEND      = 0.7               # blend velocity estimates
S_ENGAGE          = 80.0              # generic engage for vehicles/unknown
S_ENGAGE_TL       = 55.0              # traffic‑light engage distance (red)
S_ENGAGE_PED      = 45.0              # pedestrian engage distance
V_STOP            = 0.10
V_AEB_MIN_DEFAULT = 2.5               # minimum speed for obstacle-triggered AEB
GATE_CONFIRM_FRAMES_DEFAULT = 3       # frames of gate_hit before AEB entry
TTC_CONFIRM_S_DEFAULT = 2.5           # TTC threshold for obstacle confirmation
TTC_STAGE_STRONG_DEFAULT = 1.8        # TTC where controller escalates to strong braking
TTC_STAGE_FULL_DEFAULT   = 1.0        # TTC where controller escalates to full AEB

# Multi-stage AEB shaping (fraction of μg cap)
BRAKE_STAGE_COMFORT_FACTOR = 0.45
BRAKE_STAGE_STRONG_FACTOR  = 0.75
BRAKE_STAGE_FULL_FACTOR    = 1.00
AEB_RAMP_UP_DEFAULT        = 12.0     # max increase in a_des (m/s^2) per second
AEB_RAMP_DOWN_DEFAULT      = 18.0     # max decrease in a_des (m/s^2) per second

# Clear timers (per reason)
CLEAR_DELAY_OBS   = 0.9               # obstacle clear debounce
CLEAR_DELAY_RED   = 0.5               # GREEN debounce to clear red‑light hold
CLEAR_DELAY_S     = 3.0               # legacy/general; still used for HUD counters
STOP_WAIT_S       = 5.0               # stop‑sign wait
KICK_SEC          = 0.6               # start "kick"
KICK_THR          = 0.25

# Dynamic safety tuning knobs (match original)
TAU_MIN, TAU_MAX  = 0.15, 1.50
K_LAT_TAU         = 1.2    # sec/sec of pipeline latency
K_MU_TAU          = 0.25   # extra tau per (0.9 - mu)
K_UNC_TAU         = 0.35   # extra tau per (1 - conf)

D_MIN, D_MAX      = 3.0, 35.0   # meters
K_LAT_D           = 1.0         # meters per v*latency (reaction dist)
K_UNC_D           = 4.0         # meters per 1 m of depth sigma
K_MU_D            = 4.0         # meters per (0.9 - mu)

# Brake PI shaping (measured decel tracking) — original-ish gains
KPB               = 0.22
KIB               = 0.10
I_MAX             = 8.0

# Road / tire friction & low‑μ helpers
MU_DEFAULT        = 0.90
REV_PULSE_V_MAX   = 2.0
REV_THR           = 0.18
ABS_V_MAX         = 4.0
ABS_B_MIN         = 0.20
ABS_PWM_SCALE     = 0.5

# Slip controller defaults (per-wheel PI + μ adaptation)
FRICTION_CONFIGS = {
    'high':   {'lambda_star': 0.18, 'kp': 5.0, 'ki': 25.0},
    'medium': {'lambda_star': 0.15, 'kp': 4.0, 'ki': 20.0},
    'low':    {'lambda_star': 0.10, 'kp': 3.0, 'ki': 12.0},
}

# False-stop heuristics for telemetry/episode labeling
FALSE_STOP_MARGIN_M = 5.0      # if actual gap exceeds safety distance by this margin while braking → suspicious
FALSE_STOP_TTC_S    = 4.0      # TTC above this while brake engaged → likely false stop

# Actuation-latency measurement thresholds
ACTUATION_BRAKE_CMD_MIN   = 0.18   # require brake command above this before timing
ACTUATION_DECEL_THRESH    = 0.8    # m/s^2 decel magnitude that counts as "brake is biting"
ACTUATION_TIMEOUT_S       = 1.5    # give up if no response within this horizon


def compute_slip_per_wheel(v_ego: float, wheel_speeds: List[float]) -> List[float]:
    """Return longitudinal slip λ for each wheel in [0, 1]."""

    v = max(0.1, float(v_ego))
    slips: List[float] = []
    for v_w in wheel_speeds:
        try:
            v_w = max(0.0, float(v_w))
        except Exception:
            v_w = 0.0
        lam = (v - v_w) / v
        lam = max(0.0, min(1.0, lam))
        slips.append(lam)
    return slips


class PISlipChannel:
    def __init__(self, dt: float,
                 lambda_star: float = 0.15,
                 kp: float = 4.0,
                 ki: float = 20.0):
        self.dt = dt
        self.lambda_star = lambda_star
        self.kp = kp
        self.ki = ki
        self.I = 0.0
        self.f = 1.0

    def reset(self):
        self.I = 0.0
        self.f = 1.0

    def step(self, lam: float) -> float:
        e = float(self.lambda_star) - float(max(0.0, min(1.0, lam)))
        u_raw = self.kp * e + self.I
        f_unsat = u_raw
        f_sat = max(0.0, min(1.0, f_unsat))
        if (f_unsat > 1.0 and e > 0.0) or (f_unsat < 0.0 and e < 0.0):
            pass
        else:
            self.I += self.ki * e * self.dt
        self.f = f_sat
        return self.f


class FrictionEstimator:
    def __init__(self, g: float = 9.81, alpha: float = 0.05, mu_init: float = 0.8):
        self.g = g
        self.alpha = alpha
        self.mu_est = mu_init

    def reset(self):
        self.mu_est = 0.8

    def update(self, v_ego: float, lambda_max: float, a_long: float, brake_req: float):
        if v_ego < 5.0:
            return
        if brake_req < 0.3:
            return
        if lambda_max < 0.10 or lambda_max > 0.25:
            return
        if not math.isfinite(a_long):
            return
        mu_inst = abs(a_long) / self.g
        mu_inst = max(0.01, min(1.5, mu_inst))
        self.mu_est = (1.0 - self.alpha) * self.mu_est + self.alpha * mu_inst

    def regime(self) -> str:
        mu = self.mu_est
        if mu > 0.8:
            return 'high'
        if mu > 0.4:
            return 'medium'
        return 'low'


class PISlipABSActuator:
    def __init__(self, dt: float,
                 lambda_star: float = 0.15,
                 kp: float = 4.0,
                 ki: float = 20.0,
                 v_min_abs: float = 3.0,
                 wheel_count: int = 4):
        self.dt = dt
        self.v_min_abs = v_min_abs
        self.channels = [PISlipChannel(dt, lambda_star, kp, ki) for _ in range(wheel_count)]
        self.last_lambda_max = 0.0
        self.last_f_global = 1.0

    def reset(self):
        for ch in self.channels:
            ch.reset()
        self.last_lambda_max = 0.0
        self.last_f_global = 1.0

    def _eligible(self, v_ego: float, wheel_speeds: List[float]) -> bool:
        return v_ego >= self.v_min_abs and len(wheel_speeds) == len(self.channels)

    def step(self, brake_req: float, v_ego: float, wheel_speeds: List[float], a_long: float = 0.0) -> float:
        del a_long  # unused in fixed-mode actuator
        brake_req = max(0.0, min(1.0, float(brake_req)))
        if not self._eligible(v_ego, wheel_speeds):
            self.last_lambda_max = 0.0
            self.last_f_global = 1.0
            for ch in self.channels:
                ch.reset()
            return brake_req
        slips = compute_slip_per_wheel(v_ego, wheel_speeds)
        self.last_lambda_max = max(slips) if slips else 0.0
        f_list = [ch.step(lam) for ch, lam in zip(self.channels, slips)]
        f_global = min(f_list) if f_list else 1.0
        self.last_f_global = f_global
        return brake_req * f_global

    def debug_metrics(self) -> Dict[str, Any]:
        return {
            'lambda_max': self.last_lambda_max,
            'f_global': self.last_f_global,
            'mu_est': None,
            'regime': 'fixed',
        }


class AdaptivePISlipABSActuator(PISlipABSActuator):
    def __init__(self, dt: float, v_min_abs: float = 3.0,
                 wheel_count: int = 4,
                 friction_configs: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__(dt, v_min_abs=v_min_abs, wheel_count=wheel_count)
        self.friction = FrictionEstimator()
        self.friction_configs = friction_configs if friction_configs is not None else FRICTION_CONFIGS
        self.current_regime = 'medium'
        self._apply_friction_config(self.current_regime)

    def _apply_friction_config(self, regime: str):
        cfg = self.friction_configs.get(regime, self.friction_configs['medium'])
        for ch in self.channels:
            ch.lambda_star = cfg['lambda_star']
            ch.kp = cfg['kp']
            ch.ki = cfg['ki']
            ch.reset()
        self.current_regime = regime

    def reset(self):
        super().reset()
        self.friction.reset()
        self._apply_friction_config('medium')

    def step(self, brake_req: float, v_ego: float, wheel_speeds: List[float], a_long: float = 0.0) -> float:
        brake_req = max(0.0, min(1.0, float(brake_req)))
        if not self._eligible(v_ego, wheel_speeds):
            self.last_lambda_max = 0.0
            self.last_f_global = 1.0
            for ch in self.channels:
                ch.reset()
            return brake_req
        slips = compute_slip_per_wheel(v_ego, wheel_speeds)
        lambda_max = max(slips) if slips else 0.0
        self.last_lambda_max = lambda_max
        self.friction.update(v_ego, lambda_max, a_long, brake_req)
        regime = self.friction.regime()
        if regime != self.current_regime:
            self._apply_friction_config(regime)
        f_list = [ch.step(lam) for ch, lam in zip(self.channels, slips)]
        f_global = min(f_list) if f_list else 1.0
        self.last_f_global = f_global
        return brake_req * f_global

    def debug_metrics(self) -> Dict[str, Any]:
        return {
            'lambda_max': self.last_lambda_max,
            'f_global': self.last_f_global,
            'mu_est': self.friction.mu_est,
            'regime': self.current_regime,
        }


class LeadKalmanTracker:
    """Single-target constant-velocity Kalman filter for lead obstacle distance."""

    def __init__(self, dt: float,
                 process_var: float = 8.0,
                 meas_var: float = 4.0,
                 max_miss_s: float = 0.6,
                 reset_jump_m: float = 12.0,
                 min_iou_keep: float = 0.15):
        self.dt = dt
        self.process_var = process_var
        self.meas_var = meas_var
        self.max_miss_s = max(0.0, max_miss_s)
        self.reset_jump_m = reset_jump_m
        self.min_iou_keep = min_iou_keep
        self.reset()

    def reset(self):
        self.active = False
        self.x = np.zeros((2, 1), dtype=float)
        self.P = np.eye(2, dtype=float)
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.kind: Optional[str] = None
        self.state_time: Optional[float] = None
        self.last_meas_time: Optional[float] = None

    def deactivate(self):
        self.reset()

    def _predict_to(self, target_time: Optional[float]):
        if not self.active:
            self.state_time = target_time
            return
        if target_time is None:
            return
        if self.state_time is None:
            self.state_time = target_time
            return
        dt = float(target_time) - float(self.state_time)
        if not math.isfinite(dt):
            return
        if dt < 0.0:
            self.state_time = target_time
            return
        if dt < 1e-6:
            self.state_time = target_time
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.process_var
        Q = np.array([[0.25 * dt**4, 0.5 * dt**3],
                      [0.5 * dt**3, dt**2]], dtype=float) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.state_time = target_time

    def _kalman_update(self, z: float):
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.meas_var]])
        z_vec = np.array([[z]])
        y = z_vec - H @ self.x
        S = H @ self.P @ H.T + R
        if S.shape == (1, 1):
            inv_S = 1.0 / float(S[0, 0]) if S[0, 0] != 0 else 0.0
        else:
            inv_S = np.linalg.pinv(S)
        K = self.P @ H.T * inv_S
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    def step(self, sim_time: float,
             measurement: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if measurement is not None:
            dist = measurement.get('distance')
            if dist is not None and math.isfinite(dist):
                dist = float(dist)
                meas_ts = measurement.get('timestamp')
                if meas_ts is None or not math.isfinite(meas_ts):
                    meas_ts = sim_time
                box = measurement.get('box')
                kind = measurement.get('kind')
                if not self.active:
                    self.active = True
                    self.x = np.array([[dist], [0.0]], dtype=float)
                    self.P = np.eye(2, dtype=float) * 5.0
                    self.box = box
                    self.kind = kind
                    self.state_time = meas_ts
                    self.last_meas_time = meas_ts
                else:
                    self._predict_to(meas_ts)
                    pred_dist = float(self.x[0, 0])
                    same_kind = (self.kind is None or kind is None or self.kind == kind)
                    overlap = iou_xywh(self.box, box) if (self.box is not None and box is not None) else 0.0
                    if (abs(dist - pred_dist) > self.reset_jump_m and overlap < self.min_iou_keep) or (not same_kind and overlap < self.min_iou_keep):
                        self.x = np.array([[dist], [0.0]], dtype=float)
                        self.P = np.eye(2, dtype=float) * 5.0
                        self.box = box
                        self.kind = kind
                        self.state_time = meas_ts
                        self.last_meas_time = meas_ts
                    else:
                        self._kalman_update(dist)
                        self.box = box if box is not None else self.box
                        self.kind = kind if kind is not None else self.kind
                        self.last_meas_time = meas_ts
                        self.state_time = meas_ts
        if not self.active:
            return None
        if (self.last_meas_time is not None) and ((sim_time - self.last_meas_time) > self.max_miss_s):
            self.reset()
            return None
        self._predict_to(sim_time)
        if (self.last_meas_time is not None) and ((sim_time - self.last_meas_time) > self.max_miss_s):
            self.reset()
            return None
        return {
            'distance': float(self.x[0, 0]),
            'rate': float(self.x[1, 0]),
            'age': None if self.last_meas_time is None else float(sim_time - self.last_meas_time),
        }


class LeadMultiObjectTracker:
    """Maintain multiple Kalman tracks with ID assignments for lead selection."""

    def __init__(self, dt: float,
                 max_tracks: int = 4,
                 assoc_iou_min: float = 0.2,
                 tracker_kwargs: Optional[Dict[str, Any]] = None):
        self.dt = dt
        self.max_tracks = max(1, int(max_tracks))
        self.assoc_iou_min = max(0.0, float(assoc_iou_min))
        self._tracker_kwargs = tracker_kwargs or {}
        self._tracks: Dict[int, LeadKalmanTracker] = {}
        self._track_meta: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def reset(self):
        for tracker in self._tracks.values():
            tracker.reset()
        self._tracks.clear()
        self._track_meta.clear()
        self._next_id = 1

    def deactivate(self):
        self.reset()

    def _new_tracker(self) -> LeadKalmanTracker:
        return LeadKalmanTracker(dt=self.dt, **self._tracker_kwargs)

    def _assoc_score(self, track_id: int, measurement: Dict[str, Any]) -> float:
        tracker = self._tracks.get(track_id)
        if tracker is None:
            return 0.0
        overlap = iou_xywh(getattr(tracker, 'box', None), measurement.get('box'))
        if overlap < self.assoc_iou_min:
            return 0.0
        score = overlap
        kind = measurement.get('kind')
        if tracker.kind is not None and kind is not None and tracker.kind == kind:
            score += 0.2
        dist = measurement.get('distance')
        if dist is not None and tracker.active:
            try:
                pred = float(tracker.x[0, 0])
                score += max(0.0, 1.0 - abs(pred - float(dist)) / max(1.0, float(dist)))
            except Exception:
                pass
        return score

    def _ensure_capacity(self):
        if len(self._tracks) < self.max_tracks:
            return
        if not self._tracks:
            return
        far_id = None
        far_dist = -1.0
        for tid, tracker in self._tracks.items():
            try:
                dist = float(tracker.x[0, 0])
            except Exception:
                dist = 1e9
            if dist > far_dist:
                far_dist = dist
                far_id = tid
        if far_id is not None:
            self._tracks.pop(far_id, None)
            self._track_meta.pop(far_id, None)

    def step(self, sim_time: float, measurements: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        measurements_sorted = sorted(measurements or [], key=lambda m: m.get('distance', 1e9))
        assignments: Dict[int, Dict[str, Any]] = {}
        unmatched: List[Dict[str, Any]] = []
        for meas in measurements_sorted:
            best_track = None
            best_score = 0.0
            for track_id in self._tracks.keys():
                if track_id in assignments:
                    continue
                score = self._assoc_score(track_id, meas)
                if score > best_score:
                    best_score = score
                    best_track = track_id
            if best_track is not None:
                assignments[best_track] = meas
            else:
                unmatched.append(meas)

        active_states: List[Dict[str, Any]] = []
        to_remove: List[int] = []
        for track_id, tracker in list(self._tracks.items()):
            meas = assignments.get(track_id)
            state = tracker.step(sim_time, meas)
            if state is None:
                if not tracker.active:
                    to_remove.append(track_id)
                continue
            meta = self._track_meta.setdefault(track_id, {})
            meta['kind'] = tracker.kind
            meta['box'] = tracker.box
            active_states.append({
                'track_id': track_id,
                'distance': state.get('distance'),
                'rate': state.get('rate'),
                'age': state.get('age'),
                'kind': tracker.kind,
                'box': tracker.box,
            })
        for tid in to_remove:
            self._tracks.pop(tid, None)
            self._track_meta.pop(tid, None)

        for meas in unmatched:
            self._ensure_capacity()
            tracker = self._new_tracker()
            state = tracker.step(sim_time, meas)
            if state is None:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = tracker
            self._track_meta[track_id] = {'kind': tracker.kind, 'box': tracker.box}
            active_states.append({
                'track_id': track_id,
                'distance': state.get('distance'),
                'rate': state.get('rate'),
                'age': state.get('age'),
                'kind': tracker.kind,
                'box': tracker.box,
            })

        active_states.sort(key=lambda s: (float('inf') if s.get('distance') is None else float(s['distance'])))
        best_state = active_states[0] if active_states else None
        return best_state, active_states

    def active_track_count(self) -> int:
        return len(self._tracks)


# Detection (YOLO specific)
YOLO_MODEL_PATH   = 'yolo12n.pt'     # path to YOLO12n weights
CONF_THR_DEFAULT  = 0.45
NMS_THR           = 0.45
H_MIN_PX          = 10
CENTER_BAND_FRAC  = 0.35

# Classes of interest (trigger names)
VEHICLE_CLASSES     = {'car','bus','truck','motorcycle','motorbike','bicycle','train'}
PEDESTRIAN_CLASSES  = {'person'}
TRIGGER_CLASSES     = VEHICLE_CLASSES | {'traffic light','stop sign'} | PEDESTRIAN_CLASSES

# Approx real heights (meters) for monocular pinhole
OBJ_HEIGHT_M = {
    'person': 1.70,
    'car': 1.50,
    'traffic light': 2.20,
    'bus': 3.00,
    'truck': 3.20,
    'motorcycle': 1.40,
    'motorbike': 1.40,
    'bicycle': 1.40,
    'train': 3.50,
    'stop sign': 0.75,
}

# Debug toggle for traffic light mask visualization (kept)
DEBUG_TL = True

# Depth / stereo ROI + fusion defaults
DEPTH_ROI_SHRINK_DEFAULT   = 0.40
STEREO_ROI_SHRINK_DEFAULT  = 0.30
STEREO_FUSE_NEAR_WEIGHT    = 0.75  # close objects -> trust depth camera more
STEREO_FUSE_FAR_WEIGHT     = 0.45  # far objects -> lean slightly toward stereo
STEREO_FUSE_DISAGREE_M     = 12.0  # beyond this delta, pick the safer (closer) estimate


class BaseDetector:
    """Minimal detector interface so we can swap YOLO, SSD, etc.

    Implementors must provide predict_raw(bgr) -> (classIds, confs, boxes).
    """

    def predict_raw(self, bgr: np.ndarray):
        raise NotImplementedError

# ---------- label normalization ----------
def _norm_label(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalpha())  # 'traffic light' -> 'trafficlight'

TRIGGER_NAMES_NORM = {
    'trafficlight','stopsign','person','car','bus','truck','motorcycle','motorbike','bicycle','train'
}

# ---- per-class confidence parser ----
# Format: "traffic light:0.55, stop sign:0.45, person:0.40"
# Names are case-insensitive; spaces allowed; we normalize like _norm_label
# If a class isn't listed, it falls back to detector.conf_thr

def parse_per_class_conf_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping

# ---- generic per-class float/int maps for other overrides ----

def _parse_float_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping

def _parse_int_map(spec: Optional[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = int(float(v.strip()))
        except Exception:
            pass
    return mapping

# ---- per-class IoU (NMS) parser ----
# Format: "traffic light:0.40, person:0.55" -> IoU thresholds per class for a second-stage NMS

def parse_per_class_iou_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping

# ---- per-class engage distance parser ----
# Format: "person:45, traffic light:55, car:80, stopsign:80"

def parse_engage_override_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping

# ---- per-class minimum box height (px) ----

def parse_min_h_override_map(spec: Optional[str]) -> Dict[str, int]:
    return _parse_int_map(spec)

# ---- per-class center-band fraction (0..1 of IMG_W) ----

def parse_gate_frac_override_map(spec: Optional[str]) -> Dict[str, float]:
    return _parse_float_map(spec)

# ---- per-class lateral max (meters) ----

def parse_gate_lateral_override_map(spec: Optional[str]) -> Dict[str, float]:
    return _parse_float_map(spec)


# ===================== helpers =====================
def fov_y_from_x(width: int, height: int, fov_x_deg: float) -> float:
    fov_x = math.radians(fov_x_deg)
    return 2.0 * math.atan((height / width) * math.tan(fov_x / 2.0))

def focal_length_y_px(width: int, height: int, fov_x_deg: float) -> float:
    fovy = fov_y_from_x(width, height, fov_x_deg)
    return (height / 2.0) / math.tan(fovy / 2.0)

def bgr_to_pygame_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

def carla_image_to_surface(image) -> pygame.Surface:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

def wrap_pi(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def yaw_to_compass(yaw_deg: float) -> str:
    y = (yaw_deg + 360.0) % 360.0
    dirs = ['E','SE','S','SW','W','NW','N','NE','E']
    idx = int((y + 22.5) // 45)
    return dirs[idx]

def iou_xywh(a: Optional[Tuple[int,int,int,int]], b: Optional[Tuple[int,int,int,int]]) -> float:
    if not a or not b: return 0.0
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax+aw, ay+ah, bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a, area_b = aw*ah, bw*bh
    return inter / float(area_a + area_b - inter)

def shadow(surface: pygame.Surface, text: str, pos, color, shadow_color=(0,0,0), offset=1):
    font = pygame.font.SysFont('Arial', 20)
    s = font.render(text, True, shadow_color); surface.blit(s, (pos[0]+offset, pos[1]+offset))
    s2 = font.render(text, True, color);        surface.blit(s2, pos)

# Fallback labels for COCO (only used if YOLO model fails to load)
def _fallback_labels_91():
    L = [""]*90
    mapping = {
        1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 6:"bus", 7:"train", 8:"truck",
        10:"traffic light", 13:"stop sign"
    }
    for k,v in mapping.items():
        idx = k-1
        if 0 <= idx < 90:
            L[idx] = v
    return L

# ===================== Depth & TL helpers =====================
def decode_depth_meters_from_bgra(depth_bgra: np.ndarray) -> np.ndarray:
    b = depth_bgra[..., 0].astype(np.uint32)
    g = depth_bgra[..., 1].astype(np.uint32)
    r = depth_bgra[..., 2].astype(np.uint32)
    normalized = (r + g * 256 + b * 256 * 256).astype(np.float32) / float(256**3 - 1)
    return 1000.0 * normalized

def intrinsics_from_fov(width: int, height: int, fov_x_deg: float):
    fov_x = math.radians(fov_x_deg)
    fx = (width / 2.0) / math.tan(fov_x / 2.0)
    fy = focal_length_y_px(width, height, fov_x_deg)
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy

def pixel_to_camera(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float):
    if z_m <= 0 or not np.isfinite(z_m):
        return None
    x = (u - cx) * z_m / max(1e-6, fx)
    y = (v - cy) * z_m / max(1e-6, fy)
    return (float(x), float(y), float(z_m))

def median_depth_in_box(depth_m: np.ndarray, box, shrink: float = 0.4):
    x, y, w, h = box
    x0 = max(0, int(x + w*shrink/2.0))
    y0 = max(0, int(y + h*shrink/2.0))
    x1 = min(depth_m.shape[1], int(x + w - w*shrink/2.0))
    y1 = min(depth_m.shape[0], int(y + h - h*shrink/2.0))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_m[y0:y1, x0:x1]
    d = float(np.median(roi))
    if not np.isfinite(d) or d <= 0.0 or d > 999.0:
        return None
    return d

def depth_sigma_in_box(depth_m: np.ndarray, box, shrink: float = 0.4):
    x, y, w, h = box
    x0 = max(0, int(x + w*shrink/2.0)); y0 = max(0, int(y + h*shrink/2.0))
    x1 = min(depth_m.shape[1], int(x + w - w*shrink/2.0))
    y1 = min(depth_m.shape[0], int(y + h - h*shrink/2.0))
    if x1 <= x0 or y1 <= y0: return None
    roi = depth_m[y0:y1, x0:x1]

    min_pix = max(50, int(0.02 * (roi.shape[0] * roi.shape[1])))
    flat = roi[np.isfinite(roi)]
    if flat.size < min_pix:
        return None

    med = np.median(flat); mad = np.median(np.abs(flat - med))
    sigma = 1.4826 * mad
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else None

def fuse_depth_sources(s_depth: Optional[float], s_stereo: Optional[float], box_h_px: int) -> Tuple[Optional[float], str]:
    """Blend CARLA depth and stereo ranges inside a detection ROI."""
    if s_depth is None and s_stereo is None:
        return None, 'none'
    if s_depth is None:
        return s_stereo, 'stereo'
    if s_stereo is None:
        return s_depth, 'depth'
    prox = max(0.0, min(1.0, float(box_h_px) / max(1.0, float(IMG_H))))
    w_depth = STEREO_FUSE_FAR_WEIGHT + (STEREO_FUSE_NEAR_WEIGHT - STEREO_FUSE_FAR_WEIGHT) * prox
    fused = (w_depth * s_depth) + ((1.0 - w_depth) * s_stereo)
    if abs(s_depth - s_stereo) > STEREO_FUSE_DISAGREE_M:
        return min(s_depth, s_stereo), 'min'
    return fused, 'fused'

# Simple ROI TL classifier (fallback if CARLA API not reliable)
def estimate_tl_color_from_roi(bgr: np.ndarray, box: Tuple[int,int,int,int]) -> str:
    x,y,w,h = box
    if w <= 0 or h <= 0:
        return 'UNKNOWN'
    h_lim, w_lim = bgr.shape[0], bgr.shape[1]
    roi = bgr[max(0,y):min(h_lim,y+h), max(0,x):min(w_lim,x+w)]
    if roi.size == 0:
        return 'UNKNOWN'
    roi_h = roi.shape[0]
    thirds = [(0, int(roi_h/3)), (int(roi_h/3), int(2*roi_h/3)), (int(2*roi_h/3), roi_h)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    means = []
    for (y0,y1) in thirds:
        seg = hsv[y0:y1, :, :]
        if seg.size != 0:
            vch = seg[..., 2].astype(np.float32)
            means.append(float(np.mean(vch)))
        else:
            means.append(0.0)
    idx = int(np.argmax(means))
    seg = hsv[thirds[idx][0]:thirds[idx][1], :, :]
    if seg.size == 0:
        return 'UNKNOWN'
    hch = seg[...,0].astype(np.float32)
    sch = seg[...,1].astype(np.float32)/255.0
    mask_sat = sch > 0.35
    if not np.any(mask_sat):
        return 'UNKNOWN'
    hsel = hch[mask_sat]
    red_mask = (hsel < 12) | (hsel > 168)
    yellow_mask = (hsel >= 18) & (hsel <= 35)
    # Widen green range slightly to be robust in CARLA lighting
    green_mask = (hsel >= 40) & (hsel <= 105)
    # Primary decision: brightest third should match its expected color
    if idx == 0 and np.mean(red_mask) > 0.25: return 'RED'
    if idx == 1 and np.mean(yellow_mask) > 0.25: return 'YELLOW'
    if idx == 2 and np.mean(green_mask) > 0.25: return 'GREEN'
    # Fallback: if bottom third has strong green evidence even if not the absolute brightest
    b0, b1 = thirds[2]
    seg_bot = hsv[b0:b1, :, :]
    if seg_bot.size != 0:
        h_bot = seg_bot[...,0].astype(np.float32)
        s_bot = seg_bot[...,1].astype(np.float32) / 255.0
        mask_sat_bot = s_bot > 0.35
        if np.any(mask_sat_bot):
            g_frac = np.mean(((h_bot >= 40) & (h_bot <= 105))[mask_sat_bot])
            if g_frac > 0.35:
                return 'GREEN'
    return 'UNKNOWN'


def apply_digital_zoom(image: np.ndarray, zoom: float,
                       top_bias: float = TELEPHOTO_ZOOM_TOP_BIAS) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Crop/resize the upper-center region of an image to emulate extra zoom."""
    if image is None:
        return image, None
    zoom = max(1.0, float(zoom))
    if zoom <= 1.0 + 1e-3:
        return image, None
    h, w = image.shape[:2]
    crop_w = max(4, min(w, int(round(w / zoom))))
    crop_h = max(4, min(h, int(round(h / zoom))))
    x0 = max(0, (w - crop_w) // 2)
    bias = max(0.0, min(1.0, float(top_bias)))
    y0 = max(0, min(h - crop_h, int(round((h - crop_h) * bias))))
    crop = image[y0:y0 + crop_h, x0:x0 + crop_w]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = {
        'crop': (x0, y0, crop_w, crop_h),
        'scale_x': crop_w / float(w),
        'scale_y': crop_h / float(h),
    }
    return resized, meta


def remap_zoom_box(box: Tuple[int, int, int, int], meta: Optional[Dict[str, Any]],
                   full_w: int, full_h: int) -> Tuple[int, int, int, int]:
    """Convert a bounding box from zoomed coordinates back to original space."""
    if not meta:
        return box
    x0, y0, crop_w, crop_h = meta['crop']
    scale_x = meta['scale_x']
    scale_y = meta['scale_y']
    x1, y1, w, h = box
    x1_full = int(round(x0 + x1 * scale_x))
    y1_full = int(round(y0 + y1 * scale_y))
    w_full = int(round(max(1, w * scale_x)))
    h_full = int(round(max(1, h * scale_y)))
    x1_full = max(0, min(full_w - 1, x1_full))
    y1_full = max(0, min(full_h - 1, y1_full))
    if x1_full + w_full > full_w:
        w_full = max(1, full_w - x1_full)
    if y1_full + h_full > full_h:
        h_full = max(1, full_h - y1_full)
    return x1_full, y1_full, w_full, h_full

# ===================== data types =====================
from dataclasses import dataclass

@dataclass
class EgoState:
    x: float
    y: float
    z: float
    yaw_deg: float
    v_mps: float
    t: float

# ===================== components =====================
class YOLODetector(BaseDetector):
    def __init__(self,
                 conf_thr: float = CONF_THR_DEFAULT,
                 nms_thr: float = NMS_THR,
                 img_size: int = 640,
                 device: str = 'auto',
                 use_half: bool = False,
                 agnostic: bool = False,
                 classes: Optional[str] = None,
                 max_det: int = 300,
                 dnn: bool = False,
                 augment: bool = False,
                 per_class_iou_map: Optional[Dict[str, float]] = None):
        """Ultralytics YOLO wrapper with runtime-configurable options.
        - img_size: inference size (int or square side), e.g., 480 -> 480x480
        - device: 'auto' | 'cpu' | 'cuda:0' | 'cuda'
        - use_half: fp16 if device is CUDA and torch available
        - agnostic: class-agnostic NMS
        - classes: comma-separated names or ids (e.g. "person,car,traffic light" or "0,2,7"). None = all
        - max_det: maximum detections per image
        - dnn: use OpenCV DNN backend inside Ultralytics (rarely needed)
        - augment: test-time augmentation
        - per_class_iou_map: optional per-class IoU thresholds for a second-stage NMS
        """
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.img_size = int(img_size)
        self.device = device
        self.use_half = use_half
        self.agnostic = agnostic
        self.classes_raw = classes  # parse later, after labels known
        self.max_det = int(max_det)
        self.dnn = bool(dnn)
        self.augment = bool(augment)
        self.per_class_iou_map = per_class_iou_map or {}

        self.model = None
        self.labels: Optional[Dict[int, str]] = None
        self.enabled = False

        # Load model
        if _ULTRA_YOLO is not None and os.path.exists(YOLO_MODEL_PATH):
            try:
                self.model = _ULTRA_YOLO(YOLO_MODEL_PATH)
                self.enabled = True
                # Names
                try:
                    if hasattr(self.model, 'names'):
                        self.labels = {int(k): v for k, v in self.model.names.items()}  # type: ignore
                except Exception:
                    self.labels = None
                # Device/half
                dev = self._resolve_device(self.device)
                if (torch is not None) and (self.model is not None):
                    try:
                        self.model.to(dev)
                        if self.use_half and 'cuda' in dev and torch.cuda.is_available():
                            # (removed) self.model.model.half()  # let Ultralytics manage dtype via predict(half=...)  # type: ignore[attr-defined]
                            if torch.backends and hasattr(torch.backends, 'cudnn'):
                                torch.backends.cudnn.benchmark = True  # type: ignore
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] YOLO load failed: {e}")
        if self.labels is None:
            fb = _fallback_labels_91()
            self.labels = {i: v for i, v in enumerate(fb)}

    def _resolve_device(self, device: str) -> str:
        if device in (None, '', 'auto'):
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return device

    def _parse_classes(self) -> Optional[List[int]]:
        if not self.classes_raw:
            return None
        raw = [s.strip() for s in str(self.classes_raw).split(',') if s.strip()]
        idxs: List[int] = []
        for token in raw:
            if token.isdigit():
                idxs.append(int(token))
            else:
                if self.labels is None:
                    continue
                lower = token.lower()
                found = [i for i, n in self.labels.items() if isinstance(n, str) and n.lower() == lower]
                if found:
                    idxs.append(found[0])
        return idxs if idxs else None

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter + 1e-9)

    def _apply_per_class_nms(self, classIds, confs, boxes):
        # Convert to xyxy for IoU
        xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
        classIds = list(classIds); confs = list(confs)
        keep_mask = [True] * len(xyxy)
        # Group indices by normalized class name
        class_to_indices: Dict[str, List[int]] = {}
        for i, cid in enumerate(classIds):
            name = self.labels.get(cid, str(cid)) if self.labels else str(cid)
            class_to_indices.setdefault(_norm_label(name), []).append(i)
        # Run greedy NMS per class with custom IoU
        for cname, idxs in class_to_indices.items():
            iou_thr = self.per_class_iou_map.get(cname, None)
            if iou_thr is None:
                continue  # use Ultralytics default already applied
            order = sorted(idxs, key=lambda i: confs[i], reverse=True)
            kept: List[int] = []
            while order:
                i = order.pop(0)
                kept.append(i)
                rest = []
                for j in order:
                    if self._iou_xyxy(xyxy[i], xyxy[j]) <= iou_thr:
                        rest.append(j)
                    else:
                        keep_mask[j] = False
                order = rest
        # Filter outputs
        classIds_f = [c for k, c in enumerate(classIds) if keep_mask[k]]
        confs_f    = [s for k, s in enumerate(confs) if keep_mask[k]]
        boxes_f    = [b for k, b in enumerate(boxes) if keep_mask[k]]
        return classIds_f, confs_f, boxes_f

    def predict_raw(self, bgr: np.ndarray):
        if not self.enabled or self.model is None:
            return [], [], []
        boxes_out, confs_out, classIds_out = [], [], []
        try:
            dev = self._resolve_device(self.device)
            classes_arg = self._parse_classes()
            half_flag = (self.use_half and ('cuda' in dev)) and (getattr(torch, 'cuda', None) is None or torch.cuda.is_available())
            try:
                results = self.model.predict(
                    bgr,
                    imgsz=self.img_size,
                    conf=self.conf_thr,
                    iou=self.nms_thr,
                    device=dev,
                    half=half_flag,
                    classes=classes_arg,
                    agnostic_nms=self.agnostic,
                    max_det=self.max_det,
                    dnn=self.dnn,
                    augment=self.augment,
                    verbose=False)
            except Exception as _e_pred:
                msg = str(_e_pred)
                if ('same dtype' in msg) or ('Half' in msg) or ('mat1 and mat2' in msg):
                    print('[WARN] YOLO half-precision failed; retrying in FP32...')
                    results = self.model.predict(
                        bgr,
                        imgsz=self.img_size,
                        conf=self.conf_thr,
                        iou=self.nms_thr,
                        device=dev,
                        half=False,
                        classes=classes_arg,
                        agnostic_nms=self.agnostic,
                        max_det=self.max_det,
                        dnn=self.dnn,
                        augment=self.augment,
                        verbose=False)
                else:
                    raise
            for res in results:
                xyxy = getattr(res.boxes, 'xyxy', None)
                confs_tensor = getattr(res.boxes, 'conf', None)
                cls_tensor   = getattr(res.boxes, 'cls', None)
                if xyxy is None or confs_tensor is None or cls_tensor is None:
                    continue
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].cpu().numpy().tolist()
                    conf_val = float(confs_tensor[i].cpu().numpy())
                    cls_val  = int(cls_tensor[i].cpu().numpy())
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    boxes_out.append((x, y, w, h))
                    classIds_out.append(cls_val)
                    confs_out.append(conf_val)
        except Exception as e:
            print(f"[WARN] YOLO inference failed: {e}")
            return [], [], []
        # Optional second-stage per-class NMS
        if self.per_class_iou_map:
            try:
                classIds_out, confs_out, boxes_out = self._apply_per_class_nms(classIds_out, confs_out, boxes_out)
            except Exception as _e:
                pass
        return classIds_out, confs_out, boxes_out

class RangeEstimator:
    def __init__(self, use_cuda: bool = False, method: str = 'bm'):
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(IMG_W, IMG_H, FOV_X_DEG)
        self.use_cuda = bool(use_cuda)
        self.method = method.lower()
        self.stereo = None           # CPU StereoBM
        self.stereo_cuda = None      # CUDA StereoBM/SGM

# ===================== World & Sensors
    def ensure_stereo(self, num_disp: int = 16*6, block: int = 15):
        """Create a stereo matcher. If --stereo-cuda is set and OpenCV CUDA is available,
        prefer GPU; otherwise fall back to CPU StereoBM. Always returns True/False for 'ready'."""
        # Try CUDA first if asked
        if self.use_cuda and hasattr(cv2, 'cuda'):
            # Try both possible factory names depending on OpenCV build
            try:
                if self.method == 'sgm' and hasattr(cv2.cuda, 'StereoSGM_create'):
                    self.stereo_cuda = cv2.cuda.StereoSGM_create()  # type: ignore[attr-defined]
                else:
                    if hasattr(cv2.cuda, 'createStereoBM'):
                        self.stereo_cuda = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block)  # type: ignore[attr-defined]
                    elif hasattr(cv2, 'cuda_StereoBM_create'):
                        self.stereo_cuda = cv2.cuda_StereoBM_create(numDisparities=num_disp, blockSize=block)  # type: ignore[attr-defined]
            except Exception:
                self.stereo_cuda = None
        # CPU fallback
        if self.stereo_cuda is None and self.stereo is None:
            try:
                self.stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
            except Exception:
                self.stereo = None
        return (self.stereo_cuda is not None) or (self.stereo is not None)

    def stereo_depth(self, left_bgra: np.ndarray, right_bgra: np.ndarray) -> Optional[np.ndarray]:
        # GPU path
        if self.stereo_cuda is not None and hasattr(cv2, 'cuda'):
            try:
                left_gray  = cv2.cvtColor(left_bgra,  cv2.COLOR_BGRA2GRAY)
                right_gray = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2GRAY)
                gL = cv2.cuda_GpuMat(); gR = cv2.cuda_GpuMat()
                gL.upload(left_gray); gR.upload(right_gray)
                disp_gpu = self.stereo_cuda.compute(gL, gR)  # type: ignore[attr-defined]
                disp = disp_gpu.download().astype(np.float32) / 16.0
                depth = np.full_like(disp, 1e6, dtype=np.float32)
                valid = disp > 0.1
                if np.any(valid):
                    depth[valid] = (self.fx * STEREO_BASELINE_M) / disp[valid]
                return depth
            except Exception:
                pass  # fall through to CPU if GPU fails mid-run
        # CPU path
        if self.stereo is None:
            return None
        left_gray  = cv2.cvtColor(left_bgra,  cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2GRAY)
        disp = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        depth = np.full_like(disp, 1e6, dtype=np.float32)
        valid = disp > 0.1
        if np.any(valid):
            depth[valid] = (self.fx * STEREO_BASELINE_M) / disp[valid]
        return depth

# ===================== World & Sensors ====================
class SensorRig:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, range_est: str,
                 enable_top: bool = True, enable_depth: bool = True, enable_telephoto: bool = True):
        self.world = world
        self.vehicle = vehicle
        self.range_est = range_est
        self.enable_telephoto = enable_telephoto
        bp_lib = world.get_blueprint_library()

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMG_W))
        cam_bp.set_attribute('image_size_y', str(IMG_H))
        cam_bp.set_attribute('fov', str(FOV_X_DEG))
        # Aperture and focus control
        cam_bp.set_attribute('fstop', '8.0')       # larger => less depth-of-field blur
        cam_bp.set_attribute('focal_distance', '1000.0')  # e.g., 1000 Unreal units (≈10 m in 0.9.7 units) – you may need adjust for 0.10

        # Disable post-processing if you don’t want DOF, etc
        # cam_bp.set_attribute('enable_postprocess_effects', 'False')

        # Optional: remove lens distortion
        cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        cam_bp.set_attribute('lens_k', '0.0')
        # --- add motion blur config here ---
        try:
            # type: 0=None, 1=Gaussian, 2=Box (depending on CARLA version)
            if cam_bp.has_attribute('motion_blur_intensity'):
                cam_bp.set_attribute('motion_blur_intensity', '0.0')   # 0..1
            if cam_bp.has_attribute('motion_blur_max_distortion'):
                cam_bp.set_attribute('motion_blur_max_distortion', '0.0')
            if cam_bp.has_attribute('motion_blur_min_object_screen_size'):
                cam_bp.set_attribute('motion_blur_min_object_screen_size', '1.0')
        except RuntimeError:
            pass
        # -----------------------------------

        self.cam_front = world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.6, z=1.5)), attach_to=vehicle)
        self.q_front: "queue.Queue[carla.Image]" = queue.Queue()
        self.cam_front.listen(self.q_front.put)

        self.cam_top = None
        self.q_top = None
        if enable_top:
            self.cam_top = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=0.0, z=25.0), carla.Rotation(pitch=-90.0)), attach_to=vehicle)
            self.q_top = queue.Queue()
            self.cam_top.listen(self.q_top.put)

        self.cam_tele = None
        self.q_tele = None
        self.cam_tele_depth = None
        self.q_tele_depth = None
        if enable_telephoto:
            tele_bp = bp_lib.find('sensor.camera.rgb')
            tele_bp.set_attribute('image_size_x', str(TELEPHOTO_IMG_W))
            tele_bp.set_attribute('image_size_y', str(TELEPHOTO_IMG_H))
            tele_bp.set_attribute('fov', str(TELEPHOTO_FOV_X_DEG))
            tele_bp.set_attribute('fstop', '8.0')
            tele_bp.set_attribute('focal_distance', '1000.0')
            tele_bp.set_attribute('lens_circle_multiplier', '0.0')
            tele_bp.set_attribute('lens_k', '0.0')
            try:
                if tele_bp.has_attribute('motion_blur_intensity'):
                    tele_bp.set_attribute('motion_blur_intensity', '0.0')
                if tele_bp.has_attribute('motion_blur_max_distortion'):
                    tele_bp.set_attribute('motion_blur_max_distortion', '0.0')
                if tele_bp.has_attribute('motion_blur_min_object_screen_size'):
                    tele_bp.set_attribute('motion_blur_min_object_screen_size', '1.0')
            except RuntimeError:
                pass
            tele_tf = carla.Transform(carla.Location(x=1.6, z=1.5))
            self.cam_tele = world.spawn_actor(tele_bp, tele_tf, attach_to=vehicle)
            self.q_tele = queue.Queue()
            self.cam_tele.listen(self.q_tele.put)

            if enable_depth:
                tele_depth_bp = bp_lib.find('sensor.camera.depth')
                tele_depth_bp.set_attribute('image_size_x', str(TELEPHOTO_IMG_W))
                tele_depth_bp.set_attribute('image_size_y', str(TELEPHOTO_IMG_H))
                tele_depth_bp.set_attribute('fov', str(TELEPHOTO_FOV_X_DEG))
                self.cam_tele_depth = world.spawn_actor(tele_depth_bp, tele_tf, attach_to=vehicle)
                self.q_tele_depth = queue.Queue()
                self.cam_tele_depth.listen(self.q_tele_depth.put)

        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(IMG_W))
        depth_bp.set_attribute('image_size_y', str(IMG_H))
        depth_bp.set_attribute('fov', str(FOV_X_DEG))
        self.cam_depth = None
        self.q_depth = None
        if enable_depth:
            self.cam_depth = world.spawn_actor(
                depth_bp, carla.Transform(carla.Location(x=1.6, z=1.5)), attach_to=vehicle)
            self.q_depth = queue.Queue()
            self.cam_depth.listen(self.q_depth.put)

        self.cam_stereo_left = None
        self.cam_stereo_right = None
        self.q_stereo_left = None
        self.q_stereo_right = None
        if range_est == 'stereo':
            self.cam_stereo_left  = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=1.6, y=-STEREO_BASELINE_M/2.0, z=1.5)), attach_to=vehicle)
            self.cam_stereo_right = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=1.6, y= STEREO_BASELINE_M/2.0, z=1.5)), attach_to=vehicle)
            self.q_stereo_left, self.q_stereo_right = queue.Queue(), queue.Queue()
            self.cam_stereo_left.listen(self.q_stereo_left.put)
            self.cam_stereo_right.listen(self.q_stereo_right.put)

        self.actors = [a for a in [self.cam_front, self.cam_top, self.cam_depth,
                                   self.cam_stereo_left, self.cam_stereo_right,
                                   self.cam_tele, self.cam_tele_depth] if a is not None]

    @staticmethod
    def _get_latest(q: "queue.Queue[Any]", timeout: float):
        item = q.get(timeout=timeout)
        try:
            while True:
                item = q.get_nowait()
        except queue.Empty:
            pass
        return item

    @staticmethod
    def _get_for_frame(q: "queue.Queue[Any]", expected_frame: int, timeout: float):
        """Block until an item with image.frame == expected_frame arrives.
        Discards older frames; in rare cases if a newer frame arrives first, it keeps waiting a short time.
        Falls back to returning the most recent item if exact match does not arrive before timeout elapses repeatedly."""
        deadline = time.time() + max(timeout, 0.5)
        latest = None
        while time.time() < deadline:
            item = q.get(timeout=max(0.05, min(0.2, deadline - time.time())))
            latest = item
            try:
                f = getattr(item, 'frame')
            except Exception:
                f = None
            if f == expected_frame:
                return item
            # If older, keep draining; if newer, continue a bit in case others catch up
            try:
                while True:
                    nxt = q.get_nowait()
                    latest = nxt
                    f = getattr(nxt, 'frame', None)
                    if f == expected_frame:
                        return nxt
            except queue.Empty:
                pass
        return latest

    def read(self, timeout: float = 2.0, expected_frame: Optional[int] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if expected_frame is None:
            out['front'] = self._get_latest(self.q_front, timeout)
            if self.q_top is not None:
                out['top'] = self._get_latest(self.q_top, timeout)
            if self.q_depth is not None:
                out['depth'] = self._get_latest(self.q_depth, timeout)
            if self.q_stereo_left is not None and self.q_stereo_right is not None:
                out['stereo_left']  = self._get_latest(self.q_stereo_left, timeout)
                out['stereo_right'] = self._get_latest(self.q_stereo_right, timeout)
            if self.q_tele is not None:
                out['tele_rgb'] = self._get_latest(self.q_tele, timeout)
            if self.q_tele_depth is not None:
                out['tele_depth'] = self._get_latest(self.q_tele_depth, timeout)
        else:
            out['front'] = self._get_for_frame(self.q_front, expected_frame, timeout)
            if self.q_top is not None:
                out['top'] = self._get_for_frame(self.q_top, expected_frame, timeout)
            if self.q_depth is not None:
                out['depth'] = self._get_for_frame(self.q_depth, expected_frame, timeout)
            if self.q_stereo_left is not None and self.q_stereo_right is not None:
                out['stereo_left']  = self._get_for_frame(self.q_stereo_left, expected_frame, timeout)
                out['stereo_right'] = self._get_for_frame(self.q_stereo_right, expected_frame, timeout)
            if self.q_tele is not None:
                out['tele_rgb'] = self._get_for_frame(self.q_tele, expected_frame, timeout)
            if self.q_tele_depth is not None:
                out['tele_depth'] = self._get_for_frame(self.q_tele_depth, expected_frame, timeout)
        return out

    def destroy(self):
        for a in self.actors:
            try:
                a.stop()
            except Exception:
                pass
            # Detach listeners explicitly (older CARLA needed listen(None))
            try:
                a.listen(lambda *_: None)
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass

class WorldManager:
    def __init__(self, host: str, port: int, town: Optional[str], mu: float, apply_tire_friction: bool,
                 npc_vehicles: int = 0, npc_walkers: int = 0, npc_seed: Optional[int] = None,
                 npc_autopilot: bool = True, npc_speed_diff_pct: int = 10):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        if town:
            # Prefer _Opt maps when available; fall back gracefully
            load_town = town
            try:
                avail = []
                try:
                    avail = list(self.client.get_available_maps())
                except Exception:
                    avail = []
                def has_map(name: str) -> bool:
                    try:
                        return any(m.endswith('/' + name) or m.endswith('\\' + name) or m == name for m in avail)
                    except Exception:
                        return False
                if not town.endswith('_Opt'):
                    opt = town + '_Opt'
                    if has_map(opt):
                        print(f"[INFO] Auto-switching to optimized map '{opt}' (was '{town}').")
                        load_town = opt
                else:
                    base = town[:-4]
                    if not has_map(town) and has_map(base):
                        print(f"[INFO] Optimized map '{town}' not found; falling back to '{base}'.")
                        load_town = base
            except Exception:
                load_town = town
            # Ensure previous world isn't left in sync mode
            try:
                w0 = self.client.get_world(); s0 = w0.get_settings()
                if s0.synchronous_mode:
                    s0.synchronous_mode = False
                    w0.apply_settings(s0)
            except Exception:
                pass

            # Load the map with explicit map_layers keyword (positional 2nd arg is reset_settings)
            try:
                self.world = self.client.load_world(load_town, map_layers=carla.MapLayer.All)
            except TypeError:
                # Older API without map_layers kw: fall back to simple load
                self.world = self.client.load_world(load_town)

            # Remove baked layers if supported (CARLA ≥ 0.9.10 and *_Opt maps)
            try:
                if hasattr(carla, 'MapLayer'):
                    to_remove = (carla.MapLayer.ParkedVehicles | carla.MapLayer.Props)
                    self.world.unload_map_layer(to_remove)
                    # Apply change on server
                    try:
                        self.world.tick()
                    except Exception:
                        pass
                    try:
                        map_name = self.world.get_map().name
                        if not map_name.endswith('_Opt'):
                            print(f"[INFO] Map '{map_name}' is not an _Opt variant; baked parked vehicles may remain. Try --town {map_name}_Opt if available.")
                    except Exception:
                        pass
            except Exception:
                pass

            
        else:
            self.world = self.client.get_world()
            try:
                if hasattr(carla, 'MapLayer'):
                    to_remove = (carla.MapLayer.ParkedVehicles | carla.MapLayer.Props)
                    self.world.unload_map_layer(to_remove)
                    try:
                        self.world.tick()
                    except Exception:
                        pass
                    try:
                        map_name = self.world.get_map().name
                        if not map_name.endswith('_Opt'):
                            print(f"[INFO] Map '{map_name}' is not an _Opt variant; baked parked vehicles may remain. Consider switching to an _Opt map.")
                    except Exception:
                        pass
            except Exception:
                pass

        self.original_settings = self.world.get_settings()
        self.tm = self.client.get_trafficmanager()
        self.map = self.world.get_map()
        print(f"Loaded map: {self.map.name}")

        try:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        except Exception:
            pass

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)

        # --- NPC management containers ---
        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
        self.collision_sensor = None
        self.collision_happened = False

        bp_lib = self.world.get_blueprint_library()
        try:
            ego_bp = bp_lib.filter('vehicle.taxi.ford')[0]
        except Exception:
            ego_bp = bp_lib.find('vehicle.tesla.model3')
        spawn = carla.Transform(carla.Location(x=95.70, y=66.00, z=2.00), carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0))
        ego = self.world.try_spawn_actor(ego_bp, spawn)
        if ego is None:
            ego = self.world.spawn_actor(ego_bp, random.choice(self.map.get_spawn_points()))
        ego.set_autopilot(False)
        self.ego = ego

        # Attach a simple collision sensor to log collisions for results.
        try:
            col_bp = bp_lib.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                col_bp,
                carla.Transform(),
                attach_to=self.ego,
            )

            def _on_collision(event):  # type: ignore[arg-type]
                other_name = 'unknown'
                impulse_mag = None
                try:
                    other = getattr(event, 'other_actor', None)
                    if other is not None:
                        other_name = getattr(other, 'type_id', str(other))
                except Exception:
                    pass
                try:
                    impulse = getattr(event, 'normal_impulse', None)
                    if impulse is not None:
                        impulse_mag = math.sqrt(float(impulse.x)**2 + float(impulse.y)**2 + float(impulse.z)**2)
                except Exception:
                    impulse_mag = None
                self.collision_happened = True
                self.collision_last_actor = other_name
                self.collision_last_impulse = impulse_mag
                self.collision_last_time = time.time()

                try:
                    if impulse_mag is not None:
                        print(f"[COLLISION] Contact with {other_name} | impulse={impulse_mag:.1f}")
                    else:
                        print(f"[COLLISION] Contact with {other_name}")
                except Exception:
                    pass

            self.collision_sensor.listen(_on_collision)
        except Exception:
            self.collision_sensor = None

        try:
            phys = self.ego.get_physics_control()
            for w in phys.wheels:
                w.max_brake_torque     = max(8000.0, getattr(w, 'max_brake_torque', 4000.0))
                w.max_handbrake_torque = max(12000.0, getattr(w, 'max_handbrake_torque', 8000.0))
                if apply_tire_friction:
                    w.tire_friction = mu
            self.ego.apply_physics_control(phys)
        except Exception:
            pass

        # --- Optionally spawn NPCs ---
        try:
            if npc_seed is not None:
                random.seed(int(npc_seed))
            if npc_vehicles > 0:
                self._spawn_npc_vehicles(npc_vehicles, npc_autopilot, int(max(0, min(100, npc_speed_diff_pct))))
            if npc_walkers > 0:
                self._spawn_npc_walkers(npc_walkers)
        except Exception as _e:
            pass

    def tick(self):
        return self.world.tick()

    def destroy(self):
        # Destroy collision sensor if present
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
        except Exception:
            pass
        # Stop walker controllers first
        try:
            for c in self.walker_controllers:
                try: c.stop()
                except Exception: pass
        except Exception:
            pass
        # Destroy walkers
        try:
            for w in self.npc_walkers:
                try: w.destroy()
                except Exception: pass
        except Exception:
            pass
        # Destroy NPC vehicles
        try:
            for v in self.npc_vehicles:
                try: v.destroy()
                except Exception: pass
        except Exception:
            pass
        try:
            if self.ego is not None:
                self.ego.destroy()
        except Exception:
            pass
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:
                pass
        except Exception:
            pass

    def lead_distance_ahead(self, lateral_max: float = LATERAL_MAX + 0.5, max_distance: float = 150.0) -> Optional[float]:
        """Return Euclidean distance to the closest actor ahead of the ego vehicle."""

        try:
            ego = self.ego
            if ego is None:
                return None
            ego_loc = ego.get_location()
            ego_rot = ego.get_transform().rotation
        except Exception:
            return None

        try:
            actors = self.world.get_actors()
        except Exception:
            return None

        yaw = math.radians(ego_rot.yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        best = None
        for actor in actors:
            try:
                if actor.id == ego.id:
                    continue
                tid = actor.type_id.lower()
                if not (tid.startswith('vehicle.') or tid.startswith('walker.')):
                    continue
                loc = actor.get_location()
            except Exception:
                continue
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            longitudinal = dx * cos_yaw + dy * sin_yaw
            if longitudinal <= 0.0:
                continue
            lateral = -dx * sin_yaw + dy * cos_yaw
            if abs(lateral) > lateral_max:
                continue
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > max_distance:
                continue
            if (best is None) or (dist < best):
                best = dist
        return best

    # ---- NPC helpers ----
    def _spawn_npc_vehicles(self, count: int, autopilot: bool, speed_diff_pct: int):
        bp_lib = self.world.get_blueprint_library()
        spawn_points = list(self.map.get_spawn_points())
        # Avoid ego spawn point
        if self.ego is not None:
            spawn_points = [sp for sp in spawn_points if sp.location.distance(self.ego.get_location()) > 2.0]
        random.shuffle(spawn_points)
        veh_bps = [bp for bp in bp_lib.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels').as_int()) >= 4]
        used = 0
        for sp in spawn_points:
            if used >= count: break
            bp = random.choice(veh_bps)
            try:
                v = self.world.try_spawn_actor(bp, sp)
                if v is None:
                    continue
                if autopilot:
                    try:
                        v.set_autopilot(True, self.tm.get_port())
                    except Exception:
                        v.set_autopilot(True)
                self.npc_vehicles.append(v)
                used += 1
            except Exception:
                pass
        # Traffic Manager global settings
        try:
            self.tm.global_percentage_speed_difference(speed_diff_pct)
            self.tm.set_random_device_seed(random.randint(0, 2**31-1))
        except Exception:
            pass

    def _spawn_npc_walkers(self, count: int):
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        spawned = 0
        while spawned < count:
            loc = self.world.get_random_location_from_navigation()
            if not loc:
                continue
            transform = carla.Transform(loc)
            bp = random.choice(walker_bps)
            try:
                w = self.world.try_spawn_actor(bp, transform)
                if w is None:
                    continue
                ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
                ctrl.start()
                # Send the walker to a random nav location and set speed
                try:
                    ctrl.go_to_location(self.world.get_random_location_from_navigation())
                    # Default walking speed ~1.4 m/s
                    ctrl.set_max_speed(1.4 + random.uniform(-0.3, 0.6))
                except Exception:
                    pass
                self.npc_walkers.append(w)
                self.walker_controllers.append(ctrl)
                spawned += 1
            except Exception:
                pass

# ========================= App ===========================
class _TelemetryLogger:
    def __init__(self, path: str, hz: float = 10.0):
        import csv
        self.path = path
        self.hz = max(0.1, float(hz))
        self.period = 1.0 / self.hz
        self._last_t = -1e9
        self._csv_f = open(self.path, 'w', newline='')
        self._w = csv.writer(self._csv_f)
        self._w.writerow([
            't','v_mps','tau_dyn','D_safety_dyn','sigma_depth','a_des','brake',
            'lambda_max','abs_factor','mu_est','mu_regime',
            'loop_ms','loop_ms_max','detect_ms','latency_ms','a_meas','x_rel_m','range_est_m',
            'ttc_s','gate_hit','gate_confirmed','false_stop_flag','brake_stage','brake_stage_factor',
            'tracker_s_m','tracker_rate_mps','lead_track_id','active_track_count',
            'sensor_ts','control_ts','sensor_to_control_ms',
            'actuation_ts','control_to_act_ms','sensor_to_act_ms'
        ])

    def maybe_log(self, t: float, v: float, tau_dyn: Optional[float], D_safety_dyn: Optional[float],
                  sigma_depth: Optional[float], a_des: Optional[float], brake: Optional[float],
                  lambda_max: Optional[float], abs_factor: Optional[float],
                  mu_est: Optional[float], mu_regime: Optional[str],
                  loop_ms: Optional[float], loop_ms_max: Optional[float],
                  detect_ms: Optional[float], latency_ms: Optional[float],
                  a_meas: Optional[float], x_rel_m: Optional[float], range_est_m: Optional[float],
                  ttc_s: Optional[float],
                  gate_hit: Optional[bool], gate_confirmed: Optional[bool],
                  false_stop_flag: Optional[bool],
                  brake_stage: Optional[int], brake_stage_factor: Optional[float],
                  tracker_s: Optional[float], tracker_rate: Optional[float],
                  tracker_id: Optional[int], tracker_count: Optional[int],
                  sensor_ts: Optional[float], control_ts: Optional[float],
                  sensor_to_control_ms: Optional[float],
                  actuation_ts: Optional[float],
                  control_to_act_ms: Optional[float],
                  sensor_to_act_ms: Optional[float]):
        if t - self._last_t >= self.period:
            self._last_t = t
            self._w.writerow([
                float(t), float(v),
                (None if tau_dyn is None else float(tau_dyn)),
                (None if D_safety_dyn is None else float(D_safety_dyn)),
                (None if sigma_depth is None else float(sigma_depth)),
                (None if a_des is None else float(a_des)),
                (None if brake is None else float(brake)),
                (None if lambda_max is None else float(lambda_max)),
                (None if abs_factor is None else float(abs_factor)),
                (None if mu_est is None else float(mu_est)),
                mu_regime,
                (None if loop_ms is None else float(loop_ms)),
                (None if loop_ms_max is None else float(loop_ms_max)),
                (None if detect_ms is None else float(detect_ms)),
                (None if latency_ms is None else float(latency_ms)),
                (None if a_meas is None else float(a_meas)),
                (None if x_rel_m is None else float(x_rel_m)),
                (None if range_est_m is None else float(range_est_m)),
                (None if ttc_s is None else float(ttc_s)),
                (None if gate_hit is None else int(bool(gate_hit))),
                (None if gate_confirmed is None else int(bool(gate_confirmed))),
                (None if false_stop_flag is None else int(bool(false_stop_flag))),
                (None if brake_stage is None else int(brake_stage)),
                (None if brake_stage_factor is None else float(brake_stage_factor)),
                (None if tracker_s is None else float(tracker_s)),
                (None if tracker_rate is None else float(tracker_rate)),
                (None if tracker_id is None else int(tracker_id)),
                (None if tracker_count is None else int(tracker_count)),
                (None if sensor_ts is None else float(sensor_ts)),
                (None if control_ts is None else float(control_ts)),
                (None if sensor_to_control_ms is None else float(sensor_to_control_ms)),
                (None if actuation_ts is None else float(actuation_ts)),
                (None if control_to_act_ms is None else float(control_to_act_ms)),
                (None if sensor_to_act_ms is None else float(sensor_to_act_ms)),
            ])

    def close(self):
        try:
            self._csv_f.close()
        except Exception:
            pass
class _ScenarioLogger:
    """High-level braking/scenario outcomes for thesis-style results.

    Each row captures one braking episode: initial speed/distance, minimum
    distance, whether the ego stopped, and time-to-stop.
    """

    def __init__(self, path: str):
        import csv
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._f = open(path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow([
            'scenario', 'trigger_kind', 'mu',
            'v_init_mps', 's_init_m', 's_min_m',
            's_init_gt_m', 's_min_gt_m',
            'stopped', 't_to_stop_s', 'collision',
            'range_margin_m', 'tts_margin_s',
            'ttc_init_s', 'ttc_min_s', 'reaction_time_s',
            'max_lambda', 'mean_abs_factor', 'false_stop'
        ])
        self._f.flush()

    def log(self, scenario: str, trigger_kind: str, mu: float,
            v_init: float, s_init: float, s_min: float,
            s_init_gt: Optional[float], s_min_gt: Optional[float],
            stopped: bool, t_to_stop: float, collision: bool,
            range_margin: Optional[float], tts_margin: Optional[float],
            ttc_init: Optional[float], ttc_min: Optional[float],
            reaction_time: Optional[float],
            max_lambda: Optional[float], mean_abs_factor: Optional[float],
            false_stop: bool):
        self._w.writerow([
            scenario,
            trigger_kind,
            float(mu),
            float(v_init),
            float(s_init),
            float(s_min),
            (None if s_init_gt is None else float(s_init_gt)),
            (None if s_min_gt is None else float(s_min_gt)),
            bool(stopped),
            float(t_to_stop),
            bool(collision),
            (None if range_margin is None else float(range_margin)),
            (None if tts_margin is None else float(tts_margin)),
            (None if ttc_init is None else float(ttc_init)),
            (None if ttc_min is None else float(ttc_min)),
            (None if reaction_time is None else float(reaction_time)),
            (None if max_lambda is None else float(max_lambda)),
            (None if mean_abs_factor is None else float(mean_abs_factor)),
            bool(false_stop),
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
class App:
    def __init__(self, args):
        self.args = args
        # Headless mode implies no OpenCV windows
        self.headless = bool(getattr(self.args, 'headless', False))
        if self.headless:
            self.args.no_depth_viz = True
        # Alias: --no-opencv or legacy --no-depth-viz
        if not hasattr(self.args, 'no_opencv'):
            self.args.no_opencv = False
        if getattr(self.args, 'no_depth_viz', False):
            self.args.no_opencv = True
        self.detector = YOLODetector(conf_thr=CONF_THR_DEFAULT,
                                      nms_thr=NMS_THR,
                                      img_size=self.args.yolo_img,
                                      device=self.args.yolo_device,
                                      use_half=self.args.yolo_half,
                                      agnostic=self.args.yolo_agnostic,
                                      classes=self.args.yolo_classes,
                                      max_det=self.args.yolo_max_det,
                                      dnn=self.args.yolo_dnn,
                                      augment=self.args.yolo_augment,
                                      per_class_iou_map=parse_per_class_iou_map(self.args.yolo_class_iou))
        self.telephoto_enabled = not getattr(self.args, 'no_telephoto', False)
        try:
            stride_raw = int(getattr(self.args, 'telephoto_stride', TELEPHOTO_STRIDE_DEFAULT))
        except Exception:
            stride_raw = TELEPHOTO_STRIDE_DEFAULT
        self.telephoto_stride = max(2, stride_raw)
        try:
            zoom_raw = float(getattr(self.args, 'telephoto_zoom', TELEPHOTO_DIGITAL_ZOOM_DEFAULT))
        except Exception:
            zoom_raw = TELEPHOTO_DIGITAL_ZOOM_DEFAULT
        self.telephoto_zoom = max(1.0, min(TELEPHOTO_DIGITAL_ZOOM_MAX, zoom_raw))
        self.telephoto_intrinsics = intrinsics_from_fov(TELEPHOTO_IMG_W, TELEPHOTO_IMG_H, TELEPHOTO_FOV_X_DEG)
        self._tl_state_history: deque = deque(maxlen=TL_STATE_SMOOTHING_FRAMES)
        self._telephoto_compute_totals: Dict[str, Any] = {
            'primary_calls': 0,
            'primary_time_s': 0.0,
            'telephoto_calls': 0,
            'telephoto_time_s': 0.0,
            'telephoto_invocations': 0,
            'telephoto_frames_considered': 0,
            'telephoto_skipped_disabled': 0,
            'telephoto_skipped_stride': 0,
            'telephoto_cache_hits': 0,
        }
        self.telephoto_compute_log_path = getattr(self.args, 'telephoto_compute_log', None)
        self.telephoto_compute_fp = None
        self.telephoto_compute_writer = None
        if self.telephoto_compute_log_path:
            try:
                self.telephoto_compute_fp = open(self.telephoto_compute_log_path, 'w', newline='')
                self.telephoto_compute_writer = csv.writer(self.telephoto_compute_fp)
                self.telephoto_compute_writer.writerow([
                    'timestamp', 'telephoto_enabled', 'telephoto_stride', 'telephoto_zoom',
                    'primary_calls', 'primary_total_ms',
                    'telephoto_invocations', 'telephoto_frames_considered',
                    'telephoto_calls', 'telephoto_total_ms',
                    'telephoto_cache_hits', 'telephoto_skipped_stride', 'telephoto_skipped_disabled',
                    'with_telephoto_total_ms', 'without_telephoto_total_ms'
                ])
            except Exception as e:
                print(f"[WARN] Failed to open telephoto compute log: {e}")
                self.telephoto_compute_log_path = None
                self.telephoto_compute_fp = None
                self.telephoto_compute_writer = None
        self._telephoto_last_candidate: Optional[Dict[str, Any]] = None
        self._telephoto_last_time: float = -1.0
        self._frame_index = 0
        self.class_conf_map = parse_per_class_conf_map(self.args.yolo_class_thr)
        self.engage_override_map = parse_engage_override_map(self.args.engage_override)
        self.min_h_map = parse_min_h_override_map(self.args.min_h_override)
        self.gate_frac_override = parse_gate_frac_override_map(self.args.gate_frac_override)
        self.gate_lat_override = parse_gate_lateral_override_map(self.args.gate_lateral_override)
        self.class_conf_map = parse_per_class_conf_map(self.args.yolo_class_thr)
        self.range = RangeEstimator(use_cuda=self.args.stereo_cuda, method=self.args.stereo_method)
        self.world: Optional[WorldManager] = None
        self.sensors: Optional[SensorRig] = None
        self.abs_mode: str = getattr(self.args, 'abs_mode', 'adaptive')
        if self.abs_mode == 'off':
            self.abs_actuator: Optional[PISlipABSActuator] = None
        elif self.abs_mode == 'fixed':
            self.abs_actuator = PISlipABSActuator(dt=DT)
        else:
            self.abs_actuator = AdaptivePISlipABSActuator(dt=DT)
        self._lead_tracker = LeadMultiObjectTracker(dt=DT)
        # Telemetry set up
        self.telemetry: Optional[_TelemetryLogger] = None
        if getattr(self.args, 'telemetry_csv', None):
            try:
                self.telemetry = _TelemetryLogger(self.args.telemetry_csv, getattr(self.args, 'telemetry_hz', 10.0))
            except Exception:
                self.telemetry = None
        # Scenario/braking logger
        self.scenario_logger: Optional[_ScenarioLogger] = None
        if getattr(self.args, 'scenario_csv', None):
            try:
                self.scenario_logger = _ScenarioLogger(self.args.scenario_csv)
            except Exception:
                self.scenario_logger = None

        # Optional video writer for qualitative figures
        self.video_writer = None
        self.video_out_path = getattr(self.args, 'video_out', None)
        if self.video_out_path and (not self.headless):
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_out_path, fourcc, 1.0/DT, (IMG_W, IMG_H))
            except Exception:
                self.video_writer = None

        # ROI tuning knobs for depth / stereo sampling
        try:
            self.depth_roi_shrink = max(0.0, min(0.9, float(getattr(self.args, 'depth_roi_shrink', DEPTH_ROI_SHRINK_DEFAULT))))
        except Exception:
            self.depth_roi_shrink = DEPTH_ROI_SHRINK_DEFAULT
        try:
            self.stereo_roi_shrink = max(0.0, min(0.9, float(getattr(self.args, 'stereo_roi_shrink', STEREO_ROI_SHRINK_DEFAULT))))
        except Exception:
            self.stereo_roi_shrink = STEREO_ROI_SHRINK_DEFAULT

        # Decision-layer tuning knobs
        try:
            self.min_aeb_speed = max(0.0, float(getattr(self.args, 'min_aeb_speed', V_AEB_MIN_DEFAULT)))
        except Exception:
            self.min_aeb_speed = V_AEB_MIN_DEFAULT
        try:
            self.gate_confirm_frames = max(1, int(getattr(self.args, 'gate_confirm_frames', GATE_CONFIRM_FRAMES_DEFAULT)))
        except Exception:
            self.gate_confirm_frames = GATE_CONFIRM_FRAMES_DEFAULT
        try:
            self.ttc_confirm_s = max(0.5, float(getattr(self.args, 'ttc_confirm_s', TTC_CONFIRM_S_DEFAULT)))
        except Exception:
            self.ttc_confirm_s = TTC_CONFIRM_S_DEFAULT
        try:
            self.ttc_stage_strong = max(0.1, float(getattr(self.args, 'ttc_stage_strong', TTC_STAGE_STRONG_DEFAULT)))
        except Exception:
            self.ttc_stage_strong = TTC_STAGE_STRONG_DEFAULT
        try:
            self.ttc_stage_full = max(0.05, float(getattr(self.args, 'ttc_stage_full', TTC_STAGE_FULL_DEFAULT)))
        except Exception:
            self.ttc_stage_full = TTC_STAGE_FULL_DEFAULT
        # Ensure ordering: full < strong <= confirm
        if self.ttc_stage_strong <= self.ttc_stage_full:
            self.ttc_stage_strong = self.ttc_stage_full + 0.05
        self.ttc_stage_strong = min(self.ttc_confirm_s, self.ttc_stage_strong)
        if self.ttc_stage_full >= self.ttc_stage_strong:
            self.ttc_stage_full = max(0.1, self.ttc_stage_strong - 0.05)
        try:
            self.stage_factor_comfort = max(0.1, min(0.95, float(getattr(self.args, 'aeb_stage_comfort', BRAKE_STAGE_COMFORT_FACTOR))))
        except Exception:
            self.stage_factor_comfort = BRAKE_STAGE_COMFORT_FACTOR
        try:
            self.stage_factor_strong = max(self.stage_factor_comfort + 0.01,
                                           min(0.99, float(getattr(self.args, 'aeb_stage_strong', BRAKE_STAGE_STRONG_FACTOR))))
        except Exception:
            self.stage_factor_strong = BRAKE_STAGE_STRONG_FACTOR
        self.stage_factor_full = BRAKE_STAGE_FULL_FACTOR
        try:
            self.aeb_ramp_up = max(0.5, float(getattr(self.args, 'aeb_ramp_up', AEB_RAMP_UP_DEFAULT)))
        except Exception:
            self.aeb_ramp_up = AEB_RAMP_UP_DEFAULT
        try:
            self.aeb_ramp_down = max(0.5, float(getattr(self.args, 'aeb_ramp_down', AEB_RAMP_DOWN_DEFAULT)))
        except Exception:
            self.aeb_ramp_down = AEB_RAMP_DOWN_DEFAULT

        self._gate_confirm_counter = 0
        self._hazard_confirm_since = -1.0
        self._aeb_a_des = 0.0

    def _draw_hud(self, screen, bgr, img_top, perf_fps, perf_ms, x, y, z, yaw, compass,
                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target,
                   collision_flag: bool,
                   abs_lambda: Optional[float] = None,
                   abs_factor: Optional[float] = None,
                   abs_mu: Optional[float] = None,
                   abs_regime: Optional[str] = None):
        if self.headless:
            return
        surf_front = bgr_to_pygame_surface(bgr)
        screen.blit(surf_front, (0, 0))
        if img_top is not None and not getattr(self.args, 'no_top_cam', False):
            surf_top = carla_image_to_surface(img_top)
            screen.blit(surf_top,   (IMG_W, 0))
        v_kmh = v * 3.6
        txt0 = f'ego @ x={x:8.2f}  y={y:8.2f}  z={z:6.2f}  | yaw={yaw:+6.1f}° {compass}'
        txt1 = f'Frame {frame_id} | v={v_kmh:5.1f} km/h | trigger={trigger_name or "None"} | TL={tl_state}'
        txt2 = f'thr={throttle:.2f}  brk={brake:.2f}  hold={hold_blocked}({hold_reason})  clear={no_trigger_elapsed:.1f}s  red_clear={no_red_elapsed:.1f}s  stopArmed={stop_armed}  ignoreT={(max(0.0, stop_release_ignore_until-sim_time)):.1f}s'
        txt_perf = f'FPS={perf_fps:.1f}  time={perf_ms:.1f}ms'
        shadow(screen, txt_perf, (10, IMG_H-156), (255,200,0))
        shadow(screen, txt0, (10, IMG_H-134), (200,200,255))
        shadow(screen, txt1, (10, IMG_H-90), (255,255,255))
        shadow(screen, txt2, (10, IMG_H-68), (0,255,160))
        if dbg_tau_dyn is not None:
            shadow(screen, f'tau={dbg_tau_dyn:0.2f}  Dsafe={dbg_D_safety_dyn:0.1f} m  sigma={dbg_sigma_depth:0.2f} m  gate={int(dbg_gate_hit)}  a_des={dbg_a_des:0.2f}  brk={dbg_brake:0.2f}  Vtgt={v_target*3.6:0.0f}km/h',
                   (10, IMG_H-24), (255,255,0))
        if abs_lambda is not None or abs_factor is not None or abs_mu is not None:
            slip_txt = 'n/a' if abs_lambda is None else f'{abs_lambda:.2f}'
            fac_txt = '1.00' if abs_factor is None else f'{abs_factor:.2f}'
            mu_txt = 'n/a' if abs_mu is None else f'{abs_mu:.2f}'
            regime_txt = abs_regime or 'n/a'
            shadow(screen,
                   f'ABS slip={slip_txt}  f={fac_txt}  mu_est={mu_txt}  regime={regime_txt}',
                   (10, IMG_H-46), (180,255,255))
        if collision_flag:
            shadow(screen, '*** COLLISION DETECTED ***', (IMG_W//4, IMG_H//2), (255,40,40))
        pygame.display.flip()

    def _get_wheel_linear_speeds(self) -> List[float]:
        if self.abs_actuator is None:
            return []
        if self.world is None or getattr(self.world, 'ego', None) is None:
            return []
        veh = self.world.ego
        radii: List[float] = []
        try:
            phys = veh.get_physics_control()
            wheels = getattr(phys, 'wheels', None)
            if wheels:
                for w in wheels:
                    try:
                        radii.append(max(0.05, float(getattr(w, 'radius', 0.35))))
                    except Exception:
                        radii.append(0.35)
        except Exception:
            pass

        def _coerce(val: Any) -> Optional[float]:
            if val is None:
                return None
            if isinstance(val, (list, tuple)) and val:
                try:
                    return float(val[0])
                except Exception:
                    return None
            if hasattr(val, 'x') and hasattr(val, 'y') and hasattr(val, 'z'):
                try:
                    return math.sqrt(float(val.x)**2 + float(val.y)**2 + float(val.z)**2)
                except Exception:
                    return None
            try:
                return float(val)
            except Exception:
                return None

        ang_vals: List[float] = []
        method = getattr(veh, 'get_wheel_angular_velocity', None)
        if callable(method):
            seq_vals = None
            try:
                seq = method()
                if isinstance(seq, (list, tuple)):
                    seq_vals = seq
            except TypeError:
                seq_vals = None
            except Exception:
                seq_vals = None
            if seq_vals is not None:
                for val in seq_vals:
                    coerced = _coerce(val)
                    if coerced is not None:
                        ang_vals.append(max(0.0, coerced))
            if not ang_vals:
                count = len(radii) if radii else 4
                for idx in range(count):
                    try:
                        val = method(idx)
                    except TypeError:
                        break
                    except Exception:
                        continue
                    coerced = _coerce(val)
                    if coerced is not None:
                        ang_vals.append(max(0.0, coerced))

        if not ang_vals:
            return []
        if not radii:
            radii = [0.34] * len(ang_vals)
        if len(radii) < len(ang_vals):
            radii.extend([radii[-1]] * (len(ang_vals) - len(radii)))
        return [max(0.0, ang * max(0.05, radius)) for ang, radius in zip(ang_vals, radii)]

    # --- IO: read sensors and decode ---
    def _read_frames(self, frames: Dict[str, Any]) -> Dict[str, Any]:
        img_front = frames['front']
        img_top   = frames.get('top', None)
        img_depth = frames.get('depth', None)
        img_tele  = frames.get('tele_rgb', None)
        img_tele_depth = frames.get('tele_depth', None)

        arr_front = np.frombuffer(img_front.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
        bgr = arr_front[:, :, :3].copy()

        if img_depth is not None:
            depth_bgra = np.frombuffer(img_depth.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
            depth_m = decode_depth_meters_from_bgra(depth_bgra)
        else:
            # No depth camera: fill with inf so depth-based queries naturally fall back
            depth_m = np.full((IMG_H, IMG_W), np.inf, dtype=np.float32)

        depth_stereo_m = None
        if (self.args.range_est == 'stereo'
            and 'stereo_left' in frames and 'stereo_right' in frames
            and self.range is not None and (self.range.stereo is not None or self.range.stereo_cuda is not None)):
            try:
                left_bgra  = np.frombuffer(frames['stereo_left'].raw_data,  dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
                right_bgra = np.frombuffer(frames['stereo_right'].raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))
                depth_stereo_m = self.range.stereo_depth(left_bgra, right_bgra)
            except Exception:
                depth_stereo_m = None

        tele_bgr = None
        if img_tele is not None:
            arr_tele = np.frombuffer(img_tele.raw_data, dtype=np.uint8).reshape((TELEPHOTO_IMG_H, TELEPHOTO_IMG_W, 4))
            tele_bgr = arr_tele[:, :, :3].copy()
        tele_depth_m = None
        if img_tele_depth is not None:
            tele_depth_bgra = np.frombuffer(img_tele_depth.raw_data, dtype=np.uint8).reshape((TELEPHOTO_IMG_H, TELEPHOTO_IMG_W, 4))
            tele_depth_m = decode_depth_meters_from_bgra(tele_depth_bgra)

        return {
            'bgr': bgr,
            'img_top': img_top,
            'depth_m': depth_m,
            'depth_stereo_m': depth_stereo_m,
            'tele_bgr': tele_bgr,
            'tele_depth_m': tele_depth_m,
        }

    # --- Perception: detection, distances, gating, TL/stop extraction ---
    def _perception_step(self, bgr: np.ndarray, depth_m: np.ndarray, depth_stereo_m: Optional[np.ndarray],
                         FX_: float, FY_: float, CX_: float, CY_: float,
                         sim_time: float, sensor_timestamp: Optional[float], v: float, MU: float,
                         log_both: bool, csv_w,
                         tele_bgr: Optional[np.ndarray] = None,
                         tele_depth_m: Optional[np.ndarray] = None) -> Dict[str, Any]:
        labels = self.detector.labels or {}
        detect_t0 = time.perf_counter()
        classIds, confs, boxes = self.detector.predict_raw(bgr)
        self._record_compute_time('primary', time.perf_counter() - detect_t0)

        nearest_s_active = None
        nearest_kind = None
        nearest_box = None
        nearest_thr = None
        nearest_conf = None
        stop_detected_current = False

        tl_det_s, tl_det_box, tl_det_state = None, None, 'UNKNOWN'
        tl_candidate_primary_near: Optional[Dict[str, Any]] = None
        tl_candidate_primary_any: Optional[Dict[str, Any]] = None
        tl_source = 'none'
        obstacle_measurements: List[Dict[str, Any]] = []

        det_points: List[Dict[str, Any]] = []
        depth_cache_depth: Dict[Tuple[int,int,int,int], Optional[float]] = {}
        depth_cache_stereo: Dict[Tuple[int,int,int,int], Optional[float]] = {}
        depth_roi = getattr(self, 'depth_roi_shrink', DEPTH_ROI_SHRINK_DEFAULT)
        stereo_roi = getattr(self, 'stereo_roi_shrink', STEREO_ROI_SHRINK_DEFAULT)
        want_depth_samples = (self.args.range_est in ('depth', 'both', 'stereo')) or log_both or getattr(self, '_log_stereo_cmp', False)
        want_stereo_samples = ((self.args.range_est in ('stereo', 'both')) or getattr(self, '_log_stereo_cmp', False) or log_both) and (depth_stereo_m is not None)

        primary_crop_top = int(TL_PRIMARY_CROP_FRAC * IMG_H)
        if len(classIds) != 0:
            cx0 = IMG_W/2.0
            for cid, conf, box in zip(np.array(classIds).flatten(), np.array(confs).flatten(), boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h
                name = labels.get(cid, str(cid))
                norm = _norm_label(name)

                conf_req = self.class_conf_map.get(norm, self.detector.conf_thr)
                if conf < conf_req:
                    continue

                h_req = int(self.min_h_map.get(norm, H_MIN_PX))
                is_tl_like = ('traffic' in norm and 'light' in norm)
                if (norm not in TRIGGER_NAMES_NORM and not is_tl_like) or (h < h_req):
                    if not (is_tl_like and h >= h_req):
                        continue

                xc = x1 + w/2.0
                H_real = OBJ_HEIGHT_M.get(name, OBJ_HEIGHT_M.get('traffic light') if is_tl_like else None)
                s_pinhole = (FY_ * float(H_real)) / float(h) if (H_real is not None and h > 0) else None

                s_depth = None
                box_key = (x1, y1, w, h)
                if want_depth_samples:
                    if box_key not in depth_cache_depth:
                        depth_cache_depth[box_key] = median_depth_in_box(depth_m, box_key, shrink=depth_roi)
                    s_depth = depth_cache_depth[box_key]
                s_stereo = None
                if want_stereo_samples:
                    if box_key not in depth_cache_stereo:
                        depth_cache_stereo[box_key] = median_depth_in_box(depth_stereo_m, box_key, shrink=stereo_roi)
                    s_stereo = depth_cache_stereo[box_key]

                fusion_src = 'none'
                if self.args.range_est == 'pinhole': s_use = s_pinhole
                elif self.args.range_est == 'depth': s_use = s_depth
                elif self.args.range_est == 'stereo':
                    s_use, fusion_src = fuse_depth_sources(s_depth, s_stereo, h)
                else: s_use = s_depth if (s_depth is not None) else s_pinhole
                if s_use is None:
                    continue

                frac_cls = float(self.gate_frac_override.get(norm, CENTER_BAND_FRAC))
                band_px_cls = frac_cls * IMG_W
                lateral_max_m = float(self.gate_lat_override.get(norm, LATERAL_MAX))
                do_gating = (norm in VEHICLE_CLASSES) or (norm in self.gate_frac_override) or (norm in self.gate_lat_override)

                lateral_ok = True
                if do_gating:
                    if self.args.range_est == 'stereo':
                        lateral_range = s_use if s_use is not None else (s_depth if s_depth is not None else (s_pinhole if s_pinhole is not None else 0.0))
                    else:
                        lateral_range = s_pinhole if s_pinhole is not None else (s_depth if s_depth is not None else (s_stereo if s_stereo is not None else 0.0))
                    if abs(xc - cx0) > band_px_cls:
                        lateral_ok = False
                    else:
                        lateral = ((xc - cx0) / max(1e-6, FX_)) * max(1.0, lateral_range)
                        if abs(lateral) > lateral_max_m:
                            lateral_ok = False
                if not lateral_ok:
                    continue

                tl_state_roi = 'UNKNOWN'
                if is_tl_like:
                    if y2 <= primary_crop_top:
                        continue
                    y_tl = max(y1, primary_crop_top)
                    h_tl = max(1, y2 - y_tl)
                    tl_state_roi = estimate_tl_color_from_roi(bgr, (x1, y_tl, w, h_tl))
                    if s_use is not None:
                        cand = {'distance': float(s_use), 'box': (x1, y1, w, h),
                                'state': tl_state_roi, 'source': 'primary'}
                        if (tl_candidate_primary_any is None) or (cand['distance'] < tl_candidate_primary_any['distance']):
                            tl_candidate_primary_any = cand
                        if cand['distance'] <= TL_PRIMARY_SHORT_RANGE_M:
                            if (tl_candidate_primary_near is None) or (cand['distance'] < tl_candidate_primary_near['distance']):
                                tl_candidate_primary_near = cand

                if (nearest_s_active is None) or (s_use < nearest_s_active):
                    nearest_s_active = s_use
                    nearest_box      = (x1, y1, w, h)
                    nearest_conf     = float(conf)
                    nearest_kind     = name

                kind = None; thr_for_kind = None
                if norm == 'stopsign':
                    kind = 'stop sign'; thr_for_kind = S_ENGAGE
                    if s_use <= S_ENGAGE:
                        stop_detected_current = True
                elif norm in PEDESTRIAN_CLASSES:
                    kind = name; thr_for_kind = S_ENGAGE_PED
                elif norm in VEHICLE_CLASSES:
                    kind = name; thr_for_kind = S_ENGAGE
                if thr_for_kind is not None:
                    try:
                        thr_for_kind = float(self.engage_override_map.get(norm, thr_for_kind))
                    except Exception:
                        pass

                color = (0,255,255)
                if norm == 'stopsign': color = (0,0,255)
                if is_tl_like:
                    label = f'{name} {tl_state_roi}'
                else:
                    label = f'{name}'
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                ann_parts = []
                if s_pinhole is not None: ann_parts.append(f"P:{s_pinhole:.1f}m")
                if s_depth   is not None: ann_parts.append(f"D:{s_depth:.1f}m")
                if s_stereo  is not None: ann_parts.append(f"S:{s_stereo:.1f}m")
                if self.args.range_est == 'stereo' and s_use is not None and fusion_src in ('fused', 'min'):
                    ann_parts.append(f"F:{s_use:.1f}m")
                cv2.putText(bgr, f'{label} '+ ' '.join(ann_parts), (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if kind is not None and thr_for_kind is not None and nearest_thr is None:
                    nearest_thr = thr_for_kind
                if kind is not None and kind != 'stop sign' and s_use is not None:
                    obstacle_measurements.append({
                        'distance': float(s_use),
                        'box': (x1, y1, w, h),
                        'kind': kind,
                        'timestamp': sensor_timestamp if (sensor_timestamp is not None and math.isfinite(sensor_timestamp)) else sim_time,
                        'confidence': float(conf),
                    })

                # HUD points (for distance overlay window)
                u, v_ = x1 + w/2.0, y1 + h/2.0
                box_key = (x1, y1, w, h)
                z = None
                if self.args.range_est == 'stereo' and depth_stereo_m is not None:
                    z = depth_cache_stereo.get(box_key)
                    if z is None:
                        z = median_depth_in_box(depth_stereo_m, box_key, shrink=stereo_roi)
                if z is None:
                    z = depth_cache_depth.get(box_key)
                if z is None:
                    z = median_depth_in_box(depth_m, box_key, shrink=depth_roi)
                if z is not None:
                    xyz = pixel_to_camera(u, v_, z, FX_, FY_, CX_, CY_)
                    if xyz is not None:
                        det_points.append({'name': name, 'box': (x1, y1, w, h), 'xyz': xyz, 'z': z})

                if getattr(self, '_log_both', False) and (getattr(self, '_csv_w', None) is not None):
                    s_pin = s_pinhole if s_pinhole is not None else float('nan')
                    s_dep = s_depth if s_depth is not None else float('nan')
                    s_st = s_stereo if s_stereo is not None else float('nan')
                    if s_pinhole is not None and s_depth is not None:
                        abs_diff = abs(s_pinhole - s_depth)
                    else:
                        abs_diff = float('nan')
                    sigma_val = getattr(self, '_sigma_depth_ema', None)
                    sigma_out = float(sigma_val) if (sigma_val is not None and np.isfinite(sigma_val)) else float('nan')
                    self._csv_w.writerow([
                        sim_time, name, x1, y1, w, h,
                        s_pin, s_dep, abs_diff,
                        s_st, v, MU, sigma_out
                    ])

                if getattr(self, '_log_stereo_cmp', False) and (getattr(self, '_stereo_csv_w', None) is not None):
                    stereo_val = s_stereo if s_stereo is not None else float('nan')
                    depth_val = s_depth if s_depth is not None else float('nan')
                    if s_stereo is not None and s_depth is not None:
                        err_val = (s_stereo - s_depth)
                    else:
                        err_val = float('nan')
                    pin_val = s_pinhole if s_pinhole is not None else float('nan')
                    self._stereo_csv_w.writerow([
                        sim_time, name, x1, y1, w, h,
                        stereo_val, depth_val, err_val, pin_val, v, MU
                    ])

        tl_candidate = tl_candidate_primary_near
        if tl_candidate is None:
            tele_cand = self._maybe_run_telephoto_tl(tele_bgr, tele_depth_m, depth_roi, sim_time)
            if tele_cand is not None:
                tl_candidate = tele_cand
        if tl_candidate is None and tl_candidate_primary_any is not None:
            tl_candidate = tl_candidate_primary_any
        if tl_candidate is not None:
            tl_det_s = tl_candidate.get('distance')
            tl_det_box = tl_candidate.get('box') if tl_candidate.get('source') == 'primary' else None
            tl_det_state = tl_candidate.get('state', 'UNKNOWN')
            tl_source = tl_candidate.get('source', 'primary')
        else:
            tl_det_s, tl_det_box, tl_det_state = None, None, 'UNKNOWN'
            tl_source = 'none'

        tl_det_state, tl_det_s = self._smooth_tl_state(tl_det_state, tl_det_s)

        # Merge with CARLA TL actor state
        tl_state_actor = 'UNKNOWN'
        try:
            tl_actor = self.world.ego.get_traffic_light()
            if tl_actor is not None:
                st = tl_actor.get_state()
                if st == carla.TrafficLightState.Red: tl_state_actor = 'RED'
                elif st == carla.TrafficLightState.Yellow: tl_state_actor = 'YELLOW'
                elif st == carla.TrafficLightState.Green: tl_state_actor = 'GREEN'
        except Exception:
            pass
        tl_state = tl_state_actor if tl_state_actor != 'UNKNOWN' else tl_det_state

        return {
            'bgr': bgr,
            'det_points': det_points,
            'nearest_s_active': nearest_s_active,
            'nearest_kind': nearest_kind,
            'nearest_thr': nearest_thr,
            'nearest_box': nearest_box,
            'nearest_conf': nearest_conf,
            'tl_state': tl_state,
            'tl_s_active': tl_det_s,
            'tl_det_box': tl_det_box,
            'tl_source': tl_source,
            'stop_detected_current': stop_detected_current,
            'obstacle_measurements': obstacle_measurements,
        }

    def _smooth_tl_state(self, state: str, distance: Optional[float]) -> Tuple[str, Optional[float]]:
        if not hasattr(self, '_tl_state_history') or self._tl_state_history.maxlen != TL_STATE_SMOOTHING_FRAMES:
            self._tl_state_history = deque(maxlen=TL_STATE_SMOOTHING_FRAMES)
        self._tl_state_history.append({'state': state, 'distance': distance})
        votes = [entry for entry in self._tl_state_history if entry['state'] != 'UNKNOWN']
        if votes:
            counts = Counter(entry['state'] for entry in votes)
            best_state, _ = counts.most_common(1)[0]
            dists = [entry['distance'] for entry in votes if entry['state'] == best_state and entry['distance'] is not None]
            best_dist = None
            if dists:
                best_dist = float(sum(dists) / len(dists))
            elif distance is not None:
                best_dist = distance
            return best_state, best_dist
        return state, distance

    def _record_compute_time(self, scope: str, duration_s: float):
        stats = getattr(self, '_telephoto_compute_totals', None)
        if stats is None or duration_s is None:
            return
        if scope == 'primary':
            stats['primary_calls'] += 1
            stats['primary_time_s'] += max(0.0, float(duration_s))
        elif scope == 'telephoto':
            stats['telephoto_calls'] += 1
            stats['telephoto_time_s'] += max(0.0, float(duration_s))

    def _maybe_write_telephoto_compute_summary(self):
        stats = getattr(self, '_telephoto_compute_totals', None)
        if not stats:
            return
        primary_ms = stats.get('primary_time_s', 0.0) * 1000.0
        telephoto_ms = stats.get('telephoto_time_s', 0.0) * 1000.0
        with_ms = primary_ms + telephoto_ms
        without_ms = primary_ms
        if stats.get('primary_calls', 0) == 0 and stats.get('telephoto_invocations', 0) == 0:
            return
        try:
            print(
                f"[TELEPHOTO] compute totals | primary={primary_ms:.1f}ms (calls={stats.get('primary_calls', 0)}) "
                f"telephoto={telephoto_ms:.1f}ms (runs={stats.get('telephoto_calls', 0)} stride_skips={stats.get('telephoto_skipped_stride', 0)}) "
                f"with={with_ms:.1f}ms vs without={without_ms:.1f}ms"
            )
        except Exception:
            pass
        writer = getattr(self, 'telephoto_compute_writer', None)
        fp = getattr(self, 'telephoto_compute_fp', None)
        if writer is None or fp is None:
            return
        timestamp = time.time()
        try:
            writer.writerow([
                f"{timestamp:.3f}",
                bool(self.telephoto_enabled),
                int(self.telephoto_stride),
                float(self.telephoto_zoom),
                stats.get('primary_calls', 0),
                f"{primary_ms:.3f}",
                stats.get('telephoto_invocations', 0),
                stats.get('telephoto_frames_considered', 0),
                stats.get('telephoto_calls', 0),
                f"{telephoto_ms:.3f}",
                stats.get('telephoto_cache_hits', 0),
                stats.get('telephoto_skipped_stride', 0),
                stats.get('telephoto_skipped_disabled', 0),
                f"{with_ms:.3f}",
                f"{without_ms:.3f}",
            ])
            fp.flush()
        except Exception:
            pass

    def _maybe_run_telephoto_tl(self, tele_bgr: Optional[np.ndarray], tele_depth_m: Optional[np.ndarray],
                                depth_roi: float, sim_time: float) -> Optional[Dict[str, Any]]:
        stats = getattr(self, '_telephoto_compute_totals', None)
        if stats is not None:
            stats['telephoto_invocations'] += 1
        if not self.telephoto_enabled or tele_bgr is None:
            if stats is not None:
                stats['telephoto_skipped_disabled'] += 1
            return None
        stride = max(2, int(self.telephoto_stride))
        if stats is not None:
            stats['telephoto_frames_considered'] += 1
        frame_idx = getattr(self, '_frame_index', 0)
        run_now = (frame_idx % stride) == 0
        cached = getattr(self, '_telephoto_last_candidate', None)
        if not run_now:
            if stats is not None:
                stats['telephoto_skipped_stride'] += 1
            if cached is not None and (sim_time - float(self._telephoto_last_time)) <= TELEPHOTO_CACHE_MAX_AGE_S:
                if stats is not None:
                    stats['telephoto_cache_hits'] += 1
                return cached
            return None

        infer_img = tele_bgr
        zoom_meta = None
        if self.telephoto_zoom > 1.0 + 1e-3:
            infer_img, zoom_meta = apply_digital_zoom(tele_bgr, self.telephoto_zoom)
        t0 = time.perf_counter()
        classIds, confs, boxes = self.detector.predict_raw(infer_img)
        self._record_compute_time('telephoto', time.perf_counter() - t0)
        labels = self.detector.labels or {}
        best: Optional[Dict[str, Any]] = None
        _, fy_t, _, _ = self.telephoto_intrinsics
        for cid, conf, box in zip(np.array(classIds).flatten(), np.array(confs).flatten(), boxes):
            x1, y1, w, h = map(int, box)
            name = labels.get(cid, str(cid))
            norm = _norm_label(name)
            if 'traffic' not in norm or 'light' not in norm:
                continue
            conf_req = self.class_conf_map.get(norm, self.detector.conf_thr)
            if conf < conf_req:
                continue
            if h <= 0 or w <= 0:
                continue
            full_box = remap_zoom_box((x1, y1, w, h), zoom_meta, tele_bgr.shape[1], tele_bgr.shape[0])
            x1_f, y1_f, w_f, h_f = full_box
            if h_f <= 0 or w_f <= 0:
                continue
            H_real = OBJ_HEIGHT_M.get(name, OBJ_HEIGHT_M.get('traffic light'))
            s_pinhole = (fy_t * float(H_real)) / float(h_f) if (H_real is not None and h_f > 0) else None
            s_depth = None
            if tele_depth_m is not None:
                s_depth = median_depth_in_box(tele_depth_m, full_box, shrink=depth_roi)
            s_use = s_depth if s_depth is not None else s_pinhole
            if s_use is None or not math.isfinite(s_use):
                continue
            tl_state_roi = estimate_tl_color_from_roi(tele_bgr, full_box)
            cand = {'distance': float(s_use), 'box': full_box,
                    'state': tl_state_roi, 'source': 'telephoto'}
            if best is None or cand['distance'] < best['distance']:
                best = cand

        self._telephoto_last_candidate = best
        self._telephoto_last_time = sim_time
        return best

    # --- Safety envelope: compute tau_dyn, D_safety_dyn, sigma_depth with smoothing ---
    def _safety_envelope(self, v: float, MU: float, ema_loop_ms: float,
                         nearest_box: Optional[Tuple[int,int,int,int]], nearest_conf: Optional[float],
                         depth_m: np.ndarray, depth_stereo_m: Optional[np.ndarray]) -> Tuple[float, float, float]:
        if not hasattr(self, '_sigma_depth_ema'): self._sigma_depth_ema = 0.40
        if not hasattr(self, '_D_safety_dyn_prev'): self._D_safety_dyn_prev = D_MIN
        extra_ms = float(getattr(self.args, 'extra_latency_ms', 0.0) or 0.0)
        latency_s = max(DT, (ema_loop_ms + extra_ms)/1000.0) + 0.03
        sd = None
        try:
            if nearest_box is not None:
                shrink_depth = getattr(self, 'depth_roi_shrink', 0.3)
                shrink_stereo = getattr(self, 'stereo_roi_shrink', 0.3)
                if self.args.range_est == 'stereo':
                    if depth_stereo_m is not None:
                        sd = depth_sigma_in_box(depth_stereo_m, nearest_box, shrink=shrink_stereo)
                    if sd is None:
                        sd = depth_sigma_in_box(depth_m, nearest_box, shrink=shrink_depth)
                elif self.args.range_est in ('depth', 'both'):
                    sd = depth_sigma_in_box(depth_m, nearest_box, shrink=shrink_depth)
        except Exception:
            sd = None
        if (sd is not None) and np.isfinite(sd): target = float(max(0.05, min(3.0, sd)))
        else: target = self._sigma_depth_ema
        delta = max(-0.50, min(0.50, target - self._sigma_depth_ema))
        self._sigma_depth_ema += 0.30 * delta
        sigma_depth = self._sigma_depth_ema

        c_near  = nearest_conf if (nearest_conf is not None) else 0.5
        mu_short = max(0.0, 0.90 - MU)
        tau_dyn = max(TAU_MIN, min(TAU_MAX, TAU_MIN + K_LAT_TAU*latency_s + K_MU_TAU*mu_short + K_UNC_TAU*(1.0 - c_near)))
        D_safety_dyn = D_MIN + K_LAT_D*(v*latency_s) + K_UNC_D*sigma_depth + K_MU_D*mu_short
        D_safety_dyn = self._D_safety_dyn_prev + max(-1.0, min(1.0, D_safety_dyn - self._D_safety_dyn_prev))
        self._D_safety_dyn_prev = D_safety_dyn
        return tau_dyn, D_safety_dyn, sigma_depth, latency_s

    # --- Control step: throttle/brake/hold ---
    def _control_step(self,
                      trigger_name: Optional[str],
                      nearest_s_active: Optional[float], nearest_thr: Optional[float],
                      tl_state: str, tl_s_active: Optional[float],
                      v: float, v_target: float, MU: float, ema_loop_ms: float,
                      last_s0: Optional[float],
                      stop_armed: bool, stop_latch_time: float, stop_release_ignore_until: float,
                      red_green_since: float, no_trigger_elapsed: float, no_red_elapsed: float,
                      depth_m: np.ndarray, depth_stereo_m: Optional[np.ndarray],
                      nearest_box: Optional[Tuple[int,int,int,int]], nearest_conf: Optional[float],
                      I_err: float, v_prev: Optional[float]) -> Tuple[float,float,Optional[carla.VehicleControl],bool,Optional[str],bool,float,float,Dict[str,Any],float]:
        A_MU = MU * 9.81
        throttle = 0.0; brake = 0.0; ctrl = None
        hold_blocked = getattr(self, '_hold_blocked', False)
        hold_reason  = getattr(self, '_hold_reason', None)

        reason_object = False
        reason_stop   = False
        reason_red    = False

        if (nearest_s_active is not None) and (nearest_thr is not None) and (nearest_s_active <= nearest_thr):
            reason_object = True
        elif stop_armed and (not hold_blocked):
            reason_stop = True
            if nearest_s_active is None and (last_s0 is not None):
                nearest_s_active = last_s0
        elif (tl_state == 'RED') and (tl_s_active is not None) and (tl_s_active <= S_ENGAGE_TL):
            reason_red = True
            nearest_s_active = tl_s_active

        # Pre-compute dynamic envelope + gate state if we have a candidate distance
        s_used = None
        tau_dyn = None
        D_safety_dyn = None
        sigma_depth = None
        latency_s = None
        required_dist_physics = None
        gate_hit = False
        tracked_rate = getattr(self, '_tracked_rate', None)
        s_eff = None
        if (nearest_s_active is not None) or (last_s0 is not None):
            s_used = last_s0 if last_s0 is not None else nearest_s_active
            tau_dyn, D_safety_dyn, sigma_depth, latency_s = self._safety_envelope(
                v, MU, ema_loop_ms, nearest_box, nearest_conf, depth_m, depth_stereo_m)
            required_dist_physics = (v*v)/(2.0*max(1e-3, A_MU)) + v*tau_dyn + D_safety_dyn
            gate_hit = (required_dist_physics >= s_used)
            if (tau_dyn is not None) and (D_safety_dyn is not None) and (s_used is not None):
                s_eff = max(EPS, s_used - D_safety_dyn - v * tau_dyn)

        ttc = None
        if s_used is not None:
            closing_speed = v
            if tracked_rate is not None and math.isfinite(tracked_rate) and tracked_rate < -0.05:
                closing_speed = max(0.1, -tracked_rate)
            closing_speed = max(0.1, closing_speed)
            ttc = s_used / closing_speed

        in_brake_band = reason_object or reason_stop or reason_red

        brake_reason = None
        if reason_stop:
            brake_reason = 'stop_sign'
        elif reason_red:
            brake_reason = 'red_light'
        elif reason_object:
            brake_reason = 'obstacle'
        if brake_reason is None and trigger_name:
            tnorm = trigger_name.lower()
            if 'stop' in tnorm and 'sign' in tnorm:
                brake_reason = 'stop_sign'
            elif 'traffic' in tnorm and 'light' in tnorm:
                brake_reason = 'red_light'

        if reason_stop:
            if trigger_name is None:
                trigger_name = 'stop sign'
            nearest_thr = S_ENGAGE
        if reason_red:
            nearest_thr = S_ENGAGE_TL
            trigger_name = 'traffic light (RED)'

        if in_brake_band and trigger_name and ('stop sign' in trigger_name) and (stop_release_ignore_until >= 0) and (self._sim_time < stop_release_ignore_until):
            in_brake_band = False

        gate_confirmed = gate_hit
        object_ready = reason_object
        if reason_object:
            if gate_hit:
                self._gate_confirm_counter = min(self._gate_confirm_counter + 1, self.gate_confirm_frames * 2)
            else:
                self._gate_confirm_counter = max(0, self._gate_confirm_counter - 1)
            gate_confirmed = self._gate_confirm_counter >= self.gate_confirm_frames
            ttc_ok = (ttc is None) or (ttc <= self.ttc_confirm_s)
            speed_ok = v >= self.min_aeb_speed
            object_ready = gate_confirmed and ttc_ok and speed_ok
        else:
            self._gate_confirm_counter = 0
            gate_confirmed = False
            object_ready = False

        in_brake_band = object_ready or reason_stop or reason_red

        brake_stage = 0
        stage_factor = 0.0
        if in_brake_band:
            if (ttc is None) or (ttc > self.ttc_stage_strong):
                brake_stage = 1
            if (ttc is not None) and (ttc <= self.ttc_stage_strong):
                brake_stage = 2
            if (ttc is not None) and (ttc <= self.ttc_stage_full):
                brake_stage = 3
            if reason_stop or reason_red:
                brake_stage = max(1, brake_stage)
            stage_map = {
                1: self.stage_factor_comfort,
                2: self.stage_factor_strong,
                3: self.stage_factor_full,
            }
            stage_factor = stage_map.get(brake_stage, 0.0)

        dbg_target = s_used if s_used is not None else nearest_s_active
        dbg = {'tau_dyn': tau_dyn, 'D_safety_dyn': D_safety_dyn, 'sigma_depth': sigma_depth, 'gate_hit': gate_hit,
               'a_des': None, 'brake': None, 'brake_active': in_brake_band,
                'brake_reason': brake_reason, 'brake_target_dist': dbg_target,
               'latency_s': latency_s, 'ttc': ttc, 'gate_confirmed': gate_confirmed,
               'brake_stage': brake_stage, 'brake_stage_factor': stage_factor}

        if in_brake_band and (s_used is not None) and (tau_dyn is not None) and (D_safety_dyn is not None):
            a_candidate = None
            if not gate_hit and s_eff is not None:
                a_candidate = min(A_MAX, (v*v) / (2.0 * max(EPS, s_eff)))
                a_candidate = min(a_candidate, A_MU)
            if a_candidate is None:
                a_candidate = A_MU if gate_hit else min(A_MU, A_MAX)
            stage_limit = None
            if brake_stage == 1:
                stage_limit = self.stage_factor_comfort * A_MU
            elif brake_stage == 2:
                stage_limit = self.stage_factor_strong * A_MU
            elif brake_stage >= 3:
                stage_limit = self.stage_factor_full * A_MU
            a_target = a_candidate if stage_limit is None else min(a_candidate, stage_limit)
            prev_a = getattr(self, '_aeb_a_des', 0.0)
            ramp_up = self.aeb_ramp_up * DT
            ramp_down = self.aeb_ramp_down * DT
            if a_target > prev_a:
                a_des = min(a_target, prev_a + ramp_up)
            else:
                a_des = max(a_target, prev_a - ramp_down)
            self._aeb_a_des = a_des
            brake_ff = max(0.0, min(1.0, a_des / A_MAX))
            a_meas = 0.0 if v_prev is None else max(0.0, (v_prev - v) / DT)
            e = max(0.0, a_des - a_meas)
            I_err = max(-I_MAX, min(I_MAX, I_err + e*DT))
            brake = max(0.0, min(1.0, brake_ff + (KPB*e + KIB*I_err)/A_MAX))

            if v < V_STOP:
                hold_blocked = True
                if trigger_name and 'traffic light' in trigger_name:
                    hold_reason = 'red_light'
                elif trigger_name and 'stop sign' in trigger_name:
                    hold_reason = 'stop_sign'; stop_latch_time = self._sim_time; stop_armed = False
            else:
                hold_reason = 'obstacle'

            dbg.update({'tau_dyn': tau_dyn, 'D_safety_dyn': D_safety_dyn, 'sigma_depth': sigma_depth,
                        'gate_hit': gate_hit, 'a_des': a_des, 'brake': brake,
                        'brake_reason': brake_reason, 'brake_target_dist': s_used if s_used is not None else nearest_s_active,
                        'latency_s': latency_s,
                        'ttc': ttc,
                        'gate_confirmed': gate_confirmed,
                        'brake_stage': brake_stage,
                        'brake_stage_factor': stage_factor,
                        'a_des_target': a_target,
                        'brake_active': True})

        elif hold_blocked:
            release = False
            if hold_reason == 'red_light':
                if tl_state == 'GREEN' and red_green_since >= 0 and (self._sim_time - red_green_since) >= CLEAR_DELAY_RED:
                    release = True
            elif hold_reason == 'stop_sign':
                if (self._sim_time - stop_latch_time) >= STOP_WAIT_S:
                    release = True
            elif hold_reason == 'obstacle':
                if no_trigger_elapsed >= CLEAR_DELAY_OBS:
                    release = True

            if release:
                hold_blocked = False; hold_reason = None; last_s0 = None
                if hasattr(self, '_lead_tracker') and self._lead_tracker is not None:
                    self._lead_tracker.deactivate()
                self._aeb_a_des = 0.0
                throttle, brake = 0.0, 0.0
                self._kick_until = self._sim_time + KICK_SEC
                stop_release_ignore_until = self._sim_time + 2.0
            else:
                throttle, brake = 0.0, 1.0
        else:
            e_v = v_target - v
            throttle = max(0.0, min(1.0, KP_THROTTLE * e_v))
            brake = 0.0
            self._aeb_a_des = 0.0
            if self._sim_time < getattr(self, '_kick_until', 0.0) and v < 0.3:
                throttle = max(throttle, KICK_THR)
            if not hold_blocked and v < 0.25:
                throttle = max(throttle, 0.35)

        self._hold_blocked = hold_blocked
        self._hold_reason = hold_reason
        return throttle, brake, ctrl, hold_blocked, hold_reason, stop_armed, stop_latch_time, stop_release_ignore_until, {'tau_dyn': dbg.get('tau_dyn'), 'D_safety_dyn': dbg.get('D_safety_dyn'), 'sigma_depth': dbg.get('sigma_depth'), 'gate_hit': dbg.get('gate_hit'), 'a_des': dbg.get('a_des'), 'brake': dbg.get('brake')}, I_err

    def _steer_to_waypoint(self, v: float) -> float:
        tr = self.world.ego.get_transform(); loc = tr.location
        yawr = math.radians(tr.rotation.yaw)
        wp = self.world.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        LA = max(6.0, min(12.0, 0.8 * max(v, 1.0)))
        next_wps = wp.next(LA) or wp.next(5.0)
        if not next_wps:
            return 0.0
        best, best_diff = None, 1e9
        for cand in next_wps:
            yaw_t = math.radians(cand.transform.rotation.yaw)
            diff = abs(wrap_pi(yaw_t - yawr))
            if diff < best_diff:
                best, best_diff = cand, diff
        tx, ty = best.transform.location.x, best.transform.location.y
        dx, dy = tx - loc.x, ty - loc.y
        angle_to_point = math.atan2(dy, dx)
        heading_error  = wrap_pi(angle_to_point - yawr)
        cross_track    = (-math.sin(yawr))*dx + (math.cos(yawr))*dy
        steer_cmd = heading_error + math.atan2(0.8 * cross_track, v + 1e-3)
        return max(-1.0, min(1.0, steer_cmd))

    # --- Window setup/cleanup helpers ---
    def _init_windows(self):
        pygame.init()
        self.screen = None
        if not self.headless:
            win_cols = 2 if not getattr(self.args, 'no_top_cam', False) else 1
            WIN_W, WIN_H = IMG_W * win_cols, IMG_H
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption('Nearestfirst + TL/StopSign | YOLO12n | Sync (OOP)')
        if not getattr(self.args, 'no_opencv', False) and not self.headless:
            try:
                cv2.namedWindow('DEPTH', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('DEPTH', IMG_W, IMG_H)
                cv2.moveWindow('DEPTH', 960, 540)
                cv2.namedWindow('HUD_DIST', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('HUD_DIST', IMG_W, IMG_H)
                cv2.moveWindow('HUD_DIST', 0, 540)
            except Exception:
                pass

    def _close_windows(self):
        try:
            if not getattr(self.args, 'no_opencv', False) and not self.headless:
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    def run(self):
        csv_f = None; csv_w = None
        stereo_csv_f = None; stereo_csv_w = None
        self._log_both = False
        self._log_stereo_cmp = False
        self._csv_w = None
        self._stereo_csv_w = None
        compare_csv_path = getattr(self.args, 'compare_csv', None)
        stereo_compare_path = getattr(self.args, 'stereo_compare_csv', None)

        self._init_windows()
        clock = pygame.time.Clock()
        if hasattr(self, '_lead_tracker') and self._lead_tracker is not None:
            self._lead_tracker.reset()

        self.world = WorldManager(
            self.args.host, self.args.port, self.args.town, self.args.mu, self.args.apply_tire_friction,
            npc_vehicles=getattr(self.args, 'npc_vehicles', 0),
            npc_walkers=getattr(self.args, 'npc_walkers', 0),
            npc_seed=getattr(self.args, 'npc_seed', None),
            npc_autopilot=(not getattr(self.args, 'npc_disable_autopilot', False)),
            npc_speed_diff_pct=getattr(self.args, 'npc_speed_diff_pct', 10)
        )
        # Respect camera toggles; if depth is disabled but requested for range, fall back to pinhole to avoid stalls
        used_range = self.args.range_est
        if used_range in ('depth', 'both') and getattr(self.args, 'no_depth_cam', False):
            used_range = 'pinhole'

        # One-time startup summary for quick sanity check
        try:
            print(
                f"Views: top={'ON' if not getattr(self.args, 'no_top_cam', False) else 'OFF'} "
                f"depth={'ON' if not getattr(self.args, 'no_depth_cam', False) else 'OFF'} "
                f"opencv={'ON' if not getattr(self.args, 'no_opencv', False) else 'OFF'}"
            )
            print(
                f"NPCs: vehicles={getattr(self.args, 'npc_vehicles', 0)} walkers={getattr(self.args, 'npc_walkers', 0)} "
                f"autopilot={'ON' if not getattr(self.args, 'npc_disable_autopilot', False) else 'OFF'} "
                f"tm_speed_diff={getattr(self.args, 'npc_speed_diff_pct', 10)}%"
            )
            print(
                f"YOLO: img={self.args.yolo_img} conf={self.detector.conf_thr:.2f} classes={self.args.yolo_classes or 'all'}"
            )
            print(
                f"Range: mode={used_range} stereo_cuda={self.args.stereo_cuda} method={self.args.stereo_method}"
            )
            print(
                f"Logging: interval={getattr(self.args, 'log_interval_frames', 0)} frames"
            )
        except Exception:
            pass
        self.args.range_est = used_range
        log_both = bool(compare_csv_path) and self.args.range_est in ('both', 'stereo')
        if log_both:
            import csv
            csv_f = open(compare_csv_path, 'w', newline='')
            csv_w = csv.writer(csv_f)
            csv_w.writerow(['t','cls','x','y','w','h','s_pinhole_m','s_depth_m','abs_diff_m','s_stereo_m','ego_v_mps','mu','sigma_depth'])
            self._log_both = True
            self._csv_w = csv_w
        log_stereo_cmp = bool(stereo_compare_path) and self.args.range_est == 'stereo'
        if log_stereo_cmp:
            import csv
            stereo_csv_f = open(stereo_compare_path, 'w', newline='')
            stereo_csv_w = csv.writer(stereo_csv_f)
            stereo_csv_w.writerow(['t','cls','x','y','w','h','s_stereo_m','s_depth_m','err_stereo_depth_m','s_pinhole_m','ego_v_mps','mu'])
            self._log_stereo_cmp = True
            self._stereo_csv_w = stereo_csv_w
        self.sensors = SensorRig(self.world.world, self.world.ego, used_range,
                                 enable_top=(not getattr(self.args, 'no_top_cam', False)),
                                 enable_depth=(not getattr(self.args, 'no_depth_cam', False)),
                                 enable_telephoto=self.telephoto_enabled)
        if self.args.range_est == 'stereo':
            self.range.ensure_stereo()

        FX_, FY_, CX_, CY_ = intrinsics_from_fov(IMG_W, IMG_H, FOV_X_DEG)

        MU = max(0.05, min(1.2, self.args.mu))
        A_MU = MU * 9.81
        v_target = float(V_TARGET)
        dist_total = 0.0
        perf_ms = 0.0; perf_fps = 0.0; ema_loop_ms = DT * 1000.0
        v_prev = None; I_err = 0.0
        loop_ms_max = 0.0
        sigma_depth_ema = 0.40
        sigma_depth_max_step = 0.50
        stop_persist_count = 0
        hold_blocked = False
        hold_reason   = None
        last_s0 = None
        tracked_rate = None
        prev_loc = None
        sim_time = 0.0
        sensor_timestamp = None
        kick_until = 0.0
        stop_latch_time = -1.0
        stop_armed = False
        stop_release_ignore_until = -1.0
        red_green_since = -1.0
        no_trigger_elapsed = 0.0
        no_red_elapsed = 0.0
        hud_msg = ''
        hud_until = 0.0
        conf_thr = CONF_THR_DEFAULT
        D_safety_dyn_prev = D_MIN

        dbg_tau_dyn = None
        dbg_D_safety_dyn = None
        dbg_sigma_depth = None
        dbg_gate_hit = False
        dbg_a_des = None
        dbg_brake = None

        # Latency measurement states
        pending_actuation: Optional[Dict[str, float]] = None
        last_actuation_ts: Optional[float] = None
        last_control_to_act_ms: Optional[float] = None
        last_sensor_to_act_ms: Optional[float] = None
        prev_brake_cmd = 0.0

        # Braking episode tracking for scenario-level results
        episode_active = False
        episode_trigger = None
        episode_v_init = 0.0
        episode_s_init = 0.0
        episode_s_min = float('inf')
        episode_s_init_gt = float('nan')
        episode_s_min_gt = float('inf')
        episode_lambda_max = 0.0
        episode_abs_factor_sum = 0.0
        episode_abs_factor_count = 0
        episode_false_flag = False
        episode_t_start = 0.0
        episode_ttc_init = None
        episode_ttc_min = None
        episode_reaction_time = None
        if not hasattr(self, '_prev_brake_activation'):
            self._prev_brake_activation = False
        if not hasattr(self, '_collision_logged_count'):
            self._collision_logged_count = 0
        collision_event_time = getattr(self, '_collision_last_time', -1.0)
        collision_logged_count = getattr(self, '_collision_logged_count', 0)
        self._gate_confirm_counter = 0
        self._hazard_confirm_since = -1.0

        try:
            t0 = time.time()
            while True:
                frame_id = self.world.tick()
                sim_time += DT
                self._sim_time = sim_time
                tic = time.time()

                frames = self.sensors.read(expected_frame=frame_id)
                sensor_timestamp = None
                front_frame = frames.get('front')
                if front_frame is not None:
                    sensor_timestamp = getattr(front_frame, 'timestamp', None)
                io = self._read_frames(frames)
                bgr = io['bgr']
                img_top = io['img_top']
                depth_m = io['depth_m']
                depth_stereo_m = io['depth_stereo_m']
                tele_bgr = io.get('tele_bgr')
                tele_depth_m = io.get('tele_depth_m')

                tr = self.world.ego.get_transform()
                loc = tr.location; rot = tr.rotation
                x, y, z = loc.x, loc.y, loc.z
                yaw = rot.yaw; compass = yaw_to_compass(yaw)
                vel = self.world.ego.get_velocity()
                v_raw = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                if prev_loc is None:
                    v_disp = 0.0
                else:
                    dx = loc.x - prev_loc.x; dy = loc.y - prev_loc.y; dz = loc.z - prev_loc.z
                    step = math.sqrt(dx*dx + dy*dy + dz*dz)
                    v_disp = step / DT
                    dist_total += step
                prev_loc = loc
                v = ALPHA_VBLEND * v_raw + (1.0 - ALPHA_VBLEND) * v_disp
                if v < 0.05: v = 0.0
                wheel_speeds = self._get_wheel_linear_speeds()
                a_long = 0.0 if v_prev is None else (v - v_prev) / DT

                # Check if a pending actuation measurement can be resolved
                if pending_actuation is not None:
                    control_ts = pending_actuation.get('control_ts', sim_time)
                    since_control = sim_time - float(control_ts)
                    if since_control > ACTUATION_TIMEOUT_S:
                        pending_actuation = None
                    elif a_long <= -ACTUATION_DECEL_THRESH:
                        last_actuation_ts = sim_time
                        last_control_to_act_ms = max(0.0, since_control * 1000.0)
                        sensor_ts_pending = pending_actuation.get('sensor_ts')
                        if sensor_ts_pending is not None and math.isfinite(sensor_ts_pending):
                            last_sensor_to_act_ms = max(0.0, (sim_time - float(sensor_ts_pending)) * 1000.0)
                        else:
                            last_sensor_to_act_ms = None
                        pending_actuation = None

                if not getattr(self.args, 'no_opencv', False) and not self.headless:
                    max_vis = 120.0
                    vis = np.clip(depth_m / max_vis, 0.0, 1.0)
                    vis8 = (vis * 255.0).astype(np.uint8)
                    vis8 = cv2.applyColorMap(vis8, cv2.COLORMAP_PLASMA)
                    cv2.imshow('DEPTH', vis8)
                # Perception step (YOLO + depth/stereo + gating)
                detect_ms = None
                detect_t0 = time.time()
                self._frame_index += 1
                perc = self._perception_step(bgr, depth_m, depth_stereo_m, FX_, FY_, CX_, CY_,
                                             sim_time, sensor_timestamp, v, MU, log_both, csv_w,
                                             tele_bgr=tele_bgr, tele_depth_m=tele_depth_m)
                detect_ms = (time.time() - detect_t0) * 1000.0
                bgr = perc['bgr']
                det_points = perc['det_points']
                nearest_s_active = perc['nearest_s_active']
                nearest_kind = perc['nearest_kind']
                nearest_thr = perc['nearest_thr']
                nearest_box = perc['nearest_box']
                nearest_conf = perc['nearest_conf']
                tl_state = perc['tl_state']
                tl_s_active = perc['tl_s_active']
                tl_det_box = perc['tl_det_box']
                stop_detected_current = perc['stop_detected_current']
                any_red_tl = (tl_state == 'RED')

                x_rel_gt = None
                if self.world is not None:
                    try:
                        x_rel_gt = self.world.lead_distance_ahead()
                    except Exception:
                        x_rel_gt = None

                if any_red_tl and (tl_s_active is not None):
                    if (nearest_s_active is None) or (tl_s_active < nearest_s_active):
                        nearest_s_active = tl_s_active
                        nearest_kind = 'traffic light (RED)'
                        nearest_thr = S_ENGAGE_TL
                        nearest_box = tl_det_box
                        nearest_conf = 0.9 if nearest_conf is None else max(0.9, nearest_conf)

                # Optional conservative behavior for UNKNOWN TL state
                if (getattr(self.args, 'tl_unknown_conservative', False)
                    and tl_state == 'UNKNOWN' and (tl_s_active is not None)):
                    if (nearest_s_active is None) or (tl_s_active < nearest_s_active):
                        if tl_s_active <= S_ENGAGE_TL:
                            nearest_s_active = tl_s_active
                            nearest_kind = 'traffic light (UNKNOWN)'
                            nearest_thr = S_ENGAGE_TL
                            nearest_box = tl_det_box
                            nearest_conf = 0.6 if nearest_conf is None else max(0.6, nearest_conf)

                if not getattr(self.args, 'no_opencv', False) and not self.headless:
                    hud_dist = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                    y_text = 24
                    mode_label = 'depth camera'
                    if self.args.range_est == 'pinhole': mode_label = 'pinhole'
                    elif self.args.range_est == 'stereo': mode_label = 'stereo camera'
                    cv2.putText(hud_dist, f'Ego→Object distances ({mode_label}, meters):', (10, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_text += 28
                    det_points.sort(key=lambda d: d['z'])
                    if det_points:
                        for d in det_points[:22]:
                            xcam, ycam, zcam = d['xyz']
                            cv2.putText(hud_dist, f"{d['name']:<14}  Z={zcam:5.1f} m   X={xcam:+4.1f} m   Y={ycam:+4.1f} m",
                                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0), 1)
                            y_text += 22
                    else:
                        cv2.putText(hud_dist, 'No detections', (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
                    cv2.imshow('HUD_DIST', hud_dist)
                if stop_detected_current: stop_persist_count += 1
                else: stop_persist_count = 0
                if (not stop_armed) and (stop_persist_count >= self.args.persist_frames):
                    stop_armed = True; stop_latch_time = sim_time

                if nearest_s_active is None: no_trigger_elapsed += DT
                else: no_trigger_elapsed = 0.0

                if any_red_tl: no_red_elapsed = 0.0; red_green_since = -1.0
                else: no_red_elapsed += DT
                if (tl_state == 'GREEN'):
                    if red_green_since < 0:
                        red_green_since = sim_time
                else:
                    red_green_since = -1.0

                tracker_state = None
                tracker_active_states: List[Dict[str, Any]] = []
                tracked_distance_for_control = None
                tracked_rate = None
                tracker = getattr(self, '_lead_tracker', None)
                if tracker is not None:
                    obstacle_measurements = perc.get('obstacle_measurements', [])
                    tracker_state, tracker_active_states = tracker.step(sim_time, obstacle_measurements)
                    if tracker_state is not None:
                        tracked_distance_for_control = tracker_state.get('distance')
                        tracked_rate = tracker_state.get('rate')
                lead_track_id = tracker_state.get('track_id') if tracker_state is not None else None
                lead_track_count = len(tracker_active_states)
                if tracked_distance_for_control is None:
                    if nearest_s_active is not None:
                        tracked_distance_for_control = nearest_s_active
                    else:
                        tracked_distance_for_control = last_s0
                self._tracked_distance = tracked_distance_for_control
                self._tracked_rate = tracked_rate

                trigger_name = nearest_kind

                # Control step via helper
                throttle, brake, ctrl, hold_blocked, hold_reason, stop_armed, stop_latch_time, stop_release_ignore_until, dbg_map, I_err = \
                    self._control_step(trigger_name, nearest_s_active, nearest_thr,
                                       tl_state, tl_s_active, v, v_target, MU, ema_loop_ms,
                                       tracked_distance_for_control, stop_armed, stop_latch_time, stop_release_ignore_until,
                                       red_green_since, no_trigger_elapsed, no_red_elapsed,
                                       depth_m, depth_stereo_m, nearest_box, nearest_conf,
                                       I_err, v_prev)
                abs_lambda = None
                abs_factor = None
                abs_mu = None
                abs_regime = 'off' if self.abs_actuator is None else None
                abs_dbg = None
                if self.abs_actuator is not None:
                    try:
                        brake = self.abs_actuator.step(brake, v, wheel_speeds, a_long)
                        abs_dbg = self.abs_actuator.debug_metrics()
                    except Exception:
                        abs_dbg = None
                    if abs_dbg is not None:
                        abs_lambda = abs_dbg.get('lambda_max')
                        abs_factor = abs_dbg.get('f_global')
                        abs_mu = abs_dbg.get('mu_est')
                        abs_regime = abs_dbg.get('regime')

                # If a new brake command crosses the threshold, start measuring actuation latency
                brake_cmd_for_latency = float(brake)
                if (brake_cmd_for_latency >= ACTUATION_BRAKE_CMD_MIN and
                        prev_brake_cmd < ACTUATION_BRAKE_CMD_MIN):
                    pending_actuation = {
                        'sensor_ts': sensor_timestamp if (sensor_timestamp is not None and math.isfinite(sensor_timestamp)) else sim_time,
                        'control_ts': sim_time,
                        'cmd_brake': brake_cmd_for_latency,
                    }
                elif brake_cmd_for_latency < (0.5 * ACTUATION_BRAKE_CMD_MIN):
                    pending_actuation = None
                prev_brake_cmd = brake_cmd_for_latency

                dbg_map['brake'] = brake
                dbg_map['abs_lambda_max'] = abs_lambda
                dbg_map['abs_factor'] = abs_factor
                dbg_map['abs_mu'] = abs_mu
                dbg_map['abs_regime'] = abs_regime
                dbg_tau_dyn = dbg_map.get('tau_dyn')
                dbg_D_safety_dyn = dbg_map.get('D_safety_dyn')
                dbg_sigma_depth = dbg_map.get('sigma_depth')
                dbg_gate_hit = dbg_map.get('gate_hit')
                dbg_gate_confirmed = dbg_map.get('gate_confirmed')
                dbg_a_des = dbg_map.get('a_des')
                dbg_brake = dbg_map.get('brake')
                dbg_ttc = dbg_map.get('ttc')
                dbg_brake_stage = dbg_map.get('brake_stage')
                dbg_brake_stage_factor = dbg_map.get('brake_stage_factor')
                dbg_abs_lambda = dbg_map.get('abs_lambda_max')
                dbg_abs_factor = dbg_map.get('abs_factor')
                dbg_abs_mu = dbg_map.get('abs_mu')
                dbg_abs_regime = dbg_map.get('abs_regime')
                brake_active = bool(dbg_map.get('brake_active'))
                brake_reason = dbg_map.get('brake_reason')
                brake_target = dbg_map.get('brake_target_dist')
                dbg_latency_s = dbg_map.get('latency_s')
                dbg_latency_ms = None if (dbg_latency_s is None) else (float(dbg_latency_s) * 1000.0)
                if tracker_state is not None and tracker_state.get('distance') is not None:
                    last_s0 = tracker_state.get('distance')
                elif tracked_distance_for_control is not None:
                    last_s0 = tracked_distance_for_control
                else:
                    last_s0 = None
                if (not brake_reason) and hold_blocked and hold_reason in ('stop_sign','red_light','obstacle'):
                    brake_reason = hold_reason
                hazard_since_prev = getattr(self, '_hazard_confirm_since', -1.0)
                if bool(dbg_gate_confirmed):
                    if hazard_since_prev < 0.0:
                        self._hazard_confirm_since = sim_time
                else:
                    self._hazard_confirm_since = -1.0
                current_dist = None
                for cand in (brake_target, nearest_s_active, tl_s_active if tl_state == 'RED' else None, last_s0):
                    if cand is None:
                        continue
                    try:
                        val = float(cand)
                    except Exception:
                        continue
                    if not math.isfinite(val):
                        continue
                    current_dist = val
                    break

                range_est = current_dist
                false_stop_flag = False
                if brake_active:
                    gate_state = bool(dbg_gate_hit)
                    if x_rel_gt is None or not math.isfinite(x_rel_gt):
                        false_stop_flag = True
                    else:
                        if not gate_state:
                            if dbg_D_safety_dyn is not None and math.isfinite(dbg_D_safety_dyn):
                                margin_val = x_rel_gt - dbg_D_safety_dyn
                                if margin_val > FALSE_STOP_MARGIN_M:
                                    false_stop_flag = True
                            if v > 0.1:
                                ttc = x_rel_gt / max(v, 0.1)
                                if ttc > FALSE_STOP_TTC_S:
                                    false_stop_flag = True
                else:
                    false_stop_flag = False

                tracker_distance_logged = tracker_state.get('distance') if tracker_state is not None else None
                tracker_rate_logged = tracked_rate
                tracker_id_logged = tracker_state.get('track_id') if tracker_state is not None else None
                tracker_count_logged = lead_track_count
                control_timestamp = sim_time
                sensor_to_control_ms = None
                if sensor_timestamp is not None and math.isfinite(sensor_timestamp):
                    sensor_to_control_ms = max(0.0, (control_timestamp - float(sensor_timestamp)) * 1000.0)

                # --- Episode bookkeeping: start/end of braking events ---
                reason_allowed = brake_reason in ('obstacle','stop_sign','red_light')
                hold_brake = hold_blocked and hold_reason in ('stop_sign','red_light','obstacle')
                activation = reason_allowed and (brake_active or hold_brake)
                prev_activation = bool(getattr(self, '_prev_brake_activation', False))
                if (not episode_active) and activation and (not prev_activation):
                    start_dist = current_dist
                    if start_dist is None:
                        if last_s0 is not None:
                            start_dist = float(last_s0)
                        elif nearest_thr is not None:
                            start_dist = float(nearest_thr)
                        else:
                            start_dist = float(S_ENGAGE)
                    episode_active = True
                    if brake_reason == 'stop_sign':
                        episode_trigger = 'stop sign'
                    elif brake_reason == 'red_light':
                        episode_trigger = 'traffic light (RED)'
                    else:
                        episode_trigger = trigger_name or 'unknown'
                    episode_v_init = v
                    episode_s_init = float(start_dist)
                    episode_s_min = float(start_dist)
                    episode_s_init_gt = float(x_rel_gt) if (x_rel_gt is not None and math.isfinite(x_rel_gt)) else float(start_dist)
                    episode_s_min_gt = float(episode_s_init_gt)
                    episode_lambda_max = abs_lambda if (abs_lambda is not None) else 0.0
                    episode_abs_factor_sum = 0.0
                    episode_abs_factor_count = 0
                    episode_false_flag = False
                    episode_t_start = sim_time
                    if dbg_ttc is not None and math.isfinite(dbg_ttc):
                        episode_ttc_init = float(dbg_ttc)
                        episode_ttc_min = float(dbg_ttc)
                    else:
                        episode_ttc_init = None
                        episode_ttc_min = None
                    hazard_since = getattr(self, '_hazard_confirm_since', -1.0)
                    if hazard_since >= 0.0:
                        episode_reaction_time = max(0.0, sim_time - hazard_since)
                    else:
                        episode_reaction_time = None
                elif episode_active:
                    if current_dist is not None:
                        episode_s_min = min(episode_s_min, current_dist)
                    if x_rel_gt is not None and math.isfinite(x_rel_gt):
                        episode_s_min_gt = min(episode_s_min_gt, x_rel_gt)
                    if abs_lambda is not None:
                        episode_lambda_max = max(episode_lambda_max, abs_lambda)
                    if abs_factor is not None:
                        episode_abs_factor_sum += abs_factor
                        episode_abs_factor_count += 1
                    if false_stop_flag:
                        episode_false_flag = True
                    if dbg_ttc is not None and math.isfinite(dbg_ttc):
                        if episode_ttc_min is None:
                            episode_ttc_min = float(dbg_ttc)
                        else:
                            episode_ttc_min = min(episode_ttc_min, float(dbg_ttc))
                    if episode_reaction_time is None and (brake_active or hold_brake):
                        hazard_since = getattr(self, '_hazard_confirm_since', -1.0)
                        if hazard_since >= 0.0:
                            episode_reaction_time = max(0.0, sim_time - hazard_since)
                    still_active = activation
                    # Episode considered finished once vehicle has effectively stopped
                    # or braking band / hold is released (returned to cruising).
                    if (v < V_STOP) or (not still_active):
                        stopped = (v < V_STOP)
                        t_to_stop = max(0.0, sim_time - episode_t_start)
                        if self.scenario_logger is not None:
                            try:
                                s_init_logged = episode_s_init
                                s_min_logged = episode_s_min if math.isfinite(episode_s_min) else episode_s_init
                                s_init_gt_logged = episode_s_init_gt if math.isfinite(episode_s_init_gt) else None
                                s_min_gt_logged = episode_s_min_gt if math.isfinite(episode_s_min_gt) else None
                                if (s_init_gt_logged is not None) and (s_min_gt_logged is not None):
                                    range_margin = s_init_gt_logged - s_min_gt_logged
                                elif math.isfinite(s_init_logged) and math.isfinite(s_min_logged):
                                    range_margin = s_init_logged - s_min_logged
                                else:
                                    range_margin = None
                                tts_margin = None
                                if stopped:
                                    theoretical = episode_v_init / max(0.1, MU * 9.81)
                                    tts_margin = t_to_stop - theoretical
                                ttc_init_logged = episode_ttc_init if (episode_ttc_init is not None and math.isfinite(episode_ttc_init)) else None
                                ttc_min_logged = episode_ttc_min if (episode_ttc_min is not None and math.isfinite(episode_ttc_min)) else None
                                reaction_logged = episode_reaction_time if (episode_reaction_time is not None and math.isfinite(episode_reaction_time)) else None
                                max_lambda_val = episode_lambda_max if episode_lambda_max > 0 else None
                                mean_abs_factor = None
                                if episode_abs_factor_count > 0:
                                    mean_abs_factor = episode_abs_factor_sum / episode_abs_factor_count
                                collision_state = bool(self.world.collision_happened if self.world is not None else False)
                                false_stop_episode = bool(episode_false_flag or ((not collision_state) and (range_margin is not None) and (range_margin > FALSE_STOP_MARGIN_M)))
                                self.scenario_logger.log(
                                    getattr(self.args, 'scenario_tag', 'default'),
                                    str(episode_trigger or 'unknown'),
                                    MU,
                                    episode_v_init,
                                    s_init_logged,
                                    s_min_logged,
                                    s_init_gt_logged,
                                    s_min_gt_logged,
                                    bool(stopped),
                                    t_to_stop,
                                    collision_state,
                                    range_margin,
                                    tts_margin,
                                    ttc_init_logged,
                                    ttc_min_logged,
                                    reaction_logged,
                                    max_lambda_val,
                                    mean_abs_factor,
                                    false_stop_episode,
                                )
                            except Exception:
                                pass
                        episode_active = False
                        episode_trigger = None
                        self._collision_logged_count = getattr(self, '_collision_logged_count', 0)
                        episode_lambda_max = 0.0
                        episode_abs_factor_sum = 0.0
                        episode_abs_factor_count = 0
                        episode_false_flag = False
                        episode_ttc_init = None
                        episode_ttc_min = None
                        episode_reaction_time = None

                self._prev_brake_activation = activation

                steer = self._steer_to_waypoint(v)

                if ctrl is not None:
                    ctrl.steer = steer
                    self.world.ego.apply_control(ctrl)
                else:
                    self.world.ego.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer=float(steer), hand_brake=False))

                # Optional video recording of front RGB (for qualitative figures)
                if self.video_writer is not None and not self.headless:
                    try:
                        self.video_writer.write(bgr)
                    except Exception:
                        pass

                collision_flag = bool(getattr(self.world, 'collision_happened', False))
                if not self.headless:
                    self._draw_hud(self.screen, bgr, img_top, perf_fps, perf_ms, x, y, z, yaw, compass,
                                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target,
                                   collision_flag, dbg_abs_lambda, dbg_abs_factor, dbg_abs_mu, dbg_abs_regime)

                # Periodic concise console log for tuning
                logN = int(getattr(self.args, 'log_interval_frames', 0) or 0)
                if logN > 0 and (frame_id % logN == 0):
                    print(f"frame={frame_id} fps={perf_fps:.1f} v={v:4.1f} Vtgt={v_target:4.1f} mu={MU:.2f} "
                          f"mode={self.args.range_est} conf={self.detector.conf_thr:.2f} TL={tl_state} trig={trigger_name or 'None'} "
                          f"s_act={(None if nearest_s_active is None else f'{nearest_s_active:.1f}')} thr={(None if nearest_thr is None else f'{nearest_thr:.1f}')} "
                          f"thr_cmd={throttle:.2f} brk={brake:.2f} hold={hold_blocked} collision={collision_flag}")

                if collision_flag and self.scenario_logger is not None:
                    last_log_time = getattr(self, '_collision_last_log_time', None)
                    sensor_time = getattr(self.world, 'collision_last_time', None)
                    # Avoid duplicate writes if sensor reports multiple times per frame
                    should_log_collision = False
                    if sensor_time is None:
                        should_log_collision = last_log_time is None
                    else:
                        should_log_collision = (last_log_time is None) or (abs(sensor_time - last_log_time) > 1e-3)
                    if should_log_collision:
                        try:
                            self.scenario_logger.log(
                                getattr(self.args, 'scenario_tag', 'default'),
                                'collision',
                                MU,
                                float(v),
                                float(nearest_s_active or 0.0),
                                float(nearest_s_active or 0.0),
                                bool(v < V_STOP),
                                0.0,
                                True,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                False,
                            )
                        except Exception:
                            pass
                        self._collision_last_log_time = sensor_time if sensor_time is not None else sim_time
                        self._collision_logged_count = getattr(self, '_collision_logged_count', 0) + 1

                loop_ms = (time.time() - tic) * 1000.0
                perf_ms = loop_ms
                perf_fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
                ema_loop_ms = 0.9*ema_loop_ms + 0.1*loop_ms
                loop_ms_max = max(loop_ms_max, loop_ms)
                v_prev = v
                if not getattr(self.args, 'no_opencv', False) and not self.headless:
                    cv2.waitKey(1)

                if not self.headless:
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        elif e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_ESCAPE:
                                raise KeyboardInterrupt
                            elif e.key == pygame.K_LEFTBRACKET:
                                self.detector.conf_thr = max(0.05, round(self.detector.conf_thr - 0.05, 2))
                                hud_msg = f'conf -> {self.detector.conf_thr:.2f}'; hud_until = sim_time + 2.0
                            elif e.key == pygame.K_RIGHTBRACKET:
                                self.detector.conf_thr = min(0.99, round(self.detector.conf_thr + 0.05, 2))
                                hud_msg = f'conf -> {self.detector.conf_thr:.2f}'; hud_until = sim_time + 2.0
                            elif e.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                                v_target = min(40.0, v_target + 0.5556)
                                hud_msg = f'Vtgt -> {v_target*3.6:.0f} km/h'; hud_until = sim_time + 2.0
                            elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                                v_target = max(2.0, v_target - 0.5556)
                                hud_msg = f'Vtgt -> {v_target*3.6:.0f} km/h'; hud_until = sim_time + 2.0
                            elif e.key == pygame.K_0:
                                v_target = float(V_TARGET)
                                hud_msg = f'Vtgt reset -> {V_TARGET*3.6:.0f} km/h'; hud_until = sim_time + 2.0
                            elif e.key == pygame.K_s:
                                # Save a snapshot of current front RGB for figures
                                try:
                                    snap_name = f'snapshot_{frame_id}.png'
                                    cv2.imwrite(snap_name, bgr)
                                    hud_msg = f'Snapshot saved: {snap_name}'
                                    hud_until = sim_time + 2.0
                                except Exception:
                                    pass

                # Telemetry
                if self.telemetry is not None:
                    try:
                        self.telemetry.maybe_log(sim_time, v, dbg_tau_dyn, dbg_D_safety_dyn, dbg_sigma_depth,
                                                 dbg_a_des, dbg_brake,
                                                 dbg_abs_lambda, dbg_abs_factor, dbg_abs_mu, dbg_abs_regime,
                                                 loop_ms, loop_ms_max, detect_ms, dbg_latency_ms,
                                                 a_long, x_rel_gt, range_est,
                                                 dbg_ttc,
                                                 dbg_gate_hit, dbg_gate_confirmed,
                                                 false_stop_flag,
                                                 dbg_brake_stage, dbg_brake_stage_factor,
                                                 tracker_distance_logged, tracker_rate_logged,
                                                 tracker_id_logged, tracker_count_logged,
                                                 sensor_timestamp, control_timestamp,
                                                 sensor_to_control_ms,
                                                 last_actuation_ts, last_control_to_act_ms,
                                                 last_sensor_to_act_ms)
                    except Exception:
                        pass
                clock.tick(int(1.0/DT))
        except KeyboardInterrupt:
            pass
        finally:
            t0 = time.time()
            try:
                print('[SHUTDOWN] Closing windows...')
                self._close_windows()
            except Exception:
                pass
            t1 = time.time()
            try:
                print('[SHUTDOWN] Stopping sensors...')
                if self.sensors:
                    self.sensors.destroy()
            except Exception:
                pass
            t2 = time.time()
            try:
                print('[SHUTDOWN] Cleaning world...')
                if self.world:
                    self.world.destroy()
            except Exception:
                pass
            t3 = time.time()
            if csv_f is not None:
                try: csv_f.close()
                except Exception: pass
            if self.telemetry is not None:
                try:
                    self.telemetry.close()
                except Exception:
                    pass
            if self.scenario_logger is not None:
                try:
                    self.scenario_logger.close()
                except Exception:
                    pass
            try:
                self._maybe_write_telephoto_compute_summary()
            except Exception:
                pass
            if self.telephoto_compute_fp is not None:
                try:
                    self.telephoto_compute_fp.close()
                except Exception:
                    pass
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
            try:
                dt_w = (t1 - t0) * 1000.0
                dt_s = (t2 - t1) * 1000.0
                dt_wld = (t3 - t2) * 1000.0
                print(f"[SHUTDOWN] Done. windows={dt_w:.0f}ms sensors={dt_s:.0f}ms world={dt_wld:.0f}ms")
            except Exception:
                pass

# ========================= entry =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Nearestfirst + TL/StopSign (YOLO + depth + stereo) — OOP wired (+TL, ABS, timers, stop-release fix, YOLO opts, CUDA stereo, per-class conf, presets)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town10HD_Opt', help='Town name, e.g., Town03 or Town05_Opt')
    parser.add_argument('--mu', type=float, default=MU_DEFAULT, help='Road friction estimate (dry~0.9, wet~0.6, ice~0.2)')
    parser.add_argument('--abs-mode', type=str, default='adaptive', choices=['off','fixed','adaptive'],
                        help='Slip controller: off (direct brake), fixed PI ABS, or μ-adaptive PI ABS')
    parser.add_argument('--apply-tire-friction', action='store_true',
                        help='Also set wheel.tire_friction≈mu to make the sim physically slick.')
    parser.add_argument('--persist-frames', type=int, default=2,
                        help='Consecutive frames required to confirm a stop‑sign before arming the stop latch')
    parser.add_argument('--min-aeb-speed', type=float, default=V_AEB_MIN_DEFAULT,
                        help='Minimum ego speed (m/s) before obstacle-triggered AEB can engage')
    parser.add_argument('--gate-confirm-frames', type=int, default=GATE_CONFIRM_FRAMES_DEFAULT,
                        help='Consecutive gate-hit frames required before obstacle braking is allowed')
    parser.add_argument('--ttc-confirm-s', type=float, default=TTC_CONFIRM_S_DEFAULT,
                        help='TTC threshold (seconds) that must be met before obstacle braking is allowed')
    parser.add_argument('--ttc-stage-strong', type=float, default=TTC_STAGE_STRONG_DEFAULT,
                        help='TTC threshold (seconds) to escalate from comfort to strong braking once confirmed')
    parser.add_argument('--ttc-stage-full', type=float, default=TTC_STAGE_FULL_DEFAULT,
                        help='TTC threshold (seconds) to escalate from strong to full AEB braking')
    parser.add_argument('--aeb-stage-comfort', type=float, default=BRAKE_STAGE_COMFORT_FACTOR,
                        help='Fraction of μg to request during the comfort braking stage (0..1)')
    parser.add_argument('--aeb-stage-strong', type=float, default=BRAKE_STAGE_STRONG_FACTOR,
                        help='Fraction of μg to request during the strong braking stage (0..1)')
    parser.add_argument('--aeb-ramp-up', type=float, default=AEB_RAMP_UP_DEFAULT,
                        help='Max increase rate for a_des (m/s^2 per second) when escalating braking')
    parser.add_argument('--aeb-ramp-down', type=float, default=AEB_RAMP_DOWN_DEFAULT,
                        help='Max decrease rate for a_des (m/s^2 per second) when relaxing braking')
    parser.add_argument('--range-est', type=str, default='pinhole',
                        choices=['pinhole', 'depth', 'stereo', 'both'],
                        help='Distance source: monocular pinhole, CARLA depth, stereo vision, or log both (depth vs pinhole)')
    parser.add_argument('--compare-csv', type=str, default=None,
                        help='If set (range-est both/stereo), write pinhole/depth/stereo comparisons to this CSV path')
    parser.add_argument('--stereo-compare-csv', type=str, default=None,
                        help='If set and --range-est=stereo, write stereo vs depth comparisons to this CSV path')
    parser.add_argument('--depth-roi-shrink', type=float, default=DEPTH_ROI_SHRINK_DEFAULT,
                        help='ROI shrink factor (0..0.9) when sampling CARLA depth inside detection boxes')
    parser.add_argument('--stereo-roi-shrink', type=float, default=STEREO_ROI_SHRINK_DEFAULT,
                        help='ROI shrink factor (0..0.9) when sampling stereo disparity depth')
    # Visualization toggles
    parser.add_argument('--no-depth-viz', action='store_true',
                        help='Hide the DEPTH/HUD_DIST OpenCV windows (alias for --no-opencv)')
    parser.add_argument('--no-opencv', action='store_true',
                        help='Disable OpenCV windows entirely (depth/HUD_DIST)')
    parser.add_argument('--no-top-cam', action='store_true',
                        help='Disable spawning the top view camera and hide it from the HUD')
    parser.add_argument('--no-depth-cam', action='store_true',
                        help='Disable spawning the depth camera (range_est depth will be auto-fallback)')
    parser.add_argument('--no-telephoto', action='store_true',
                        help='Disable the telephoto traffic-light helper camera and detector scheduling')
    parser.add_argument('--telephoto-stride', type=int, default=TELEPHOTO_STRIDE_DEFAULT,
                        help='Run telephoto YOLO inference every N frames (>=2). Default=3')
    parser.add_argument('--telephoto-zoom', type=float, default=TELEPHOTO_DIGITAL_ZOOM_DEFAULT,
                        help='Digital zoom factor (>=1.0) for the telephoto feed (crop upper-center, resize, reuse boxes).'
                             ' Default=1.5; set to 1.0 to disable')
    parser.add_argument('--telephoto-compute-log', type=str, default=None,
                        help='Optional CSV path to log total compute time with/without telephoto assists (plus cache/skip stats)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI windows (also disables OpenCV windows)')

    parser.add_argument('--video-out', type=str, default=None,
                        help='Optional path to write an MP4 video of the front RGB view (qualitative results)')

    # YOLO options
    parser.add_argument('--yolo-img', type=int, default=640,
                        help='YOLO inference size (square). Example: 480 -> 480x480')
    parser.add_argument('--yolo-device', type=str, default='cuda',
                        help="Inference device: 'auto'|'cpu'|'cuda'|'cuda:0'")
    parser.add_argument('--yolo-half', action='store_true', help='Use FP16 if CUDA is available')
    parser.add_argument('--yolo-agnostic', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--yolo-classes', type=str, default=None,
                        help='Comma-separated class names or indices (e.g., "person,car,traffic light" or "0,2,7"). None=all')
    parser.add_argument('--yolo-class-thr', type=str, default=None,
                        help='Per-class confidence thresholds: "traffic light:0.55, stop sign:0.45"')
    parser.add_argument('--yolo-class-iou', type=str, default=None,
                        help='Per-class NMS IoU thresholds: "traffic light:0.40, person:0.55"')
    parser.add_argument('--yolo-max-det', type=int, default=200, help='Max detections per image')
    parser.add_argument('--yolo-dnn', action='store_true', help='Use OpenCV DNN backend (if supported)')
    parser.add_argument('--yolo-augment', action='store_true', help='Enable TTA/augment for detection')

    # Stereo CUDA options
    parser.add_argument('--stereo-cuda', action='store_true', help='Use OpenCV CUDA StereoBM/SGM if available')
    parser.add_argument('--stereo-method', type=str, default='bm', choices=['bm','sgm'], help='Stereo method for CUDA path')

    # Per-class detection & gating overrides
    parser.add_argument('--min-h-override', type=str, default=None,
                        help='Per-class min box height (px): "person:18, traffic light:14"')
    parser.add_argument('--gate-frac-override', type=str, default=None,
                        help='Per-class center-band fraction (0..1): "car:0.35, person:0.45"')
    parser.add_argument('--gate-lateral-override', type=str, default=None,
                        help='Per-class lateral max in meters: "car:2.2, person:3.0"')

    # Engage threshold overrides
    parser.add_argument('--engage-override', type=str, default=None,
                        help='Per-class engage distances (m): "person:45, traffic light:55, car:80, stopsign:80"')
    parser.add_argument('--tl-unknown-conservative', action='store_true',
                        help='If a TL is detected but color is UNKNOWN and within engage distance, pre-brake conservatively')

    # Presets to quickly set common combos
    parser.add_argument('--preset', type=str, default=None, choices=['fast','quality','gpu480','cpu480'],
                        help='Quick config: fast|quality|gpu480|cpu480')

    # Telemetry options
    parser.add_argument('--telemetry-csv', type=str, default=None,
                        help='Write telemetry CSV with control/safety signals to this path')
    parser.add_argument('--telemetry-hz', type=float, default=10.0,
                        help='Telemetry logging frequency in Hz (default 10)')
    parser.add_argument('--log-interval-frames', type=int, default=5,
                        help='Print a concise state line to the console every N frames (0 to disable)')

    # Scenario / results logging options
    parser.add_argument('--scenario-tag', type=str, default='default',
                        help='Freeform tag to identify this run/scenario in results CSVs')
    parser.add_argument('--scenario-csv', type=str, default=None,
                        help='If set, write high-level braking episode summaries to this CSV path')

    # Ablation knob: artificial extra latency added to safety-envelope computation
    parser.add_argument('--extra-latency-ms', type=float, default=0.0,
                        help='Artificial extra latency (ms) added in safety envelope for sensitivity studies')

    # Detector selection (future extension: MobileNetSSD, etc.)
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo'],
                        help='Perception backend: currently only "yolo" is implemented; hook for future detectors')

    # NPC options
    parser.add_argument('--npc-vehicles', type=int, default=15, help='Number of NPC vehicles to spawn')
    parser.add_argument('--npc-walkers', type=int, default=5, help='Number of NPC walkers to spawn')
    parser.add_argument('--npc-seed', type=int, default=None, help='Random seed for NPC spawning')
    parser.add_argument('--npc-disable-autopilot', action='store_true', help='Spawn vehicles without autopilot')
    parser.add_argument('--npc-speed-diff-pct', type=int, default=10, help='TrafficManager global percentage speed difference (0..100)')

    return parser.parse_args()


def _apply_preset(args):
    """Mutate parsed args based on a chosen preset."""
    if not args.preset:
        return args
    p = args.preset
    if p == 'fast':
        args.yolo_img = 480
        args.yolo_device = 'cuda' if (torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        args.yolo_half = (args.yolo_device.startswith('cuda'))
        args.yolo_agnostic = True
        args.yolo_max_det = 150
        args.stereo_cuda = False
        args.range_est = 'depth'
    elif p == 'quality':
        args.yolo_img = 640
        args.yolo_device = 'cuda' if (torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        args.yolo_half = False
        args.yolo_max_det = 300
        args.yolo_augment = True
        args.stereo_cuda = True
        args.stereo_method = 'sgm'
    elif p == 'gpu480':
        args.yolo_img = 480
        args.yolo_device = 'cuda'
        args.yolo_half = True
        args.range_est = 'pinhole'
    elif p == 'cpu480':
        args.yolo_img = 480
        args.yolo_device = 'cpu'
        args.yolo_half = False
        args.range_est = 'depth'
    return args


if __name__ == '__main__':
    args = parse_args()
    args = _apply_preset(args)
    App(args).run()
