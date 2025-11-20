"""Lead tracking utilities extracted from dynamic_brake_state."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def iou_xywh(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    """IoU between (x,y,w,h) boxes."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    denom = float(area_a + area_b - inter + 1e-9)
    return inter / denom if denom > 0 else 0.0


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

    def _new_tracker(self) -> LeadKalmanTracker:
        return LeadKalmanTracker(dt=self.dt, **self._tracker_kwargs)

    def _assoc_score(self, track_box: Optional[Tuple[int, int, int, int]], meas_box: Optional[Tuple[int, int, int, int]]) -> float:
        return iou_xywh(track_box, meas_box)

    def _assign_measurement(self, measurement: Dict[str, Any]):
        if not self._tracks:
            return None
        best_score = self.assoc_iou_min
        best_track = None
        for tid, tracker in self._tracks.items():
            meta = self._track_meta.get(tid, {})
            track_box = meta.get('box') if meta else getattr(tracker, 'box', None)
            score = self._assoc_score(track_box, measurement.get('box'))
            if score > best_score:
                best_score = score
                best_track = tid
        return best_track

    def step(self, sim_time: float, measurement: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if measurement is not None:
            tid = self._assign_measurement(measurement)
            if tid is None and len(self._tracks) < self.max_tracks:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = self._new_tracker()
            if tid is not None:
                tracker = self._tracks.get(tid)
                if tracker is not None:
                    meta = self._track_meta.setdefault(tid, {})
                    meta['box'] = measurement.get('box')
                    meta['kind'] = measurement.get('kind')
                    meta['last_update'] = sim_time
                    tracker.box = measurement.get('box')
                    tracker.kind = measurement.get('kind')
                    tracker.step(sim_time, measurement)
        to_drop = []
        for tid, tracker in self._tracks.items():
            meta = self._track_meta.get(tid, {})
            meas_age = None if meta.get('last_update') is None else sim_time - meta.get('last_update', sim_time)
            state = tracker.step(sim_time, None)
            if state is None or (meas_age is not None and meas_age > tracker.max_miss_s):
                to_drop.append(tid)
        for tid in to_drop:
            self._tracks.pop(tid, None)
            self._track_meta.pop(tid, None)
        if not self._tracks:
            return None
        lead_tid = min(self._tracks.keys())
        lead_state = self._tracks[lead_tid].step(sim_time, None)
        if lead_state is None:
            return None
        meta = self._track_meta.get(lead_tid, {})
        return {
            'id': lead_tid,
            'distance': lead_state.get('distance'),
            'rate': lead_state.get('rate'),
            'age': lead_state.get('age'),
            'box': meta.get('box'),
            'kind': meta.get('kind'),
            'tracker_count': len(self._tracks),
        }

