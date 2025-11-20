"""Telemetry and scenario logging helpers for braking experiments."""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Optional


class _TelemetryLogger:
    def __init__(self, csv_path: str, hz: float = 10.0):
        folder = os.path.dirname(os.path.abspath(csv_path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = csv_path
        self.hz = max(1e-6, float(hz))
        self.dt = 1.0 / self.hz
        self.last_time = None
        self._csv_f = open(csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow([
            'timestamp', 'sim_time', 'abs_mode', 'trig', 'mu', 'v_ego', 'v_ego_gt', 'v_ego_filt', 'v_ego_sync', 'a_long_sync',
            'gate_hit', 'gate_kind', 'gate_s', 'gate_l', 'gate_w', 'gate_conf', 'gate_dist_sigma', 'nn_mu',
            'stop_armed', 'stop_elapsed', 'stop_latch_elapsed', 'stop_release_ignore', 'stop_latch_hold', 'trigger_cooldown',
            'nearest_s', 'nearest_conf', 'nearest_kind', 'nearest_box', 'nearest_box_zoom', 'nearest_index', 'nearest_delta_t',
            'tracking_s', 'tracking_rate', 'tracking_age', 'tracking_id', 'tracking_count', 'tracking_ds', 'tracking_rate_ds',
            'tracking_box', 'tracking_kind', 'tracking_source', 'tracking_source2', 'tracking_lambda', 'tracking_f_global', 'lambda_max',
            'trigger_name', 'trigger_conf', 'trigger_kind', 'trigger_box', 'trigger_source', 'trigger_inference_ms', 'trigger_latency_ms',
            'trigger_tracking_latency_ms', 'trigger_sigma', 'yolo_time_ms', 'tl_state', 'tl_s', 'tl_conf', 'tl_box',
            'a_des', 'a_cap', 'a_error', 'b_cmd', 'b_out', 'thr', 'thr_out', 'I', 'I_b', 'heading_diff', 'concavity',
            'dist_clear_timer', 'stop_wait_timer', 'hold_blocked', 'hold_blocked_dist', 'brake_mode', 'rlc', 'rlc_d_min',
            'sensor_timestamp', 'control_timestamp', 'actuation_timestamp', 'sensor_to_control_ms', 'control_to_act_ms', 'sensor_to_act_ms'
        ])
        self._csv_f.flush()

    def maybe_log(self, ts: float, sim_time: float, fields: Any):
        if self.last_time is None or (sim_time - self.last_time) >= self.dt:
            self.last_time = sim_time
            self._csv.writerow([
                time.asctime(),
                sim_time,
                fields.get('abs_mode'),
                fields.get('trigger_name'),
                fields.get('mu'),
                fields.get('v_ego'),
                fields.get('v_ego_gt'),
                fields.get('v_ego_filt'),
                fields.get('v_ego_sync'),
                fields.get('a_long_sync'),
                fields.get('gate_hit'),
                fields.get('gate_kind'),
                fields.get('gate_s'),
                fields.get('gate_l'),
                fields.get('gate_w'),
                fields.get('gate_conf'),
                fields.get('gate_dist_sigma'),
                fields.get('nn_mu'),
                fields.get('stop_armed'),
                fields.get('stop_elapsed'),
                fields.get('stop_latch_elapsed'),
                fields.get('stop_release_ignore'),
                fields.get('stop_latch_hold'),
                fields.get('trigger_cooldown'),
                fields.get('nearest_s'),
                fields.get('nearest_conf'),
                fields.get('nearest_kind'),
                fields.get('nearest_box'),
                fields.get('nearest_box_zoom'),
                fields.get('nearest_index'),
                fields.get('nearest_delta_t'),
                fields.get('tracking_s'),
                fields.get('tracking_rate'),
                fields.get('tracking_age'),
                fields.get('tracking_id'),
                fields.get('tracking_count'),
                fields.get('tracking_ds'),
                fields.get('tracking_rate_ds'),
                fields.get('tracking_box'),
                fields.get('tracking_kind'),
                fields.get('tracking_source'),
                fields.get('tracking_source2'),
                fields.get('tracking_lambda'),
                fields.get('tracking_f_global'),
                fields.get('lambda_max'),
                fields.get('trigger_name'),
                fields.get('trigger_conf'),
                fields.get('trigger_kind'),
                fields.get('trigger_box'),
                fields.get('trigger_source'),
                fields.get('trigger_inference_ms'),
                fields.get('trigger_latency_ms'),
                fields.get('trigger_tracking_latency_ms'),
                fields.get('trigger_sigma'),
                fields.get('yolo_time_ms'),
                fields.get('tl_state'),
                fields.get('tl_s'),
                fields.get('tl_conf'),
                fields.get('tl_box'),
                fields.get('a_des'),
                fields.get('a_cap'),
                fields.get('a_error'),
                fields.get('b_cmd'),
                fields.get('b_out'),
                fields.get('thr'),
                fields.get('thr_out'),
                fields.get('I'),
                fields.get('I_b'),
                fields.get('heading_diff'),
                fields.get('concavity'),
                fields.get('dist_clear_timer'),
                fields.get('stop_wait_timer'),
                fields.get('hold_blocked'),
                fields.get('hold_blocked_dist'),
                fields.get('brake_mode'),
                fields.get('rlc'),
                fields.get('rlc_d_min'),
                fields.get('sensor_timestamp'),
                fields.get('control_timestamp'),
                fields.get('actuation_timestamp'),
                fields.get('sensor_to_control_ms'),
                fields.get('control_to_act_ms'),
                fields.get('sensor_to_act_ms'),
            ])

    def close(self):
        try:
            self._csv_f.close()
        except Exception:
            pass


class _ScenarioLogger:
    """High-level braking/scenario outcomes for thesis-style results."""

    def __init__(self, path: str):
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

