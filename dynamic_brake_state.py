from __future__ import print_function
import os, sys, math, random, queue, csv, multiprocessing as mp
import time, traceback
from typing import Optional, Tuple, List, Dict, Any
from collections import deque, Counter

import numpy as np
import pygame
import cv2

from calibrations import load_aeb_calibration, load_bus_calibration, load_safety_calibration
from ecu import (
    ActuationECU,
    ActuationJob,
    DistributedECUPipeline,
    MessageBus,
    PerceptionECU,
    PerceptionJob,
    PlanningECU,
    PlanningJob,
    PlanningDecision,
    SafetyManager,
    ActuationResult,
    SafetyManager,
)
from abs_system import AdaptivePISlipABSActuator, PISlipABSActuator
from tracking_system import LeadMultiObjectTracker
from telemetry import _ScenarioLogger, _TelemetryLogger
from config import *  # noqa: F401,F403
from label_utils import (
    TRIGGER_NAMES_NORM,
    _norm_label,
    parse_engage_override_map,
    parse_gate_frac_override_map,
    parse_gate_lateral_override_map,
    parse_min_h_override_map,
    parse_per_class_conf_map,
    parse_per_class_iou_map,
)
from detectors import BaseDetector, YOLODetector, CONF_THR_DEFAULT, NMS_THR
from cli_parser import parse_args, apply_preset
from camera_utils import (
    bgr_to_pygame_surface,
    carla_image_to_surface,
    decode_depth_meters_from_bgra,
    fov_y_from_x,
    intrinsics_from_fov,
    pixel_to_camera,
)
from carla_utils import import_carla
from ego_state import EgoState
from range_estimator import RangeEstimator
from sensor_rig import SensorRig
from world_manager import WorldManager

carla = import_carla()

from functools import partial


def _perception_job_handler(ecu: PerceptionECU, job: PerceptionJob) -> Any:
    return ecu.process(
        job.bgr,
        job.depth_m,
        job.depth_stereo_m,
        job.fx,
        job.fy,
        job.cx,
        job.cy,
        job.sim_time,
        job.sensor_timestamp,
        job.v,
        job.mu,
        job.log_both,
        job.csv_writer,
        tele_bgr=job.tele_bgr,
        tele_depth_m=job.tele_depth_m,
    )


def _planning_job_handler(ecu: PlanningECU, job: PlanningJob) -> PlanningDecision:
    return ecu.plan(
        job.trigger_name,
        job.nearest_s_active,
        job.nearest_thr,
        job.tl_state,
        job.tl_s_active,
        job.v,
        job.v_target,
        job.mu,
        job.ema_loop_ms,
        job.tracked_distance_for_control,
        job.stop_armed,
        job.stop_latch_time,
        job.stop_release_ignore_until,
        job.red_green_since,
        job.no_trigger_elapsed,
        job.no_red_elapsed,
        job.depth_m,
        job.depth_stereo_m,
        job.nearest_box,
        job.nearest_conf,
        job.I_err,
        job.v_prev,
    )


def _actuation_job_handler(ecu: ActuationECU, job: ActuationJob) -> ActuationResult:
    return ecu.apply_abs(job.brake_cmd, job.v_ego, job.wheel_speeds, job.a_long)


# ===================== helpers =====================
def wrap_pi(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def yaw_to_compass(yaw_deg: float) -> str:
    y = (yaw_deg + 360.0) % 360.0
    dirs = ['E','SE','S','SW','W','NW','N','NE','E']
    idx = int((y + 22.5) // 45)
    return dirs[idx]

def shadow(surface: pygame.Surface, text: str, pos, color, shadow_color=(0,0,0), offset=1):
    font = pygame.font.SysFont('Arial', 20)
    s = font.render(text, True, shadow_color); surface.blit(s, (pos[0]+offset, pos[1]+offset))
    s2 = font.render(text, True, color);        surface.blit(s2, pos)

# ===================== Depth & TL helpers =====================
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
        # ECU adapters make the perception/control/actuation responsibilities explicit.
        self.multiprocess_ecus = bool(getattr(self.args, 'multiprocess_ecus', False))
        self.ecu_process_timeout = max(0.05, float(getattr(self.args, 'ecu_process_timeout', 0.35)))
        self.bus_manager = mp.Manager() if self.multiprocess_ecus else None
        self.bus = MessageBus(self.bus_manager)
        self.bus_latency_perception = max(0.0, float(getattr(self.args, 'bus_latency_perception', 0.0)))
        self.bus_latency_planning = max(0.0, float(getattr(self.args, 'bus_latency_planning', 0.0)))
        bus_defaults = {
            'perception': {
                'drop_rate': max(0.0, min(1.0, float(getattr(self.args, 'bus_drop_perception', 0.0)))),
                'jitter_s': max(0.0, float(getattr(self.args, 'bus_jitter_perception', 0.0))),
                'max_age_s': 0.35,
                'max_depth': 8,
                'deadline_s': max(0.05, float(getattr(self.args, 'bus_deadline_perception', 0.15))),
                'priority': 1,
            },
            'planning': {
                'drop_rate': max(0.0, min(1.0, float(getattr(self.args, 'bus_drop_planning', 0.0)))),
                'jitter_s': max(0.0, float(getattr(self.args, 'bus_jitter_planning', 0.0))),
                'max_age_s': 0.35,
                'max_depth': 8,
                'deadline_s': max(0.05, float(getattr(self.args, 'bus_deadline_planning', 0.15))),
                'priority': 0,
            },
        }
        self.bus_calibration = load_bus_calibration(getattr(self.args, 'bus_calibration_file', None), bus_defaults)
        for topic, cfg in self.bus_calibration.items():
            self.bus.configure_topic(
                topic,
                drop_rate=cfg.drop_rate,
                jitter_s=cfg.jitter_s,
                max_age_s=cfg.max_age_s,
                max_depth=cfg.max_depth,
                deadline_s=cfg.deadline_s,
                priority=cfg.priority,
            )
        self.perception_ecu = PerceptionECU(self._perception_step)
        self.planning_ecu = PlanningECU(self._control_step)
        self.actuation_ecu = ActuationECU(
            abs_fn=(None if self.abs_actuator is None else self.abs_actuator.step),
            abs_debug_fn=(None if self.abs_actuator is None else self.abs_actuator.debug_metrics),
        )
        self.ecu_pipeline = None
        if self.multiprocess_ecus:
            self.ecu_pipeline = DistributedECUPipeline(
                partial(_perception_job_handler, self.perception_ecu),
                partial(_planning_job_handler, self.planning_ecu),
                partial(_actuation_job_handler, self.actuation_ecu),
            )
        safety_defaults = {
            'perception_freshness_s': float(getattr(self.args, 'perception_freshness_s', 0.35)),
            'planning_freshness_s': float(getattr(self.args, 'planning_freshness_s', 0.35)),
            'actuation_freshness_s': float(getattr(self.args, 'actuation_freshness_s', 0.35)),
            'ttc_floor_s': float(getattr(self.args, 'safety_ttc_floor_s', 0.25)),
            'v_min_plausible': float(getattr(self.args, 'safety_v_min_plausible', 0.5)),
            'wheel_slip_max': float(getattr(self.args, 'safety_wheel_slip_max', 0.45)),
            'brake_fail_safe': float(getattr(self.args, 'brake_fail_safe', 1.0)),
        }
        self.safety_calibration = load_safety_calibration(getattr(self.args, 'safety_calibration_file', None), safety_defaults)
        self.safety_manager = SafetyManager(
            brake_fail_safe=self.safety_calibration.brake_fail_safe,
            perception_freshness_s=self.safety_calibration.perception_freshness_s,
            planning_freshness_s=self.safety_calibration.planning_freshness_s,
            actuation_freshness_s=self.safety_calibration.actuation_freshness_s,
            ttc_floor_s=self.safety_calibration.ttc_floor_s,
            v_min_plausible=self.safety_calibration.v_min_plausible,
            wheel_slip_max=self.safety_calibration.wheel_slip_max,
        )
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

        # Decision-layer tuning knobs, validated via calibration payloads
        cal_defaults = {
            'min_aeb_speed': getattr(self.args, 'min_aeb_speed', V_AEB_MIN_DEFAULT),
            'gate_confirm_frames': getattr(self.args, 'gate_confirm_frames', GATE_CONFIRM_FRAMES_DEFAULT),
            'ttc_confirm_s': getattr(self.args, 'ttc_confirm_s', TTC_CONFIRM_S_DEFAULT),
            'ttc_stage_strong': getattr(self.args, 'ttc_stage_strong', TTC_STAGE_STRONG_DEFAULT),
            'ttc_stage_full': getattr(self.args, 'ttc_stage_full', TTC_STAGE_FULL_DEFAULT),
            'stage_factor_comfort': getattr(self.args, 'aeb_stage_comfort', BRAKE_STAGE_COMFORT_FACTOR),
            'stage_factor_strong': getattr(self.args, 'aeb_stage_strong', BRAKE_STAGE_STRONG_FACTOR),
            'aeb_ramp_up': getattr(self.args, 'aeb_ramp_up', AEB_RAMP_UP_DEFAULT),
            'aeb_ramp_down': getattr(self.args, 'aeb_ramp_down', AEB_RAMP_DOWN_DEFAULT),
        }
        self.calibration = load_aeb_calibration(getattr(self.args, 'calibration_file', None), cal_defaults)
        self.min_aeb_speed = self.calibration.min_aeb_speed
        self.gate_confirm_frames = self.calibration.gate_confirm_frames
        self.ttc_confirm_s = self.calibration.ttc_confirm_s
        self.ttc_stage_strong = self.calibration.ttc_stage_strong
        self.ttc_stage_full = self.calibration.ttc_stage_full
        self.stage_factor_comfort = self.calibration.stage_factor_comfort
        self.stage_factor_strong = self.calibration.stage_factor_strong
        self.stage_factor_full = BRAKE_STAGE_FULL_FACTOR
        self.aeb_ramp_up = self.calibration.aeb_ramp_up
        self.aeb_ramp_down = self.calibration.aeb_ramp_down

        self._gate_confirm_counter = 0
        self._hazard_confirm_since = -1.0
        self._aeb_a_des = 0.0

    def _draw_hud(self, screen, bgr, perf_fps, perf_ms, x, y, z, yaw, compass,
                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target,
                   collision_flag: bool,
                   det_points: Optional[List[Dict[str, Any]]] = None,
                   range_mode_label: Optional[str] = None,
                   abs_lambda: Optional[float] = None,
                   abs_factor: Optional[float] = None,
                   abs_mu: Optional[float] = None,
                   abs_regime: Optional[str] = None,
                   tele_bgr: Optional[np.ndarray] = None):
        if self.headless:
            return
        screen.fill((0, 0, 0))
        surf_front = bgr_to_pygame_surface(bgr)
        screen.blit(surf_front, (0, 0))
        if tele_bgr is not None and self.telephoto_enabled:
            if tele_bgr.shape[0] != IMG_H or tele_bgr.shape[1] != IMG_W:
                tele_bgr = cv2.resize(tele_bgr, (IMG_W, IMG_H))
            surf_tele = bgr_to_pygame_surface(tele_bgr)
            screen.blit(surf_tele, (IMG_W, 0))
        v_kmh = v * 3.6
        txt0 = f'ego @ x={x:8.2f}  y={y:8.2f}  z={z:6.2f}  | yaw={yaw:+6.1f}° {compass}'
        txt1 = f'Frame {frame_id} | v={v_kmh:5.1f} km/h | trigger={trigger_name or "None"} | TL={tl_state}'
        txt2 = f'thr={throttle:.2f}  brk={brake:.2f}  hold={hold_blocked}({hold_reason})  clear={no_trigger_elapsed:.1f}s  red_clear={no_red_elapsed:.1f}s  stopArmed={stop_armed}  ignoreT={(max(0.0, stop_release_ignore_until-sim_time)):.1f}s'
        txt_perf = f'FPS={perf_fps:.1f}  time={perf_ms:.1f}ms'
        shadow(screen, txt_perf, (10, IMG_H-156), (255,200,0))
        shadow(screen, txt0, (10, IMG_H-134), (200,200,255))
        shadow(screen, txt1, (10, IMG_H-90), (255,255,255))
        shadow(screen, txt2, (10, IMG_H-68), (0,255,160))
        if any(val is not None for val in (dbg_tau_dyn, dbg_D_safety_dyn, dbg_sigma_depth,
                                           dbg_gate_hit, dbg_a_des, dbg_brake, v_target)):
            tau_txt = 'n/a' if dbg_tau_dyn is None else f'{dbg_tau_dyn:0.2f}'
            dsafe_txt = 'n/a' if dbg_D_safety_dyn is None else f'{dbg_D_safety_dyn:0.1f}'
            sigma_txt = 'n/a' if dbg_sigma_depth is None else f'{dbg_sigma_depth:0.2f}'
            gate_txt = 'n/a' if dbg_gate_hit is None else f'{int(bool(dbg_gate_hit))}'
            a_des_txt = 'n/a' if dbg_a_des is None else f'{dbg_a_des:0.2f}'
            brk_txt = 'n/a' if dbg_brake is None else f'{dbg_brake:0.2f}'
            vtgt_txt = 'n/a' if v_target is None else f'{v_target*3.6:0.0f}'
            shadow(screen, f'tau={tau_txt}  Dsafe={dsafe_txt} m  sigma={sigma_txt} m  gate={gate_txt}  a_des={a_des_txt}  brk={brk_txt}  Vtgt={vtgt_txt}km/h',
                   (10, IMG_H-24), (255,255,0))
        if abs_lambda is not None or abs_factor is not None or abs_mu is not None:
            slip_txt = 'n/a' if abs_lambda is None else f'{abs_lambda:.2f}'
            fac_txt = '1.00' if abs_factor is None else f'{abs_factor:.2f}'
            mu_txt = 'n/a' if abs_mu is None else f'{abs_mu:.2f}'
            regime_txt = abs_regime or 'n/a'
            shadow(screen,
                   f'ABS slip={slip_txt}  f={fac_txt}  mu_est={mu_txt}  regime={regime_txt}',
                   (10, IMG_H-46), (180,255,255))
        log_y = 10
        log_x = 10
        mode_label = range_mode_label or 'range'
        shadow(screen, f'Ego→Object distances ({mode_label}, meters):', (log_x, log_y), (0, 255, 255))
        log_y += 22
        det_points_sorted: List[Dict[str, Any]] = []
        if det_points:
            det_points_sorted = sorted(det_points, key=lambda d: d.get('z', float('inf')))
        if det_points_sorted:
            for d in det_points_sorted[:22]:
                xcam, ycam, zcam = d['xyz']
                shadow(screen,
                       f"{d['name']:<14}  Z={zcam:5.1f} m   X={xcam:+4.1f} m   Y={ycam:+4.1f} m",
                       (log_x, log_y), (180, 255, 180))
                log_y += 18
        else:
            shadow(screen, 'No detections', (log_x, log_y), (180, 180, 180))
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
            win_cols = 2 if self.telephoto_enabled else 1
            WIN_W, WIN_H = IMG_W * win_cols, IMG_H
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption('Nearestfirst + TL/StopSign | YOLO12n | Sync (OOP)')

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
        shutdown_reason = 'running'
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
                f"Views: telephoto={'ON' if self.telephoto_enabled else 'OFF'} "
                f"depth={'ON' if not getattr(self.args, 'no_depth_cam', False) else 'OFF'}"
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
                try:
                    frame_id = self.world.tick()
                except Exception as e:
                    shutdown_reason = f'world.tick failed: {e}'
                    raise
                sim_time += DT
                self._sim_time = sim_time
                tic = time.time()

                try:
                    frames = self.sensors.read(expected_frame=frame_id)
                except queue.Empty:
                    shutdown_reason = 'sensor queue timeout (no new frames)'
                    break
                sensor_timestamp = None
                front_frame = frames.get('front')
                if front_frame is not None:
                    sensor_timestamp = getattr(front_frame, 'timestamp', None)
                io = self._read_frames(frames)
                bgr = io['bgr']
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

                # Perception step (YOLO + depth/stereo + gating)
                detect_ms = None
                detect_t0 = time.time()
                self._frame_index += 1
                if self.ecu_pipeline is not None:
                    perc_job = PerceptionJob(
                        bgr=bgr,
                        depth_m=depth_m,
                        depth_stereo_m=depth_stereo_m,
                        fx=FX_,
                        fy=FY_,
                        cx=CX_,
                        cy=CY_,
                        sim_time=sim_time,
                        sensor_timestamp=sensor_timestamp,
                        v=v,
                        mu=MU,
                        log_both=log_both,
                        csv_writer=csv_w,
                        tele_bgr=tele_bgr,
                        tele_depth_m=tele_depth_m,
                    )
                    try:
                        perc = self.ecu_pipeline.run_perception(perc_job, timeout=self.ecu_process_timeout)
                    except Exception as exc:
                        perc = PerceptionSignal(
                            bgr=bgr,
                            det_points=[],
                            nearest_s_active=None,
                            nearest_kind=None,
                            nearest_thr=None,
                            nearest_box=None,
                            nearest_conf=None,
                            tl_state='UNKNOWN',
                            tl_s_active=None,
                            tl_det_box=None,
                            stop_detected_current=False,
                            fault_code=f'PERCEPTION_PROC_FAIL:{exc}',
                            valid=False,
                        )
                        perc.timestamp = time.time()
                else:
                    perc = self.perception_ecu.process(
                        bgr,
                        depth_m,
                        depth_stereo_m,
                        FX_,
                        FY_,
                        CX_,
                        CY_,
                        sim_time,
                        sensor_timestamp,
                        v,
                        MU,
                        log_both,
                        csv_w,
                        tele_bgr=tele_bgr,
                        tele_depth_m=tele_depth_m,
                    )
                detect_ms = (time.time() - detect_t0) * 1000.0
                perc.frame_id = self._frame_index
                perc.validate(freshness_s=self.safety_calibration.perception_freshness_s)
                self.bus.send('perception', perc, now=sim_time, latency_s=self.bus_latency_perception)
                perc_bus = self.bus.receive_latest('perception', now=sim_time, max_age_s=0.3)
                perc = perc_bus or perc
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
                tracked_distance_for_control = None
                tracked_rate = None
                tracker = getattr(self, '_lead_tracker', None)
                if tracker is not None:
                    obstacle_measurements = perc.get('obstacle_measurements', [])
                    tracker_state = tracker.step(sim_time, obstacle_measurements)
                    if tracker_state is not None:
                        tracked_distance_for_control = tracker_state.get('distance')
                        tracked_rate = tracker_state.get('rate')
                lead_track_id = tracker_state.get('id') if tracker_state is not None else None
                lead_track_count = tracker_state.get('tracker_count', 0) if tracker_state is not None else 0
                if tracked_distance_for_control is None:
                    if nearest_s_active is not None:
                        tracked_distance_for_control = nearest_s_active
                    else:
                        tracked_distance_for_control = last_s0
                self._tracked_distance = tracked_distance_for_control
                self._tracked_rate = tracked_rate

                trigger_name = nearest_kind

                # Control step via helper (conceptually ADAS domain ECU → Brake ECU request)
                if self.ecu_pipeline is not None:
                    plan_job = PlanningJob(
                        trigger_name=trigger_name,
                        nearest_s_active=nearest_s_active,
                        nearest_thr=nearest_thr,
                        tl_state=tl_state,
                        tl_s_active=tl_s_active,
                        v=v,
                        v_target=v_target,
                        mu=MU,
                        ema_loop_ms=ema_loop_ms,
                        tracked_distance_for_control=tracked_distance_for_control,
                        stop_armed=stop_armed,
                        stop_latch_time=stop_latch_time,
                        stop_release_ignore_until=stop_release_ignore_until,
                        red_green_since=red_green_since,
                        no_trigger_elapsed=no_trigger_elapsed,
                        no_red_elapsed=no_red_elapsed,
                        depth_m=depth_m,
                        depth_stereo_m=depth_stereo_m,
                        nearest_box=nearest_box,
                        nearest_conf=nearest_conf,
                        I_err=I_err,
                        v_prev=v_prev,
                    )
                    try:
                        planning_decision = self.ecu_pipeline.run_planning(plan_job, timeout=self.ecu_process_timeout)
                    except Exception as exc:
                        planning_decision = PlanningDecision(
                            throttle=0.0,
                            brake=1.0,
                            ctrl=None,
                            hold_blocked=True,
                            hold_reason=f'PLANNING_PROC_FAIL:{exc}',
                            stop_armed=stop_armed,
                            stop_latch_time=stop_latch_time,
                            stop_release_ignore_until=stop_release_ignore_until,
                            debug={},
                            integral_error=I_err,
                            aeb_request=None,
                            valid=False,
                            fault_code=f'PLANNING_PROC_FAIL:{exc}',
                        )
                        planning_decision.timestamp = time.time()
                else:
                    planning_decision = self.planning_ecu.plan(
                        trigger_name,
                        nearest_s_active,
                        nearest_thr,
                        tl_state,
                        tl_s_active,
                        v,
                        v_target,
                        MU,
                        ema_loop_ms,
                        tracked_distance_for_control,
                        stop_armed,
                        stop_latch_time,
                        stop_release_ignore_until,
                        red_green_since,
                        no_trigger_elapsed,
                        no_red_elapsed,
                        depth_m,
                        depth_stereo_m,
                        nearest_box,
                        nearest_conf,
                        I_err,
                        v_prev,
                    )
                planning_decision.validate(freshness_s=self.safety_calibration.planning_freshness_s)
                self.bus.send('planning', planning_decision, now=sim_time, latency_s=self.bus_latency_planning)
                planning_from_bus = self.bus.receive_latest('planning', now=sim_time, max_age_s=0.3)
                if planning_from_bus is not None:
                    planning_decision = planning_from_bus
                if not planning_decision.valid:
                    planning_decision.brake = max(planning_decision.brake, 1.0)
                    planning_decision.throttle = 0.0
                throttle = planning_decision.throttle
                brake = planning_decision.brake
                ctrl = planning_decision.ctrl
                hold_blocked = planning_decision.hold_blocked
                hold_reason = planning_decision.hold_reason
                stop_armed = planning_decision.stop_armed
                stop_latch_time = planning_decision.stop_latch_time
                stop_release_ignore_until = planning_decision.stop_release_ignore_until
                dbg_map = planning_decision.debug
                I_err = planning_decision.integral_error
                perception_latency_ms = (time.time() - perc.timestamp) * 1000.0 if hasattr(perc, 'timestamp') else None
                planning_latency_ms = (time.time() - planning_decision.timestamp) * 1000.0
                dbg_map = dict(dbg_map or {})
                dbg_map['perception_bus_latency_ms'] = perception_latency_ms
                dbg_map['planning_bus_latency_ms'] = planning_latency_ms
                dbg_map['planning_valid'] = planning_decision.valid
                dbg_map['bus_metrics'] = {
                    'perception': dict(self.bus.metrics.get('perception', {})),
                    'planning': dict(self.bus.metrics.get('planning', {})),
                }
                dbg_map['ecu_process_mode'] = 'distributed' if self.ecu_pipeline is not None else 'in_process'
                dbg_map['calibration_meta'] = {
                    'aeb': getattr(self.calibration, 'metadata', {}),
                    'safety': getattr(self.safety_calibration, 'metadata', {}),
                    'bus': {k: vars(v) for k, v in self.bus_calibration.items()},
                }
                if planning_decision.aeb_request is not None:
                    dbg_map['aeb_request'] = {
                        'mode': planning_decision.aeb_request.mode,
                        'target_decel': planning_decision.aeb_request.target_decel,
                        'priority': planning_decision.aeb_request.priority,
                    }
                abs_lambda = None
                abs_factor = None
                abs_mu = None
                abs_regime = 'off' if self.abs_actuator is None else None
                abs_dbg = None
                if self.ecu_pipeline is not None:
                    abs_job = ActuationJob(brake_cmd=brake, v_ego=v, wheel_speeds=wheel_speeds, a_long=a_long)
                    try:
                        abs_result = self.ecu_pipeline.run_actuation(abs_job, timeout=self.ecu_process_timeout)
                    except Exception as exc:
                        abs_result = ActuationResult(brake=brake, abs_dbg=None, fault_code=f'ACT_PROC_FAIL:{exc}', valid=False)
                        abs_result.timestamp = time.time()
                else:
                    abs_result = self.actuation_ecu.apply_abs(brake, v, wheel_speeds, a_long)
                abs_result.validate(freshness_s=self.safety_calibration.actuation_freshness_s)
                brake = abs_result.get('brake', brake)
                abs_dbg = abs_result.get('abs_dbg')
                planning_decision.brake = brake
                if not abs_result.valid:
                    brake = max(brake, 1.0)
                    dbg_map['abs_fault'] = abs_result.fault_code or 'UNKNOWN_ABS_FAULT'
                safety = self.safety_manager.evaluate(perc, planning_decision, abs_result, v_ego=v, ttc=dbg_map.get('ttc'))
                if safety.mode != 'NOMINAL':
                    throttle = safety.throttle
                    brake = max(brake, safety.brake)
                dbg_map['safety_mode'] = safety.mode
                if safety.faults:
                    dbg_map['safety_faults'] = list(safety.faults)
                if safety.latched:
                    dbg_map['safety_faults_latched'] = list(safety.latched)
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
                    mode_label = 'depth camera'
                    if self.args.range_est == 'pinhole': mode_label = 'pinhole'
                    elif self.args.range_est == 'stereo': mode_label = 'stereo camera'
                    elif self.args.range_est == 'both': mode_label = 'depth + pinhole log'
                    self._draw_hud(self.screen, bgr, perf_fps, perf_ms, x, y, z, yaw, compass,
                                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target,
                                   collision_flag, det_points, mode_label,
                                   dbg_abs_lambda, dbg_abs_factor, dbg_abs_mu, dbg_abs_regime,
                                   tele_bgr)

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

                if not self.headless:
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            shutdown_reason = 'pygame QUIT event'
                            raise KeyboardInterrupt
                        elif e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_ESCAPE:
                                shutdown_reason = 'ESC pressed'
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
            if shutdown_reason == 'running':
                shutdown_reason = 'KeyboardInterrupt'
        except Exception as e:
            shutdown_reason = f'unhandled exception: {e.__class__.__name__}: {e}'
            traceback.print_exc()
        finally:
            t0 = time.time()
            try:
                print(f'[SHUTDOWN] Reason: {shutdown_reason}')
            except Exception:
                pass
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

if __name__ == '__main__':
    args = parse_args()
    args = apply_preset(args)
    App(args).run()
