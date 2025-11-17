from __future__ import print_function
import os, sys, math, argparse, random, queue, glob
import time
from typing import Optional, Tuple, List, Dict, Any

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

# Simple ROI TL classifier (fallback if CARLA API not reliable)
def estimate_tl_color_from_roi(bgr: np.ndarray, box: Tuple[int,int,int,int]) -> str:
    x,y,w,h = box
    if w <= 0 or h <= 0:
        return 'UNKNOWN'
    roi = bgr[max(0,y):min(IMG_H,y+h), max(0,x):min(IMG_W,x+w)]
    if roi.size == 0:
        return 'UNKNOWN'
    thirds = [(0, int(h/3)), (int(h/3), int(2*h/3)), (int(2*h/3), h)]
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
                 enable_top: bool = True, enable_depth: bool = True):
        self.world = world
        self.vehicle = vehicle
        self.range_est = range_est
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

        self.actors = [a for a in [self.cam_front, self.cam_top, self.cam_depth, self.cam_stereo_left, self.cam_stereo_right] if a is not None]

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
        else:
            out['front'] = self._get_for_frame(self.q_front, expected_frame, timeout)
            if self.q_top is not None:
                out['top'] = self._get_for_frame(self.q_top, expected_frame, timeout)
            if self.q_depth is not None:
                out['depth'] = self._get_for_frame(self.q_depth, expected_frame, timeout)
            if self.q_stereo_left is not None and self.q_stereo_right is not None:
                out['stereo_left']  = self._get_for_frame(self.q_stereo_left, expected_frame, timeout)
                out['stereo_right'] = self._get_for_frame(self.q_stereo_right, expected_frame, timeout)
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
                try:
                    self.collision_happened = True
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
        self._w.writerow(['t','v_mps','tau_dyn','D_safety_dyn','sigma_depth','a_des','brake'])

    def maybe_log(self, t: float, v: float, tau_dyn: Optional[float], D_safety_dyn: Optional[float],
                  sigma_depth: Optional[float], a_des: Optional[float], brake: Optional[float]):
        if t - self._last_t >= self.period:
            self._last_t = t
            self._w.writerow([
                float(t), float(v),
                (None if tau_dyn is None else float(tau_dyn)),
                (None if D_safety_dyn is None else float(D_safety_dyn)),
                (None if sigma_depth is None else float(sigma_depth)),
                (None if a_des is None else float(a_des)),
                (None if brake is None else float(brake)),
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
        self._f = open(path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow([
            'scenario', 'trigger_kind', 'mu',
            'v_init_mps', 's_init_m', 's_min_m',
            'stopped', 't_to_stop_s', 'collision'
        ])

    def log(self, scenario: str, trigger_kind: str, mu: float,
            v_init: float, s_init: float, s_min: float,
            stopped: bool, t_to_stop: float, collision: bool):
        self._w.writerow([
            scenario,
            trigger_kind,
            float(mu),
            float(v_init),
            float(s_init),
            float(s_min),
            bool(stopped),
            float(t_to_stop),
            bool(collision),
        ])

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
        self.class_conf_map = parse_per_class_conf_map(self.args.yolo_class_thr)
        self.engage_override_map = parse_engage_override_map(self.args.engage_override)
        self.min_h_map = parse_min_h_override_map(self.args.min_h_override)
        self.gate_frac_override = parse_gate_frac_override_map(self.args.gate_frac_override)
        self.gate_lat_override = parse_gate_lateral_override_map(self.args.gate_lateral_override)
        self.class_conf_map = parse_per_class_conf_map(self.args.yolo_class_thr)
        self.range = RangeEstimator(use_cuda=self.args.stereo_cuda, method=self.args.stereo_method)
        self.world: Optional[WorldManager] = None
        self.sensors: Optional[SensorRig] = None
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

    def _draw_hud(self, screen, bgr, img_top, perf_fps, perf_ms, x, y, z, yaw, compass,
                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target):
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
        pygame.display.flip()

    # --- IO: read sensors and decode ---
    def _read_frames(self, frames: Dict[str, Any]) -> Dict[str, Any]:
        img_front = frames['front']
        img_top   = frames.get('top', None)
        img_depth = frames.get('depth', None)

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

        return {
            'bgr': bgr,
            'img_top': img_top,
            'depth_m': depth_m,
            'depth_stereo_m': depth_stereo_m,
        }

    # --- Perception: detection, distances, gating, TL/stop extraction ---
    def _perception_step(self, bgr: np.ndarray, depth_m: np.ndarray, depth_stereo_m: Optional[np.ndarray],
                         FX_: float, FY_: float, CX_: float, CY_: float,
                         sim_time: float, v: float, MU: float,
                         log_both: bool, csv_w) -> Dict[str, Any]:
        labels = self.detector.labels or {}
        classIds, confs, boxes = self.detector.predict_raw(bgr)

        nearest_s_active = None
        nearest_kind = None
        nearest_box = None
        nearest_thr = None
        nearest_conf = None
        stop_detected_current = False

        tl_det_s, tl_det_box, tl_det_state = None, None, 'UNKNOWN'

        det_points: List[Dict[str, Any]] = []
        depth_cache_depth: Dict[Tuple[int,int,int,int], Optional[float]] = {}
        depth_cache_stereo: Dict[Tuple[int,int,int,int], Optional[float]] = {}

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

                if self.args.range_est in ('depth','both'):
                    if (x1, y1, w, h) not in depth_cache_depth:
                        depth_cache_depth[(x1, y1, w, h)] = median_depth_in_box(depth_m, (x1, y1, w, h), shrink=0.4)
                    s_depth = depth_cache_depth[(x1, y1, w, h)]
                else:
                    s_depth = None
                if (self.args.range_est == 'stereo') and (depth_stereo_m is not None):
                    if (x1, y1, w, h) not in depth_cache_stereo:
                        depth_cache_stereo[(x1, y1, w, h)] = median_depth_in_box(depth_stereo_m, (x1, y1, w, h), shrink=0.4)
                    s_stereo = depth_cache_stereo[(x1, y1, w, h)]
                else:
                    s_stereo = None

                if self.args.range_est == 'pinhole': s_use = s_pinhole
                elif self.args.range_est == 'depth': s_use = s_depth
                elif self.args.range_est == 'stereo': s_use = s_stereo
                else: s_use = s_depth if (s_depth is not None) else s_pinhole
                if s_use is None:
                    continue

                frac_cls = float(self.gate_frac_override.get(norm, CENTER_BAND_FRAC))
                band_px_cls = frac_cls * IMG_W
                lateral_max_m = float(self.gate_lat_override.get(norm, LATERAL_MAX))
                do_gating = (norm in VEHICLE_CLASSES) or (norm in self.gate_frac_override) or (norm in self.gate_lat_override)

                lateral_ok = True
                if do_gating:
                    if abs(xc - cx0) > band_px_cls:
                        lateral_ok = False
                    else:
                        lateral_range = s_pinhole if s_pinhole is not None else (s_depth if s_depth is not None else (s_stereo if s_stereo is not None else 0.0))
                        lateral = ((xc - cx0) / max(1e-6, FX_)) * max(1.0, lateral_range)
                        if abs(lateral) > lateral_max_m:
                            lateral_ok = False
                if not lateral_ok:
                    continue

                if is_tl_like:
                    tl_state_roi = estimate_tl_color_from_roi(bgr, (x1, y1, w, h))
                    if (tl_det_s is None) or (s_use < tl_det_s):
                        tl_det_s, tl_det_box, tl_det_state = s_use, (x1, y1, w, h), tl_state_roi

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
                    label = f'{name} {tl_det_state}'
                else:
                    label = f'{name}'
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                ann_parts = []
                if s_pinhole is not None: ann_parts.append(f"P:{s_pinhole:.1f}m")
                if s_depth   is not None: ann_parts.append(f"D:{s_depth:.1f}m")
                if s_stereo  is not None: ann_parts.append(f"S:{s_stereo:.1f}m")
                cv2.putText(bgr, f'{label} '+ ' '.join(ann_parts), (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if kind is not None and thr_for_kind is not None and nearest_thr is None:
                    nearest_thr = thr_for_kind

                # HUD points (for distance overlay window)
                u, v_ = x1 + w/2.0, y1 + h/2.0
                if self.args.range_est == 'stereo' and depth_stereo_m is not None:
                    z = depth_cache_stereo.get((x1, y1, w, h))
                else:
                    z = depth_cache_depth.get((x1, y1, w, h))
                if z is None:
                    z = median_depth_in_box(depth_m if depth_stereo_m is None else (depth_stereo_m if self.args.range_est=='stereo' else depth_m),
                                            (x1, y1, w, h), shrink=0.4)
                if z is not None:
                    xyz = pixel_to_camera(u, v_, z, FX_, FY_, CX_, CY_)
                    if xyz is not None:
                        det_points.append({'name': name, 'box': (x1, y1, w, h), 'xyz': xyz, 'z': z})

                if log_both and csv_w is not None:
                    if (s_pinhole is not None) and (s_depth is not None):
                        csv_w.writerow([sim_time, name, x1, y1, w, h, s_pinhole, s_depth, abs(s_pinhole - s_depth), v, MU, None])

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
            'stop_detected_current': stop_detected_current,
        }

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
            if (self.args.range_est in ('depth','both')) and (nearest_box is not None):
                sd = depth_sigma_in_box(depth_m, nearest_box, shrink=0.3)
            elif (self.args.range_est == 'stereo') and (nearest_box is not None) and (depth_stereo_m is not None):
                sd = depth_sigma_in_box(depth_stereo_m, nearest_box, shrink=0.3)
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
        return tau_dyn, D_safety_dyn, sigma_depth

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
        in_brake_band = False

        if (nearest_s_active is not None) and (nearest_thr is not None) and (nearest_s_active <= nearest_thr):
            in_brake_band = True
        if (not in_brake_band) and stop_armed and (not hold_blocked):
            in_brake_band = True
            if nearest_s_active is None and (last_s0 is not None):
                nearest_s_active = last_s0
            if trigger_name is None: trigger_name = 'stop sign'
            nearest_thr = S_ENGAGE
        if (not in_brake_band) and (tl_state == 'RED') and (tl_s_active is not None) and (tl_s_active <= S_ENGAGE_TL):
            in_brake_band = True
            nearest_s_active = tl_s_active
            nearest_thr = S_ENGAGE_TL
            trigger_name = 'traffic light (RED)'

        if in_brake_band and trigger_name and ('stop sign' in trigger_name) and (stop_release_ignore_until >= 0) and (self._sim_time < stop_release_ignore_until):
            in_brake_band = False

        dbg = {'tau_dyn': None, 'D_safety_dyn': None, 'sigma_depth': None, 'gate_hit': False, 'a_des': None, 'brake': None}

        if in_brake_band and (nearest_s_active is not None):
            s_used = 0.7 * (last_s0 if last_s0 is not None else nearest_s_active) + 0.3 * nearest_s_active
            tau_dyn, D_safety_dyn, sigma_depth = self._safety_envelope(v, MU, ema_loop_ms, nearest_box, nearest_conf, depth_m, depth_stereo_m)
            required_dist_physics = (v*v)/(2.0*max(1e-3, A_MU)) + v*tau_dyn + D_safety_dyn
            gate_hit = (required_dist_physics >= s_used)

            if gate_hit:
                a_des = A_MU
                brake_ff = min(1.0, a_des / A_MAX)
                a_meas = 0.0 if v_prev is None else max(0.0, (v_prev - v) / DT)
                e = max(0.0, a_des - a_meas)
                I_err = max(-I_MAX, min(I_MAX, I_err + e*DT))
                brake = max(0.0, min(1.0, brake_ff + (KPB*e + KIB*I_err)/A_MAX))
            else:
                s_eff = max(s_used - D_safety_dyn - v*tau_dyn, EPS)
                a_des = min((v*v) / (2.0 * s_eff), A_MAX)
                a_des = min(a_des, A_MU)
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
                        'gate_hit': gate_hit, 'a_des': a_des, 'brake': brake})

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
                throttle, brake = 0.0, 0.0
                self._kick_until = self._sim_time + KICK_SEC
                stop_release_ignore_until = self._sim_time + 2.0
            else:
                throttle, brake = 0.0, 1.0
        else:
            e_v = v_target - v
            throttle = max(0.0, min(1.0, KP_THROTTLE * e_v))
            brake = 0.0
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
        log_both = (self.args.range_est == 'both' and self.args.compare_csv)
        if log_both:
            import csv
            csv_f = open(self.args.compare_csv, 'w', newline='')
            csv_w = csv.writer(csv_f)
            csv_w.writerow(['t','cls','x','y','w','h','s_pinhole_m','s_depth_m','abs_diff_m','ego_v_mps','mu','sigma_depth'])

        self._init_windows()
        clock = pygame.time.Clock()

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
        self.sensors = SensorRig(self.world.world, self.world.ego, used_range,
                                 enable_top=(not getattr(self.args, 'no_top_cam', False)),
                                 enable_depth=(not getattr(self.args, 'no_depth_cam', False)))
        if self.args.range_est == 'stereo':
            self.range.ensure_stereo()

        FX_, FY_, CX_, CY_ = intrinsics_from_fov(IMG_W, IMG_H, FOV_X_DEG)

        MU = max(0.05, min(1.2, self.args.mu))
        A_MU = MU * 9.81
        v_target = float(V_TARGET)
        dist_total = 0.0
        perf_ms = 0.0; perf_fps = 0.0; ema_loop_ms = DT * 1000.0
        v_prev = None; I_err = 0.0
        gate_hit_ema = 0.0
        sigma_depth_ema = 0.40
        sigma_depth_max_step = 0.50
        stop_persist_count = 0
        hold_blocked = False
        hold_reason   = None
        last_s0 = None
        prev_loc = None
        sim_time = 0.0
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

        # Braking episode tracking for scenario-level results
        episode_active = False
        episode_trigger = None
        episode_v_init = 0.0
        episode_s_init = 0.0
        episode_s_min = float('inf')
        episode_t_start = 0.0

        try:
            t0 = time.time()
            while True:
                frame_id = self.world.tick()
                sim_time += DT
                self._sim_time = sim_time
                tic = time.time()

                frames = self.sensors.read(expected_frame=frame_id)
                io = self._read_frames(frames)
                bgr = io['bgr']
                img_top = io['img_top']
                depth_m = io['depth_m']
                depth_stereo_m = io['depth_stereo_m']

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

                if not getattr(self.args, 'no_opencv', False) and not self.headless:
                    max_vis = 120.0
                    vis = np.clip(depth_m / max_vis, 0.0, 1.0)
                    vis8 = (vis * 255.0).astype(np.uint8)
                    vis8 = cv2.applyColorMap(vis8, cv2.COLORMAP_PLASMA)
                    cv2.imshow('DEPTH', vis8)
                # Perception step (YOLO + depth/stereo + gating)
                perc = self._perception_step(bgr, depth_m, depth_stereo_m, FX_, FY_, CX_, CY_, sim_time, v, MU, log_both, csv_w)
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

                trigger_name = nearest_kind
                if nearest_s_active is not None:
                    last_s0 = nearest_s_active

                # Control step via helper
                throttle, brake, ctrl, hold_blocked, hold_reason, stop_armed, stop_latch_time, stop_release_ignore_until, dbg_map, I_err = \
                    self._control_step(trigger_name, nearest_s_active, nearest_thr,
                                       tl_state, tl_s_active, v, v_target, MU, ema_loop_ms,
                                       last_s0, stop_armed, stop_latch_time, stop_release_ignore_until,
                                       red_green_since, no_trigger_elapsed, no_red_elapsed,
                                       depth_m, depth_stereo_m, nearest_box, nearest_conf,
                                       I_err, v_prev)
                dbg_tau_dyn = dbg_map.get('tau_dyn')
                dbg_D_safety_dyn = dbg_map.get('D_safety_dyn')
                dbg_sigma_depth = dbg_map.get('sigma_depth')
                dbg_gate_hit = dbg_map.get('gate_hit')
                dbg_a_des = dbg_map.get('a_des')
                dbg_brake = dbg_map.get('brake')

                # --- Episode bookkeeping: start/end of braking events ---
                if (not episode_active) and dbg_gate_hit and (nearest_s_active is not None):
                    episode_active = True
                    episode_trigger = trigger_name or 'unknown'
                    episode_v_init = v
                    episode_s_init = nearest_s_active
                    episode_s_min = nearest_s_active
                    episode_t_start = sim_time
                elif episode_active:
                    if nearest_s_active is not None:
                        episode_s_min = min(episode_s_min, nearest_s_active)
                    # Episode considered finished once vehicle has effectively stopped
                    # or gate is no longer hit (returned to cruising).
                    if (v < V_STOP) or (not dbg_gate_hit):
                        stopped = (v < V_STOP)
                        t_to_stop = max(0.0, sim_time - episode_t_start)
                        if self.scenario_logger is not None:
                            try:
                                self.scenario_logger.log(
                                    getattr(self.args, 'scenario_tag', 'default'),
                                    str(episode_trigger or 'unknown'),
                                    MU,
                                    episode_v_init,
                                    episode_s_init,
                                    episode_s_min if math.isfinite(episode_s_min) else episode_s_init,
                                    bool(stopped),
                                    t_to_stop,
                                    bool(self.world.collision_happened if self.world is not None else False),
                                )
                            except Exception:
                                pass
                        episode_active = False
                        episode_trigger = None

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

                if not self.headless:
                    self._draw_hud(self.screen, bgr, img_top, perf_fps, perf_ms, x, y, z, yaw, compass,
                                   frame_id, v, trigger_name, tl_state, throttle, brake, hold_blocked,
                                   hold_reason, no_trigger_elapsed, no_red_elapsed, stop_armed,
                                   stop_release_ignore_until, sim_time, dbg_tau_dyn, dbg_D_safety_dyn,
                                   dbg_sigma_depth, dbg_gate_hit, dbg_a_des, dbg_brake, v_target)

                # Periodic concise console log for tuning
                logN = int(getattr(self.args, 'log_interval_frames', 0) or 0)
                if logN > 0 and (frame_id % logN == 0):
                    print(f"frame={frame_id} fps={perf_fps:.1f} v={v:4.1f} Vtgt={v_target:4.1f} mu={MU:.2f} "
                          f"mode={self.args.range_est} conf={self.detector.conf_thr:.2f} TL={tl_state} trig={trigger_name or 'None'} "
                          f"s_act={(None if nearest_s_active is None else f'{nearest_s_active:.1f}')} thr={(None if nearest_thr is None else f'{nearest_thr:.1f}')} "
                          f"thr_cmd={throttle:.2f} brk={brake:.2f} hold={hold_blocked}")

                loop_ms = (time.time() - tic) * 1000.0
                perf_ms = loop_ms
                perf_fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
                ema_loop_ms = 0.9*ema_loop_ms + 0.1*loop_ms
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
                        self.telemetry.maybe_log(sim_time, v, dbg_tau_dyn, dbg_D_safety_dyn, dbg_sigma_depth, dbg_a_des, dbg_brake)
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
    parser.add_argument('--apply-tire-friction', action='store_true',
                        help='Also set wheel.tire_friction≈mu to make the sim physically slick.')
    parser.add_argument('--persist-frames', type=int, default=2,
                        help='Consecutive frames required to confirm a stop‑sign before arming the stop latch')
    parser.add_argument('--range-est', type=str, default='pinhole',
                        choices=['pinhole', 'depth', 'stereo', 'both'],
                        help='Distance source: monocular pinhole, CARLA depth, stereo vision, or log both (depth vs pinhole)')
    parser.add_argument('--compare-csv', type=str, default=None,
                        help='If set and --range-est=both, write comparisons to this CSV path')
    # Visualization toggles
    parser.add_argument('--no-depth-viz', action='store_true',
                        help='Hide the DEPTH/HUD_DIST OpenCV windows (alias for --no-opencv)')
    parser.add_argument('--no-opencv', action='store_true',
                        help='Disable OpenCV windows entirely (depth/HUD_DIST)')
    parser.add_argument('--no-top-cam', action='store_true',
                        help='Disable spawning the top view camera and hide it from the HUD')
    parser.add_argument('--no-depth-cam', action='store_true',
                        help='Disable spawning the depth camera (range_est depth will be auto-fallback)')
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
