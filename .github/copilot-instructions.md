# AI Coding Agent Instructions for Dynamic Brake State

These instructions summarize project‑specific architecture, workflows, and conventions so an AI agent can be productive quickly. Focus on extending `dynamic_brake_state.py` safely and consistently.

## Overview
Single‑file OOP CARLA driving stack (`dynamic_brake_state.py`). Main loop orchestrated by `App.run()` performing: sensor reads → perception (YOLO + depth/stereo + gating) → safety envelope computation → longitudinal & steering control → HUD/logging.

Core components:
- `YOLODetector`: Ultralytics wrapper (configurable size, device, half, classes, per‑class conf & IoU second‑stage NMS). Returns raw boxes before project‑specific gating.
- `RangeEstimator`: Provides stereo disparity→depth (CPU `StereoBM` or CUDA BM/SGM). Fallback to pinhole or CARLA depth depending on `--range-est`.
- `SensorRig`: Spawns RGB front, optional top, optional depth, optional stereo pair. Queues deliver latest (or frame‑matched) images in synchronous world mode.
- `WorldManager`: Connects to CARLA, loads (prefers `_Opt` maps), sets sync mode & fixed `DT`, spawns ego taxi / Tesla fallback and optional NPC vehicles & walkers (with autopilot & AI controllers). Applies tire friction if requested.
- `App`: Argument parsing, preset application, initialization, main runtime loop, dynamic braking logic, HUD & telemetry.

## Runtime Control & Hotkeys
- `[` / `]`: Decrease / increase detector confidence (`YOLODetector.conf_thr` in 0.05 steps).
- `+` / `-` (including keypad): Increase / decrease target speed (`v_target`) by ≈2 km/h (0.5556 m/s).
- `0`: Reset target speed to default `V_TARGET`.
- `ESC` or window close: Graceful shutdown path (destroys sensors, world, closes windows).

## Distance / Range Modes (`--range-est`)
- `pinhole`: Monocular range via object pixel height & `OBJ_HEIGHT_M` table.
- `depth`: CARLA depth camera (`median_depth_in_box`).
- `stereo`: Disparity → depth via `RangeEstimator.stereo_depth()`; CUDA optional (`--stereo-cuda`, `--stereo-method bm|sgm`).
- `both`: Logs pinhole vs depth comparisons to CSV (requires `--compare-csv`). Internal control uses depth first if available.
Fallback rules: If selected mode disabled by flags (e.g. `--no-depth-cam`), auto-switch to pinhole.

## Perception Pipeline Highlights
1. `YOLODetector.predict_raw(bgr)` produces `(classIds, confs, boxes)`.
2. Per‑class confidence override via `--yolo-class-thr` parsed by `parse_per_class_conf_map` (normalized label names) before gating.
3. Gating: lateral band & max lateral meters per class via `--gate-frac-override`, `--gate-lateral-override`; vehicle & specified classes only. Center band computed with image width & intrinsics.
4. Range selection & caching per box (`depth_cache_depth`, `depth_cache_stereo`).
5. Traffic light color classification fallback: `estimate_tl_color_from_roi` when CARLA TL actor color unknown.
6. Merge ROI TL inference with CARLA actor state → final `tl_state`.
7. Stop sign persistence: `--persist-frames` consecutive frames before arming `stop_armed`.

## Dynamic Safety Envelope & Braking
- `_safety_envelope()` computes smoothed `tau_dyn`, `D_safety_dyn`, and `sigma_depth` using latency estimate, detection confidence, friction (`mu_short = 0.90 - MU`), and depth sigma from MAD.
- Braking logic in `_control_step()` distinguishes gate hit (required physics distance ≥ smoothed measured distance) vs approach modulation.
- PI shaping: feedforward fraction + error integration (`KPB`, `KIB`, limited by `I_MAX`).
- Hold states: `red_light`, `stop_sign`, `obstacle` with individual debounce / wait timers (`CLEAR_DELAY_*`, `STOP_WAIT_S`). Release triggers a throttle "kick" (`KICK_SEC`, `KICK_THR`).

## Extending Object Classes
To add a new trigger class:
1. Add height (meters) to `OBJ_HEIGHT_M`.
2. Include normalized label in `TRIGGER_NAMES_NORM` (use `_norm_label`).
3. Optionally define engage distance with `--engage-override` (or extend parser defaults).
4. Provide gating overrides if needed (`--gate-frac-override`, `--gate-lateral-override`).
5. If part of braking triggers, ensure detection name passes the norm filters in `_perception_step`.

## Presets (`--preset fast|quality|gpu480|cpu480`)
- `fast`: Smaller image (480), half precision if CUDA, agnostic NMS, depth range.
- `quality`: Full image (640), CUDA (if avail), augment, stereo CUDA SGM.
- `gpu480`: 480 + CUDA + half + pinhole.
- `cpu480`: 480 + CPU + depth camera.
These mutate parsed `args` before constructing `App`.

## Logging & Telemetry
- Comparison CSV (range mode `both`): columns `[t, cls, x, y, w, h, s_pinhole_m, s_depth_m, abs_diff_m, ego_v_mps, mu, sigma_depth]`.
- Telemetry CSV (`--telemetry-csv`): `[t, v_mps, tau_dyn, D_safety_dyn, sigma_depth, a_des, brake]` via `_TelemetryLogger` at `--telemetry-hz`.
- Console periodic frame log controlled by `--log-interval-frames` (0 disables).

## Key Synchronous Simulation Assumptions
- CARLA synchronous mode with fixed `DT=0.02` (50 Hz) → world tick precedes sensor read; sensors drained to latest frame.
- Latency estimation uses smoothed loop time (`ema_loop_ms`) influencing safety envelope.
- Ego steering via local waypoint lookahead (pure pursuit style) in `_steer_to_waypoint` with dynamic lookahead distance.

## Shutdown Order
App finalizer: windows → sensors (`SensorRig.destroy()`) → world (`WorldManager.destroy()`) → close CSV / telemetry. Keep this ordering when modifying cleanup logic.

## Contribution Guidelines (Project-Specific)
- Preserve real-time loop constraints: avoid blocking > `DT` inside perception/control blocks.
- When adding new sensors, integrate into `SensorRig.read()` using same queue fetch semantics (`_get_latest` or `_get_for_frame`).
- Extend arguments in `parse_args()`; mirror any preset impacts in `_apply_preset`.
- Maintain label normalization via `_norm_label` for any per-class maps.
- Use existing smoothing patterns (EMA) for new dynamic parameters to avoid abrupt control changes.

## Testing Suggestions
- Run with `--headless` for non-visual CI-style checks.
- Use `--preset fast` for quicker startup; validate braking behavior by spawning NPCs (`--npc-vehicles 5`).
- For range comparisons, run `--range-est both --compare-csv compare.csv` and inspect diff distributions offline.

## Typical Run Commands
```powershell
python dynamic_brake_state.py --preset fast --town Town10HD_Opt --npc-vehicles 10 --log-interval-frames 30
python dynamic_brake_state.py --range-est both --compare-csv ranges.csv --yolo-class-thr "traffic light:0.55, stop sign:0.45"
python dynamic_brake_state.py --preset quality --telemetry-csv telem.csv --telemetry-hz 20
```

Please review: Are engage distance overrides & class addition steps clear? Any missing internal invariants you want documented?
