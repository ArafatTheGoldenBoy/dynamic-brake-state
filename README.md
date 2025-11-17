# Dynamic Brake State (CARLA + YOLO)

This project implements a single‑file, object‑aware driving stack for CARLA. It runs an ego vehicle in synchronous mode, detects obstacles with YOLO, estimates distance (depth / stereo / pinhole), and computes a **dynamic safety envelope** to control throttle and brake.

Core script: `dynamic_brake_state.py`

---

## Features

- **CARLA integration**
  - Connects to a CARLA server (`--host`, `--port`).
  - Loads a town (prefers `_Opt` maps such as `Town10HD_Opt`).
  - Spawns an ego taxi (Ford) or Tesla fallback.
  - Optionally spawns NPC vehicles and walkers with Traffic Manager control.

- **Sensor rig**
  - Front RGB camera (main perception input).
  - Optional top‑down RGB camera for visualization.
  - Optional CARLA depth camera.
  - Optional stereo pair for disparity‑based range estimation.
  - Synchronous mode: camera frames are matched to world ticks.

- **Perception**
  - Ultralytics YOLO (`yolo12n.pt` by default) on the front RGB view.
  - Per‑class confidence thresholds and NMS IoU overrides.
  - Lateral gating to ignore other‑lane traffic using intrinsics and object range.
  - Range estimation modes:
    - `pinhole`: monocular using object pixel height (`OBJ_HEIGHT_M`).
    - `depth`: CARLA depth camera (median depth in ROI).
    - `stereo`: OpenCV StereoBM / CUDA SGM.
    - `both`: log pinhole vs depth for offline comparison.
  - Traffic light color inference from ROI if CARLA TL state is unknown.
  - Stop sign persistence and arming logic.

- **Dynamic safety envelope & control**
  - `_safety_envelope()` computes a distance margin using:
    - Ego speed, estimated latency, road friction `mu`, depth uncertainty.
  - `_control_step()` chooses between:
    - Full braking when the “gate” is hit.
    - Approach modulation when still outside the strict safety distance.
  - PI‑shaped brake control tracking desired deceleration.
  - Hold states for `red_light`, `stop_sign`, and generic `obstacle` with:
    - Debounce / wait timers.
    - Throttle “kick” on release to overcome static friction.
  - Steering via waypoint lookahead (pure‑pursuit style) along the lane.

- **Visualization & telemetry**
  - Pygame HUD: ego pose, speed, trigger object, TL state, control signals.
  - Optional OpenCV windows:
    - `DEPTH` pseudo‑color view.
    - `HUD_DIST` text overlay with per‑object X/Y/Z in camera frame.
  - Telemetry CSV logging (`--telemetry-csv`) with safety envelope and control terms.
  - Range comparison CSV when `--range-est both` and `--compare-csv` are set.

- **Presets & hotkeys**
  - Presets (`--preset fast|quality|gpu480|cpu480`) tune YOLO size, device, stereo, and range mode.
  - Runtime hotkeys:
    - `[` / `]`: decrease / increase YOLO confidence threshold.
    - `+` / `-` (incl. keypad): increase / decrease target speed.
    - `0`: reset target speed to default.
    - `ESC` / window close: clean shutdown (windows → sensors → world).

---

## Requirements

- CARLA server (tested with 0.9.x) running on the given `--host`/`--port`.
- Python 3.x with:
  - `carla` Python egg on `PYTHONPATH` (the script auto‑searches common locations).
  - `ultralytics` (for YOLO).
  - `numpy`, `opencv-python`, `pygame`, `torch` (for GPU use).
- YOLO weights file (default: `yolo12n.pt`) in the repo root.

Install example (from the repo folder):

```powershell
python -m pip install ultralytics numpy opencv-python pygame torch
```

Make sure the CARLA egg is discoverable (the script will try typical `dist/carla-*.egg` paths).

---

## How to Run

From the project directory:

```powershell
# Fast preset, fewer pixels, depth range, NPC traffic
python dynamic_brake_state.py --preset fast --town Town10HD_Opt --npc-vehicles 10 --log-interval-frames 30

# Compare pinhole vs depth range estimates (logs to CSV)
python dynamic_brake_state.py --range-est both --compare-csv ranges.csv --yolo-class-thr "traffic light:0.55, stop sign:0.45"

# High quality run with telemetry logging
python dynamic_brake_state.py --preset quality --telemetry-csv telem.csv --telemetry-hz 20
```

For non‑visual / CI‑style runs:

```powershell
python dynamic_brake_state.py --preset fast --headless --no-opencv
```

---

## Notes for Extending

- To add a new trigger object class:
  - Add its height to `OBJ_HEIGHT_M`.
  - Add its normalized name to `TRIGGER_NAMES_NORM`.
  - Optionally set its engage distance via `--engage-override`.
  - Adjust gating with `--gate-frac-override` / `--gate-lateral-override` if needed.
- When adding new sensors, hook them into `SensorRig` using the same queue semantics.
