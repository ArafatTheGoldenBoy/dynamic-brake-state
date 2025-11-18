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
  - Episode‑level scenario CSV logging (`--scenario-csv`) with initial speed/distance, min distance, stop time, and collision flag.
  - Range comparison CSV when `--range-est both` and `--compare-csv` are set.
  - Optional MP4 video recording (`--video-out`) of the front camera feed.
  - On‑demand image snapshots from the front camera by pressing `S`.

- **Presets & hotkeys**
  - Presets (`--preset fast|quality|gpu480|cpu480`) tune YOLO size, device, stereo, and range mode.
  - Runtime hotkeys:
    - `[` / `]`: decrease / increase YOLO confidence threshold.
    - `+` / `-` (incl. keypad): increase / decrease target speed.
    - `0`: reset target speed to default.
    - `S`: save a snapshot PNG from the front camera.
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

## Experiments, Logging, and Analysis

- **Telemetry & scenarios**
  - Use `--telemetry-csv telem.csv` to log frame‑level data: speed, dynamic safety distance, time‑to‑collision surrogate, depth uncertainty, and brake command.
  - Use `--scenario-csv scenarios.csv --scenario-tag my_scenario` to log episode‑level braking outcomes: initial speed/distance, minimum distance, stop time, and whether a collision occurred.

- **Range estimation evaluation**
  - Run with `--range-est both --compare-csv ranges.csv` to log pinhole vs depth distance estimates per detection.
  - Analyze distance accuracy (MAE/RMSE, per‑class error, error vs. range) with the helper script `results_analysis.py`.

- **Latency ablation**
  - Use `--extra-latency-ms N` to inject additional perception/control latency into the safety envelope computation, emulating slower perception stacks.
  - Compare braking performance across different `N` using telemetry and scenario CSVs.

- **Offline analysis script**
  - `results_analysis.py` provides quick plots and summaries for telemetry, scenario, and range comparison CSVs.
  - Example:

    ```powershell
    python results_analysis.py --telemetry-csv telem.csv --scenario-csv scenarios.csv --compare-csv ranges.csv --out-dir results --tag baseline
    ```

- **Stereo range experiments**
  - Example combined run using stereo range estimation with telemetry, scenario, and range comparison logging:

    ```powershell
    python dynamic_brake_state.py `
      --stereo-cuda `
      --range-est stereo `
      --telemetry-csv telemetry_stereo.csv `
      --scenario-csv scenarios_stereo.csv `
      --compare-csv ranges_stereo.csv `
      --stereo-compare-csv stereo_vs_depth.csv
    ```

  - Offline analysis of the stereo range comparison CSV:

    ```powershell
    python results_analysis.py `
      --stereo-compare-csv ranges_stereo.csv `
      --out-dir results_stereo `
      --tag stereo
    ```

- **Lead vehicle stereo scenario (15 m/s)**
  - Example CARLA run with a lead vehicle scenario at ~15 m/s, logging telemetry, scenarios, range comparisons, and stereo-vs-depth errors while spawning extra NPC traffic:

    ```powershell
    python dynamic_brake_state.py `
      --stereo-cuda `
      --scenario-tag lead_vehicle_15mps1 `
      --scenario-csv logs/scenarios_lead_vehicle_yolo12_stereo1.csv `
      --telemetry-csv logs/telemetry_lead_vehicle_yolo12_stereo1.csv `
      --range-est stereo `
      --compare-csv logs/ranges_lead_vehicle_yolo12_stereo1.csv `
      --stereo-compare-csv logs/stereo_vs_depth_lead_vehicle1.csv `
      --log-interval-frames 10 `
      --npc-vehicles 40
    ```

  - Offline analysis for this lead-vehicle stereo run:

    ```powershell
    python results_analysis.py `
      --telemetry-csv logs/telemetry_lead_vehicle_yolo12_stereo1.csv `
      --scenario-csv logs/scenarios_lead_vehicle_yolo12_stereo1.csv `
      --compare-csv logs/ranges_lead_vehicle_yolo12_stereo1.csv `
      --stereo-compare-csv logs/stereo_vs_depth_lead_vehicle1.csv `
      --out-dir results_lead_vehicle `
      --tag lead_vehicle
    ```

For more detailed experiment recipes and thesis‑style result suggestions, see `RESULTS.md`.

### ABS actuator experiments

- `ABS_EXPERIMENTS.md` documents the complete workflow for the slip-based ABS actuator:
  - Signal definitions, per-wheel PI control, and μ-adaptation strategy.
  - Runtime commands for `--abs-mode off|fixed|adaptive`, telemetry/scenario logging, and HUD diagnostics.
  - Surface/controller experiment grid, required metrics, and ready-to-fill tables for the thesis Results chapter.
  - Offline analysis workflow (`results_analysis.py`) plus a publication-ready checklist.

---

## Thesis Usage

In a research or thesis context, this project is used to quantitatively evaluate a dynamic safety‑envelope‑based braking controller under different sensing and latency conditions in CARLA. The simulator is configured to generate repeatable traffic scenarios with controlled ego speed, road friction, and obstacle placement. For each run, the system logs frame‑level telemetry (vehicle speed, dynamic safety distance, time‑to‑collision surrogate, depth uncertainty, and brake command) as well as episode‑level outcomes (initial speed and distance, minimum distance, stopping time, and collision events) using `--telemetry-csv` and `--scenario-csv`. Additional experiments compare monocular, depth‑camera, and stereo range estimates via `--range-est both --compare-csv`, and emulate slower perception stacks using `--extra-latency-ms`. The accompanying `results_analysis.py` script converts these logs into summary statistics and plots (e.g., braking distance vs. speed, collision rate per scenario, range‑estimation error vs. distance) that can be used directly in the quantitative tables and figures of the thesis Results chapter.

---

## Notes for Extending

- To add a new trigger object class:
  - Add its height to `OBJ_HEIGHT_M`.
  - Add its normalized name to `TRIGGER_NAMES_NORM`.
  - Optionally set its engage distance via `--engage-override`.
  - Adjust gating with `--gate-frac-override` / `--gate-lateral-override` if needed.
- When adding new sensors, hook them into `SensorRig` using the same queue semantics.
