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
  - Telemetry CSV logging (`--telemetry-csv`) with safety envelope terms, ABS slip stats, loop/detector timing, ground-truth headway, and false-stop flags.
  - Episode‑level scenario CSV logging (`--scenario-csv`) with initial headway (estimate + GT), min gap, stop time, collision flag, range/time margins, and ABS duty.
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

# High quality run with telemetry logging + TTC/persistence tuning
python dynamic_brake_state.py --preset quality --telemetry-csv telem.csv --telemetry-hz 20 \
  --gate-confirm-frames 4 --ttc-confirm-s 2.0 --min-aeb-speed 4.0
```

For non-visual / CI-style runs:

```powershell
python dynamic_brake_state.py --preset fast --headless --no-opencv
```

If you need tighter or looser AEB sensitivity, tune:

- `--min-aeb-speed` – minimum ego speed (m/s) required before automatic braking is allowed.
- `--gate-confirm-frames` – consecutive frames that must hit the safety gate before braking starts.
- `--ttc-confirm-s` – minimum (low) TTC duration that must persist before AEB is confirmed.

These knobs feed into the hazard-confirm timer so the telemetry/scenario CSVs now include TTC and reaction-time metrics you can cite in a thesis.

---

## Experiments, Logging, and Analysis

### Thesis workflow at a glance

1. **Start CARLA** in synchronous mode (`CarlaUE4.exe -quality-level=Epic`) and load the desired town.
2. **Run `dynamic_brake_state.py`** with telemetry + scenario logging enabled (examples below). Keep YOLO weights in the repo root.
3. **Collect CSVs**: telemetry (per frame), scenarios (per braking event), range comparisons (per detection), stereo-vs-depth (if stereo).
4. **Post-process** all CSVs with `results_analysis.py` to generate plots/tables (`results_<tag>/...`).
5. **Insert the figures/tables** (speed vs distance, TTC margins, slip histograms, false-stop counts, runtime histograms) directly into the Results chapter.

### Logged metrics (CSV columns)

- **Telemetry (`--telemetry-csv`)**: `t`, `v_mps`, `tau_dyn`, `D_safety_dyn`, `sigma_depth`, `a_des`, `brake`, `lambda_max`, `abs_factor`, `mu_est`, `mu_regime`, `loop_ms`, `loop_ms_max`, `detect_ms`, `latency_ms`, `a_meas`, `x_rel_m`, `range_est_m`, `ttc_s`, `gate_hit`, `gate_confirmed`, `false_stop_flag`.
- **Braking episodes (`--scenario-csv`)**: `scenario`, `trigger_kind`, `mu`, `v_init_mps`, `s_init_m`, `s_min_m`, `s_init_gt_m`, `s_min_gt_m`, `stopped`, `t_to_stop_s`, `collision`, `range_margin_m`, `tts_margin_s`, `ttc_init_s`, `ttc_min_s`, `reaction_time_s`, `max_lambda`, `mean_abs_factor`, `false_stop`.
- **Range comparison (`--range-est both --compare-csv`)**: raw detections with pinhole vs depth distances plus per-class errors, μ, ego speed, and depth-uncertainty snapshots.
- **Stereo comparison (`--stereo-compare-csv`)**: disparity-based vs depth-camera ground-truth to quantify stereo bias.

### Capturing runs for the thesis

```powershell
# Baseline lead-vehicle study (telemetry + scenario + range)
python dynamic_brake_state.py `
  --preset fast `
  --scenario-tag lead_vehicle_15mps `
  --scenario-csv logs/scenarios_lead_vehicle_15mps.csv `
  --telemetry-csv logs/telemetry_lead_vehicle_15mps.csv `
  --range-est both `
  --compare-csv logs/ranges_lead_vehicle_15mps.csv

# Latency ablation (repeat for 0/25/50/100 ms)
python dynamic_brake_state.py `
  --preset fast `
  --extra-latency-ms 50 `
  --scenario-tag lead_vehicle_latency50 `
  --scenario-csv logs/scenarios_latency50.csv `
  --telemetry-csv logs/telemetry_latency50.csv

# Stereo-focused experiment with CUDA SGM and TTC gating tuned for distance noise
python dynamic_brake_state.py `
  --range-est stereo `
  --stereo-cuda `
  --telemetry-csv logs/telemetry_stereo.csv `
  --scenario-csv logs/scenarios_stereo.csv `
  --compare-csv logs/ranges_stereo.csv `
  --stereo-compare-csv logs/stereo_vs_depth.csv `
  --gate-confirm-frames 5 `
  --ttc-confirm-s 1.5
```

### Offline analysis script (`results_analysis.py`)

This helper turns CSVs into thesis-ready material:

- Telemetry plots: `*_speed_gap.png`, `*_brake_slip.png`, `*_ttc.png`, `*_tau_dyn.png`.
- Runtime and latency histograms (`*_loop_ms_hist.png`, `*_latency_ecdf.png`).
- μ-regime summary tables, time-to-stop pivots, and reaction-time box plots (`*_reaction_box.png`).
- Range-error violin/CDF plots and per-class MAE tables.
- False-stop counts per trigger type using the new `false_stop_flag` columns.

Example end-to-end analysis:

```powershell
python results_analysis.py `
  --telemetry-csv logs/telemetry_lead_vehicle_15mps.csv `
  --scenario-csv logs/scenarios_lead_vehicle_15mps.csv `
  --compare-csv logs/ranges_lead_vehicle_15mps.csv `
  --out-dir results_lead_vehicle `
  --tag lead_vehicle
```

Repeat the command per scenario/tag; the script writes PNGs and CSV summary tables (mean TTC, braking margin, WCET stats) to the chosen `--out-dir`. Include those artifacts directly in the thesis.

### Suggested thesis experiments

| Study | Purpose | Recommended settings |
| --- | --- | --- |
| μ sweep | Demonstrate robustness to friction | Run `--mu 0.3/0.6/0.9` with `--apply-tire-friction`, keep scenario, compare `range_margin_m`, `collision`. |
| Latency ablation | Show safety envelope expansion with added delay | Sweep `--extra-latency-ms` (0–100), plot `tau_dyn` and `t_to_stop_s` shifts. |
| Range-estimator comparison | Quantify depth vs stereo vs pinhole | Use `--range-est both`, run `results_analysis.py` to produce per-class MAE; optionally repeat with `--range-est stereo` + `--stereo-compare-csv`. |
| Detector runtime | Evaluate WCET / FPS | Collect `loop_ms`, `loop_ms_max`, `detect_ms` columns; report percentiles. |
| False-stop auditing | Document failure cases | Use telemetry `false_stop_flag` and scenario `false_stop` columns; capture qualitative frames with `S`/`--video-out`. |
| ABS/slip control | Compare `--abs-mode off|fixed|adaptive` | Log `lambda_max`, `abs_factor`, `mu_est`, plus `t_to_stop_s` from scenario CSVs. |

### Open items & terminology

- **Persistent object tracking** – `_control_step()` currently selects the obstacle that is nearest in the *current* frame and only smooths its distance via a single-pole EMA (`s_used = 0.7·last_s0 + 0.3·nearest_s_active`). There is no notion of a track ID, IoU-based association, or per-object life cycle, so the braking target can change whenever detections flicker. Closing this gap would involve keeping a persistent track (ID, distance, velocity) across frames and only switching to a new obstacle after multi-frame confirmation/hysteresis.
- **True measured sensor-to-actuator delay** – `_safety_envelope()` derives `latency_s = max(DT, (ema_loop_ms + extra_ms)/1000.0) + 0.03`, which is an estimate based on observed loop timing plus a fixed pad. This number is logged in telemetry, but it is *not* a direct measurement of “camera frame timestamp minus brake command timestamp.” To publish true sensor-to-actuator latency you would need to record precise timestamps for (a) when CARLA delivers each sensor frame, (b) when the controller issues a non-zero brake command, and ideally (c) when the vehicle dynamics reflect that command, then log their differences. Until that instrumentation exists, describe the logged latency as an estimate, not an empirical measurement.

### ABS actuator experiments (summary)

- Switch controller modes with `--abs-mode off|fixed|adaptive`; keep `--apply-tire-friction` enabled so CARLA tire friction matches the requested μ.
- Telemetry columns `lambda_max`, `abs_factor`, and `mu_est` let you compute slip duty cycles and μ-regime transitions.
- Recommended grid: μ = 0.9 / 0.6 / 0.3 (plus split-μ), speeds = 15 / 22 m/s, controllers = 3 modes → 12+ runs.
- Use scenario CSVs to tabulate `t_to_stop_s`, `s_min_gt_m`, and collision flags. Pair plots with telemetry-derived jerk/slip charts from `results_analysis.py`.

### Lead vehicle stereo example (ready to copy)

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

python results_analysis.py `
  --telemetry-csv logs/telemetry_lead_vehicle_yolo12_stereo1.csv `
  --scenario-csv logs/scenarios_lead_vehicle_yolo12_stereo1.csv `
  --compare-csv logs/ranges_lead_vehicle_yolo12_stereo1.csv `
  --stereo-compare-csv logs/stereo_vs_depth_lead_vehicle1.csv `
  --out-dir results_lead_vehicle `
  --tag lead_vehicle
```

Drop the generated PNGs/tables into your paper along with frame grabs from `--video-out`.

---

## Thesis Usage

In a research or thesis context, this project is used to quantitatively evaluate a dynamic safety-envelope-based braking controller under different sensing, friction, and latency conditions in CARLA. The simulator is configured to generate repeatable traffic scenarios with controlled ego speed, road friction, and obstacle placement. Each run logs frame-level telemetry (vehicle speed, safety distance, τ, TTC, depth uncertainty, ABS slip metrics, ground-truth headway, loop/detector timing/WCET, and false-stop flags) plus episode-level outcomes (initial headway estimate + GT, minimum gap, stopping time, range/time/TTC margins, reaction time, ABS duty, and collision/false-stop status) via `--telemetry-csv` and `--scenario-csv`. Additional experiments compare monocular, depth-camera, and stereo range estimates via `--range-est both --compare-csv`, emulate slower perception stacks using `--extra-latency-ms`, and toggle slip control modes with `--abs-mode`. The accompanying `results_analysis.py` script turns these logs into telemetry plots, μ-regime tables, margin CDFs, runtime histograms, TTC box plots, range-error charts, and false-stop summaries that can be dropped directly into the Results chapter.

---

## Notes for Extending

- To add a new trigger object class:
  - Add its height to `OBJ_HEIGHT_M`.
  - Add its normalized name to `TRIGGER_NAMES_NORM`.
  - Optionally set its engage distance via `--engage-override`.
  - Adjust gating with `--gate-frac-override` / `--gate-lateral-override` if needed.
- When adding new sensors, hook them into `SensorRig` using the same queue semantics.
