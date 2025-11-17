# Experimental Results & Evaluation

This document explains how to use the repository to produce quantitative results for a thesis or paper. It focuses on **what is logged**, **how to run experiments**, and **how to analyze the CSVs**.

## 1. Logged Metrics

### 1.1 Telemetry (`--telemetry-csv`)

Per time step the script logs:

- `t` – simulation time [s]
- `v_mps` – ego speed [m/s]
- `tau_dyn` – dynamic time headway τ (from `_safety_envelope`)
- `D_safety_dyn` – dynamic safety distance [m]
- `sigma_depth` – depth uncertainty [m]
- `a_des` – desired deceleration [m/s²]
- `brake` – normalized brake command [0–1]

Use this for plots such as `v(t)`, `D_safety_dyn(t)`, and `tau_dyn(t)`.

### 1.2 Braking Episodes (`--scenario-tag` + `--scenario-csv`)

Each braking “episode” (gate hit + braking until stop/release) logs a single row:

- `scenario` – tag from `--scenario-tag` (e.g. `lead_vehicle_15mps`)
- `trigger_kind` – type of trigger (`car`, `person`, `traffic light`, `stop sign`, etc.)
- `mu` – friction coefficient in that run
- `v_init_mps` – ego speed when braking episode started
- `s_init_m` – distance to trigger when braking started
- `s_min_m` – minimum distance reached during episode
- `stopped` – boolean: whether ego reached near-zero speed (`v < V_STOP`)
- `t_to_stop_s` – time from episode start to stop
- `collision` – boolean: whether any collision was detected

Use this to compute stopping margins, time-to-stop, and collision rates.

### 1.3 Range Comparison (`--range-est both` + `--compare-csv`)

When `--range-est both` is used, the script logs per detection:

- `t` – time [s]
- `cls` – class label (e.g. `car`, `person`, `traffic light`)
- `x, y, w, h` – bounding box in pixels
- `s_pinhole_m` – pinhole (monocular) distance estimate [m]
- `s_depth_m` – depth-camera-based distance [m]
- `abs_diff_m` – absolute distance error |`s_pinhole_m - s_depth_m`| [m]
- `ego_v_mps` – ego speed [m/s] at that time
- `mu` – friction
- `sigma_depth` – depth uncertainty estimate [m]

Use this for range accuracy: MAE/RMSE by distance bin and by class.

### 1.4 Latency & Video

- `--extra-latency-ms` – artificial latency (ms) added into the safety-envelope computation for sensitivity analysis.
- `--video-out` – optional MP4 recording of the front camera view.
- Snapshot key `S` – saves `snapshot_<frame_id>.png` for qualitative figures.

---

## 2. Running Experiments

### 2.1 Telemetry + Braking Episodes

Example: lead vehicle scenario at ~15 m/s, logging telemetry and episodes:

```powershell
python dynamic_brake_state.py `
  --preset fast `
  --scenario-tag lead_vehicle_15mps `
  --scenario-csv scenarios_lead_vehicle_15mps.csv `
  --telemetry-csv telemetry_lead_vehicle_15mps.csv
```

Repeat with different `--mu`, `--preset`, initial speeds, or detector configs to build tables.

### 2.2 Range-Estimation Accuracy

Compare pinhole vs depth:

```powershell
python dynamic_brake_state.py `
  --range-est both `
  --compare-csv ranges.csv `
  --telemetry-csv telemetry_ranges.csv
```

This produces `ranges.csv` with per-detection distance errors.

### 2.3 Latency Ablation

Study how dynamic safety envelope reacts to added latency:

```powershell
python dynamic_brake_state.py `
  --preset fast `
  --extra-latency-ms 50 `
  --scenario-tag lead_vehicle_15mps_latency50 `
  --scenario-csv scenarios_latency50.csv `
  --telemetry-csv telemetry_latency50.csv
```

Sweep `--extra-latency-ms` (e.g. 0, 25, 50, 100 ms) and compare `tau_dyn` / `D_safety_dyn` and collision rates.

### 2.4 Qualitative Videos & Snapshots

Record video and snapshots for figures:

```powershell
python dynamic_brake_state.py `
  --preset fast `
  --video-out run_lead_vehicle.mp4
```

During the run (with GUI enabled), press `S` to save annotated PNG snapshots.

---

## 3. Analyzing Results (`results_analysis.py`)

The script `results_analysis.py` summarizes CSVs and produces ready-to-use plots.

### 3.1 Telemetry + Scenarios

```powershell
python results_analysis.py `
  --telemetry-csv telemetry_lead_vehicle_15mps.csv `
  --scenario-csv scenarios_lead_vehicle_15mps.csv `
  --out-dir results_lead_vehicle_15mps `
  --tag lead_vehicle_15mps
```

Outputs:

- `results_lead_vehicle_15mps/lead_vehicle_15mps_speed_brake.png`
- `results_lead_vehicle_15mps/lead_vehicle_15mps_D_safety.png`
- `results_lead_vehicle_15mps/lead_vehicle_15mps_tau_dyn.png`
- Console summary of braking episodes per `scenario`:
  - mean `v_init_mps`, `s_init_m`, `s_min_m`
  - mean `t_to_stop_s` (stopped-only)
  - collision rate

### 3.2 Range Comparison

```powershell
python results_analysis.py `
  --compare-csv ranges.csv `
  --out-dir results_ranges `
  --tag ranges
```

Outputs:

- `results_ranges/ranges_range_error_scatter.png` – error vs depth distance.
- Console summary:
  - overall MAE & RMSE
  - MAE/RMSE by distance bins (0–10, 10–20, … m)
  - MAE per class label.

### 3.3 Combined

You can pass all three at once:

```powershell
python results_analysis.py `
  --telemetry-csv telemetry.csv `
  --scenario-csv scenarios.csv `
  --compare-csv ranges.csv `
  --out-dir results_all `
  --tag experiment1
```

Figures and summaries are then grouped under `results_all/`.

---

## 4. Using This in a Thesis / Paper

- **Detection & range accuracy**: use `ranges.csv` + `results_analysis.py` to report MAE/RMSE per distance bin and per class.
- **Braking performance**: use `scenarios_*.csv` to report stopping distances, time-to-stop, and collision rates for each scenario and friction level.
- **Dynamic safety envelope**: use telemetry plots to show how `tau_dyn` and `D_safety_dyn` adapt to speed, friction, and latency.
- **Latency ablation**: sweep `--extra-latency-ms` and compare `D_safety_dyn` and collision rates across runs.
- **Qualitative evidence**: embed PNG snapshots and MP4 clips generated via `--video-out` and the `S` key.

This structure makes it easy for reviewers (and your future self) to see exactly how experiments were run and how each figure in the Results section was obtained.
