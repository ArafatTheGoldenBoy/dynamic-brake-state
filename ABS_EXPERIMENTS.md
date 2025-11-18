# ABS Actuator Experiment Playbook

This guide captures everything needed to reproduce, observe, and document the
slip-based ABS actuator experiments introduced in `dynamic_brake_state.py`. It
covers runtime setup, live instrumentation, telemetry logging, μ-adaptation, and
the analysis artifacts you can lift directly into a thesis Results chapter.

---

## 1. Signals, architecture, and control loop

Per control tick (Δt ≈ 0.02–0.05 s) the actuator consumes:

| Signal | Units | Source | Notes |
| ------ | ----- | ------ | ----- |
| `v_ego` | m/s | CARLA ego speed | Used for slip normalization and μ estimation. |
| `wheel_speeds[4]` | m/s | CARLA wheel linear speeds | Converted to slips λ₁…λ₄. |
| `a_long` | m/s² | CARLA IMU or v̇ | Needed for μ estimation (`|a_long|/g`). |
| `brake_req` | [0, 1] | High-level AEB logic | Baseline brake demand before ABS. |
| `u_brake` | [0, 1] | Output to `VehicleControl.brake` | Final brake after slip modulation. |

Overall chain:

```
Perception → AEB (brake_req) → Slip-based ABS actuator → Vehicle (CARLA)
                              ↑
                         wheel speeds, v_ego, a_long
```

Everything below happens inside the slip-based ABS actuator.

---

## 2. Slip computation per wheel

Longitudinal slip definition (clamped for robustness):

\[
\lambda_i = \frac{v_\text{ego} - v_{\text{wheel},i}}{\max(v_\text{ego}, \epsilon)}, \quad \lambda_i \in [0, 1]
\]

Implementation snippet from `dynamic_brake_state.py`:

```python
slips = []
for v_w in wheel_speeds:
    v = max(v_ego, 0.1)
    lam = (v - max(0.0, v_w)) / v
    slips.append(min(1.0, max(0.0, lam)))
```

This runs every tick before the PI channels.

---

## 3. Per-wheel PI slip channels

Each wheel owns a virtual PI regulator that outputs an "allowed brake factor"
`f_i ∈ [0, 1]` relative to the base `brake_req`:

- Target slip λ⋆ per wheel (0.10–0.18 depending on μ regime).
- Error `e_i = λ⋆ − λ_i`.
- PI update with anti-windup:

\[
f_i[k] = \text{sat}_{[0,1]}(K_p e_i[k] + I_i[k])
\]
\[
I_i[k] = I_i[k-1] + K_i e_i[k] Δt
\]

If a channel saturates against the error sign, the integral freezes to avoid
windup. `f_i = 1` means the wheel allows full brake usage; `f_i < 1` demands a
reduction because slip exceeded λ⋆.

---

## 4. Scalar brake synthesis

All four per-wheel factors collapse into a single multiplier:

\[
f_\text{global} = \min_i f_i, \qquad u_\text{ABS} = f_\text{global}·\text{brake_req}
\]

Below `v_min_abs` (default 3 m/s) the actuator bypasses slip control and
returns the raw `brake_req`.

---

## 5. μ-adaptation (optional but recommended)

A lightweight estimator tracks the current friction regime:

1. Update μ̂ only when `v_ego > 5 m/s`, `brake_req > 0.3`, and
   `lambda_max ∈ [0.10, 0.25]`.
2. Instantaneous estimate: `μ_inst = |a_long| / g`.
3. Exponential moving average with α ≈ 0.05.
4. Map μ̂ → {high, medium, low} to retune λ⋆, Kp, and Ki, e.g.:

| Regime | μ̂ range | λ⋆ | Kp | Ki |
| ------ | -------- | --- | -- | -- |
| High (dry) | μ̂ > 0.8 | 0.18 | 5.0 | 25.0 |
| Medium (wet) | 0.4 < μ̂ ≤ 0.8 | 0.15 | 4.0 | 20.0 |
| Low (ice) | μ̂ ≤ 0.4 | 0.10 | 3.0 | 12.0 |

This logic lives in `AdaptivePISlipABSActuator` behind the CLI flag
`--abs-mode adaptive` (default). Use `--abs-mode fixed` to lock the gains and
`--abs-mode off` to bypass the actuator entirely.

---

## 6. Runtime recipe

### 6.1. Launching the experiment

Run the main script from the repo root. Example adaptive ABS session with HUD,
NPCs, periodic console logs, telemetry, and scenario logging:

```bash
python dynamic_brake_state.py \
    --preset fast \
    --town Town10HD_Opt \
    --npc-vehicles 10 \
    --log-interval-frames 30 \
    --abs-mode adaptive \
    --telemetry-csv logs/telemetry_abs_adaptive.csv \
    --telemetry-hz 20 \
    --scenario-tag lead_vehicle_15mps \
    --scenario-csv logs/scenarios_lead_vehicle_15mps.csv
```

Repeat the same command with `--abs-mode off` and `--abs-mode fixed` to gather
baselines.

### 6.2. Live instrumentation

- **HUD overlay**: Displays ego speed, control commands, trigger info, plus ABS
  diagnostics (`ABS slip`, `ABS f`, `μ_est`, `regime`).
- **Console breadcrumbs**: `--log-interval-frames N` prints frame summaries
  (FPS, speed, trigger, brake) every N ticks.
- **Snapshots / video**: Press `S` for annotated PNGs or pass `--video-out` to
  record MP4 evidence for qualitative figures.

---

## 7. Telemetry and scenario logs

Frame-level telemetry CSV columns (see `RESULTS.md` §1.1) include:

- Time stamp and ego pose/speed.
- Safety envelope terms (distance margin, TTC surrogate, depth uncertainty).
- Control commands (throttle, brake_req, u_brake).
- ABS diagnostics (`lambda_max`, `abs_factor`, `mu_est`, `abs_regime`).

Scenario CSV rows (see `RESULTS.md` §1.2) summarize each braking episode:
initial speed, initial headway, stop time, stopping distance, and collision flag.
Tag runs with `--scenario-tag` so they’re easy to group offline.

---

## 8. Experiment grid for the Results chapter

For each surface condition configured via `WheelPhysicsControl` or CARLA
material presets:

| Surface | Friction multiplier |
| ------- | ------------------- |
| Dry | 1.0 |
| Wet | 0.6 |
| Ice | 0.2 |
| Split-μ | e.g., left wheels 0.2, right wheels 1.0 |

Run the following controller modes per surface and initial speed (e.g., 60 and
80 km/h):

1. **No ABS** – `--abs-mode off`, `abs_enabled=False` on CARLA wheels.
2. **Fixed PI-ABS** – `--abs-mode fixed`.
3. **Adaptive PI-ABS** – `--abs-mode adaptive`.
4. *(Optional)* CARLA built-in ABS for reference.

Log telemetry/scenario CSVs for each run. Afterward populate the table template
below (copy/paste into your thesis):

| Surface | Controller | Stopping dist [m] | Impact v [km/h] | Peak decel [m/s²] | Mean slip [-] | Slip overshoot [-] | ABS duty [%] | Comfort (max jerk) [m/s³] |
| ------- | ---------- | ----------------- | --------------- | ----------------- | ------------- | ------------------ | ------------ | ------------------------- |
| Dry     | No ABS          | ... | ... | ... | ... | ... | ... | ... |
| Dry     | Fixed PI-ABS    | ... | ... | ... | ... | ... | ... | ... |
| Dry     | Adaptive PI-ABS | ... | ... | ... | ... | ... | ... | ... |
| Wet     | ...             | ... | ... | ... | ... | ... | ... | ... |
| Ice     | ...             | ... | ... | ... | ... | ... | ... | ... |
| Split-μ | ...             | ... | ... | ... | ... | ... | ... | ... |

Recommended derived metrics:

- **Stopping distance**: `x_stop − x_trigger`.
- **Impact speed**: residual speed at collision (if any).
- **Peak deceleration**: max |a_long|.
- **Mean slip**: average `lambda_max` during braking.
- **Slip overshoot**: max(`lambda_max − λ⋆_regime`, 0).
- **ABS duty**: percentage of braking samples with `abs_factor < 1`.
- **Comfort proxy**: max |jerk| from the derivative of `a_long`.

---

## 9. Offline analysis pipeline

1. Run `results_analysis.py` to convert CSVs into plots and aggregates:

   ```bash
   python results_analysis.py \
       --telemetry-csv logs/telemetry_abs_adaptive.csv \
       --scenario-csv logs/scenarios_lead_vehicle_15mps.csv \
       --out-dir results_abs_adaptive \
       --tag abs_adaptive
   ```

2. Use the emitted charts (speed vs. time, brake vs. time, λ vs. μ) and the
   summary JSON/CSV files to populate the thesis Results chapter.

3. Repeat for the `off` and `fixed` modes and compare across friction surfaces
   using the table in §8.

---

## 10. Checklist before publishing

- [ ] Gather HUD screenshots or MP4 clips per surface/controller combo.
- [ ] Verify telemetry and scenario CSVs exist for each run.
- [ ] Run `results_analysis.py` and archive the plots.
- [ ] Fill the comparison table with measured values.
- [ ] Reference this README (`ABS_EXPERIMENTS.md`) from your thesis or GitHub
      README so others can reproduce the actuator study.

Happy experimenting!
