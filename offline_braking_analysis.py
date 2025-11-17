import argparse
from pathlib import Path

import numpy as np
import pandas as pd


V_STOP = 0.10      # same as dynamic_brake_state.py
BRAKE_THR = 0.05   # treat >5% brake as “braking”


def extract_episodes_from_telemetry(tele: pd.DataFrame) -> pd.DataFrame:
    """Infer braking episodes from telemetry: brake > BRAKE_THR."""
    if not {"t", "v_mps", "brake"}.issubset(tele.columns):
        return pd.DataFrame()

    t = tele["t"].astype(float).values
    v = tele["v_mps"].astype(float).values
    brk = tele["brake"].fillna(0.0).astype(float).values

    if t.size < 2:
        return pd.DataFrame()

    # Instantaneous decel estimate (positive when slowing)
    dt = np.diff(t)
    dv = np.diff(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_inst = -(dv / dt)
    a_inst[~np.isfinite(a_inst)] = 0.0

    brake_on = brk > BRAKE_THR

    # Rising edges mark episode starts
    starts = brake_on & ~np.roll(brake_on, 1)
    starts[0] = brake_on[0]
    episode_id = np.cumsum(starts) * brake_on

    episodes = []
    for eid in np.unique(episode_id):
        if eid == 0:
            continue
        idx = np.where(episode_id == eid)[0]
        if idx.size < 3:
            continue

        i0 = idx[0]
        t0 = t[i0]
        v0 = v[i0]

        # First time below V_STOP after start -> stop index
        post_idx = np.arange(i0, len(t))
        stopped_idx = post_idx[v[post_idx] < V_STOP]
        if stopped_idx.size > 0:
            i_stop = int(stopped_idx[0])
            t_stop = t[i_stop]
            stopped = True
        else:
            i_stop = idx[-1]
            t_stop = t[i_stop]
            stopped = False

        # Stopping distance via integral of v over t
        dist = float(np.trapz(v[i0:i_stop + 1], t[i0:i_stop + 1]))

        # Max decel in this window
        if i_stop > i0:
            a_seg = a_inst[i0:i_stop]
            max_decel = float(np.nanmax(np.clip(a_seg, 0.0, np.inf)))
        else:
            max_decel = 0.0

        tau0 = tele.get("tau_dyn", pd.Series([np.nan] * len(tele))).iloc[i0]
        Dsafe0 = tele.get("D_safety_dyn", pd.Series([np.nan] * len(tele))).iloc[i0]

        episodes.append({
            "episode_idx": int(eid),
            "t_start": float(t0),
            "t_stop": float(t_stop),
            "t_to_stop_est": float(t_stop - t0) if stopped else np.nan,
            "v_init_mps_tele": float(v0),
            "stopped_tele": bool(stopped),
            "dist_travelled_m": dist,
            "max_decel_mps2": max_decel,
            "tau_dyn_start": float(tau0) if pd.notna(tau0) else np.nan,
            "D_safety_dyn_start": float(Dsafe0) if pd.notna(Dsafe0) else np.nan,
        })

    return pd.DataFrame(episodes)


def merge_with_scenarios(episodes: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """Align telemetry-derived episodes with scenario CSV rows by order."""
    if episodes.empty or scen.empty:
        return pd.DataFrame()

    scen = scen.reset_index(drop=True)
    episodes = episodes.reset_index(drop=True)

    n = min(len(episodes), len(scen))
    if len(episodes) != len(scen):
        print(f"[WARN] Episode count mismatch: telemetry={len(episodes)}, scenario={len(scen)}; truncating to {n}")
    episodes = episodes.iloc[:n].reset_index(drop=True)
    scen = scen.iloc[:n].reset_index(drop=True)

    merged = pd.concat([scen, episodes], axis=1)
    return merged


def summarize_for_table(merged: pd.DataFrame) -> None:
    """Print a compact braking performance summary."""
    if merged.empty:
        print("\nNo merged episodes to summarize.")
        return

    print("\n=== Braking performance summary (merged) ===")

    collided = merged["collision"].astype(bool)
    stopped = merged["stopped"].astype(bool)

    collision_rate = 100.0 * collided.mean()
    missed_stop_rate = 100.0 * ((~stopped) | collided).mean()

    trig = merged["trigger_kind"].astype(str).str.lower()
    false_mask = (~collided) & (trig.str.contains("unknown"))
    false_stop_rate = 100.0 * false_mask.mean()

    print(f"  episodes: {len(merged)}")
    print(f"  mean v_init (scenario): {merged['v_init_mps'].mean():.2f} m/s")
    print(f"  mean v_init (telemetry): {merged['v_init_mps_tele'].mean():.2f} m/s")
    print(f"  mean stopping distance: {merged['dist_travelled_m'].mean():.2f} m")
    print(f"  mean max decel: {merged['max_decel_mps2'].mean():.2f} m/s²")
    print(f"  collision rate: {collision_rate:.1f}%")
    print(f"  missed-stop rate (stopped==False or collision): {missed_stop_rate:.1f}%")
    print(f"  false-stop rate (trigger_kind ~ 'unknown'): {false_stop_rate:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline braking episode analysis for dynamic_brake_state telemetry.")
    ap.add_argument("--telemetry-csv", required=True, help="Telemetry CSV path")
    ap.add_argument("--scenario-csv", required=True, help="Scenario CSV path")
    args = ap.parse_args()

    tele_path = Path(args.telemetry_csv)
    scen_path = Path(args.scenario_csv)

    tele = pd.read_csv(tele_path)
    scen = pd.read_csv(scen_path)

    print("Loaded telemetry samples:", tele.shape[0])
    print("Loaded scenario episodes:", scen.shape[0])

    episodes = extract_episodes_from_telemetry(tele)
    print("Extracted episodes from telemetry:", episodes.shape[0])

    merged = merge_with_scenarios(episodes, scen)
    if not merged.empty:
        out_path = scen_path.with_suffix(".merged.csv")
        merged.to_csv(out_path, index=False)
        print(f"Merged episode CSV written to: {out_path}")

    summarize_for_table(merged)


if __name__ == "__main__":
    main()

