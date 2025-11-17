import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_telemetry(path: Path):
    t, v, tau, d_safe, sigma, a_des, brake = [], [], [], [], [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            v.append(float(row["v_mps"]))
            tau.append(float(row["tau_dyn"]) if row["tau_dyn"] not in ("", "None") else np.nan)
            d_safe.append(float(row["D_safety_dyn"]) if row["D_safety_dyn"] not in ("", "None") else np.nan)
            sigma.append(float(row["sigma_depth"]) if row["sigma_depth"] not in ("", "None") else np.nan)
            a_des.append(float(row["a_des"]) if row["a_des"] not in ("", "None") else np.nan)
            brake.append(float(row["brake"]) if row["brake"] not in ("", "None") else np.nan)
    return {
        "t": np.array(t),
        "v": np.array(v),
        "tau": np.array(tau),
        "D_safe": np.array(d_safe),
        "sigma": np.array(sigma),
        "a_des": np.array(a_des),
        "brake": np.array(brake),
    }


def load_scenarios(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_range_compare(path: Path):
    """Load --range-est both compare CSV: pinhole vs depth distances.

    Expected columns: t, cls, x, y, w, h, s_pinhole_m, s_depth_m,
    abs_diff_m, ego_v_mps, mu, sigma_depth
    """
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["s_pinhole_m"] = float(r["s_pinhole_m"])
                r["s_depth_m"] = float(r["s_depth_m"])
                r["abs_diff_m"] = float(r["abs_diff_m"])
            except Exception:
                continue
            rows.append(r)
    return rows


def summarize_scenarios(rows):
    by_scenario = {}
    for r in rows:
        scen = r.get("scenario", "default")
        by_scenario.setdefault(scen, []).append(r)

    print("\n=== Scenario summary ===")
    for scen, rs in by_scenario.items():
        v_init = np.array([float(r["v_init_mps"]) for r in rs])
        s_init = np.array([float(r["s_init_m"]) for r in rs])
        s_min = np.array([float(r["s_min_m"]) for r in rs])
        t_stop = np.array([float(r["t_to_stop_s"]) for r in rs])
        stopped = np.array([r["stopped"].lower() == "true" for r in rs])
        collided = np.array([r["collision"].lower() == "true" for r in rs])

        print(f"\nScenario: {scen}")
        print(f"  episodes: {len(rs)}")
        print(f"  mean v_init: {v_init.mean():.2f} m/s")
        print(f"  mean s_init: {s_init.mean():.2f} m")
        print(f"  mean s_min:  {s_min.mean():.2f} m")
        if stopped.any():
            print(f"  mean t_stop (stopped only): {t_stop[stopped].mean():.2f} s")
        else:
            print("  mean t_stop: n/a (no full stops)")
        print(f"  collision rate: {collided.mean() * 100:.1f}%")


def plot_telemetry(tele, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    t = tele["t"]
    v = tele["v"]
    D_safe = tele["D_safe"]
    tau = tele["tau"]
    brake = tele["brake"]

    # Speed and brake vs time
    plt.figure(figsize=(8, 4))
    plt.plot(t, v, label="speed v(t) [m/s]")
    if np.isfinite(brake).any():
        plt.plot(t, brake * max(1.0, np.nanmax(v)), label="brake (scaled)")
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_speed_brake.png", dpi=200)
    plt.close()

    # Safety distance vs time
    plt.figure(figsize=(8, 4))
    plt.plot(t, D_safe, label="D_safety_dyn(t)")
    plt.xlabel("time [s]")
    plt.ylabel("distance [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_D_safety.png", dpi=200)
    plt.close()

    # Tau vs time
    plt.figure(figsize=(8, 4))
    plt.plot(t, tau, label="tau_dyn(t)")
    plt.xlabel("time [s]")
    plt.ylabel("tau [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_tau_dyn.png", dpi=200)
    plt.close()


def analyze_range_compare(rows, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("\nNo range-compare rows to analyze.")
        return

    cls_names = sorted(set(r["cls"] for r in rows))
    dists = np.array([r["s_depth_m"] for r in rows])
    errs = np.array([r["abs_diff_m"] for r in rows])

    print("\n=== Range comparison summary (pinhole vs depth) ===")
    print(f"  total detections: {len(rows)}")
    print(f"  overall MAE: {np.mean(np.abs(errs)):.3f} m")
    print(f"  overall RMSE: {np.sqrt(np.mean(errs**2)):.3f} m")

    # Error vs distance bins
    bins = [0, 10, 20, 30, 40, 60, 80]
    bin_indices = np.digitize(dists, bins)
    print("\n  Error by range bin:")
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if not mask.any():
            continue
        mae = np.mean(np.abs(errs[mask]))
        rmse = np.sqrt(np.mean(errs[mask] ** 2))
        print(f"    {bins[i-1]:2d}-{bins[i]:2d} m: MAE={mae:.3f} m, RMSE={rmse:.3f} m, N={mask.sum()}")

    # Per-class MAE
    print("\n  Error by class:")
    for cls in cls_names:
        mask = [r["cls"] == cls for r in rows]
        if not any(mask):
            continue
        e = np.array([rows[i]["abs_diff_m"] for i, m in enumerate(mask) if m])
        print(f"    {cls:15s}: MAE={np.mean(np.abs(e)):.3f} m, N={e.size}")

    # Scatter plot error vs distance
    plt.figure(figsize=(6, 5))
    plt.scatter(dists, errs, s=4, alpha=0.3)
    plt.axhline(0.0, color="k", linewidth=0.5)
    plt.xlabel("depth distance s_depth [m]")
    plt.ylabel("error |s_pinhole - s_depth| [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_range_error_scatter.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Quick analysis of dynamic_brake_state logs for thesis plots.")
    ap.add_argument("--telemetry-csv", type=str, required=False,
                    help="Path to telemetry CSV produced by --telemetry-csv")
    ap.add_argument("--scenario-csv", type=str, required=False,
                    help="Path to scenario CSV produced by --scenario-csv")
    ap.add_argument("--compare-csv", type=str, required=False,
                    help="Path to range comparison CSV produced by --range-est both --compare-csv")
    ap.add_argument("--out-dir", type=str, default="results_figs", help="Directory to save plots")
    ap.add_argument("--tag", type=str, default="run", help="Tag to prefix figure filenames")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.telemetry_csv:
        tele = load_telemetry(Path(args.telemetry_csv))
        print("Loaded telemetry samples:", tele["t"].size)
        plot_telemetry(tele, out_dir, args.tag)

    if args.scenario_csv:
        scen_rows = load_scenarios(Path(args.scenario_csv))
        print(f"\nLoaded scenario episodes: {len(scen_rows)}")
        summarize_scenarios(scen_rows)

    if args.compare_csv:
        cmp_rows = load_range_compare(Path(args.compare_csv))
        print(f"\nLoaded range-compare rows: {len(cmp_rows)}")
        analyze_range_compare(cmp_rows, out_dir, args.tag)

    print(f"\nFigures (if any) saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
