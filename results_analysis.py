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
    """Load --range-est both compare CSV: pinhole vs depth (+stereo) distances.

    Expected columns (latest format):
      t, cls, x, y, w, h,
      s_pinhole_m, s_depth_m, abs_diff_m,
      s_stereo_m, ego_v_mps, mu, sigma_depth

    Falls back gracefully if s_stereo_m is missing.
    """
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["s_pinhole_m"] = float(r["s_pinhole_m"])
                r["s_depth_m"] = float(r["s_depth_m"])
                r["abs_diff_m"] = float(r["abs_diff_m"])
                if "s_stereo_m" in r and r["s_stereo_m"] not in ("", "None"):
                    r["s_stereo_m"] = float(r["s_stereo_m"])
                else:
                    r["s_stereo_m"] = float("nan")
            except Exception:
                continue
            rows.append(r)
    return rows


def load_stereo_compare(path: Path):
    """Load stereo vs depth comparison CSV produced by --stereo-compare-csv.

    Expected columns: t, cls, x, y, w, h,
    s_stereo_m, s_depth_m, err_stereo_depth_m, s_pinhole_m (optional), ego_v_mps, mu
    """
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["s_stereo_m"] = float(r["s_stereo_m"])
            except Exception:
                r["s_stereo_m"] = float("nan")
            try:
                r["s_depth_m"] = float(r["s_depth_m"])
            except Exception:
                r["s_depth_m"] = float("nan")
            err_val = r.get("err_stereo_depth_m", "")
            try:
                if err_val in ("", "None", None):
                    if np.isfinite(r["s_stereo_m"]) and np.isfinite(r["s_depth_m"]):
                        r["err_stereo_depth_m"] = r["s_stereo_m"] - r["s_depth_m"]
                    else:
                        r["err_stereo_depth_m"] = float("nan")
                else:
                    r["err_stereo_depth_m"] = float(err_val)
            except Exception:
                r["err_stereo_depth_m"] = float("nan")
            pin_val = r.get("s_pinhole_m", None)
            try:
                if pin_val in ("", "None", None):
                    r["s_pinhole_m"] = float("nan")
                else:
                    r["s_pinhole_m"] = float(pin_val)
            except Exception:
                r["s_pinhole_m"] = float("nan")
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
    # Pinhole vs depth
    e_p = np.array([r["s_pinhole_m"] - r["s_depth_m"] for r in rows])
    # Stereo vs depth (may contain NaNs if stereo not available)
    e_s = np.array([r["s_stereo_m"] - r["s_depth_m"] for r in rows])

    print("\n=== Range comparison summary ===")
    print(f"  total detections: {len(rows)}")
    print("\n  Overall errors vs depth:")
    print(f"    pinhole: MAE={np.nanmean(np.abs(e_p)):.3f} m, RMSE={np.sqrt(np.nanmean(e_p**2)):.3f} m")
    if np.isfinite(e_s).any():
        print(f"    stereo : MAE={np.nanmean(np.abs(e_s)):.3f} m, RMSE={np.sqrt(np.nanmean(e_s**2)):.3f} m")
    else:
        print("    stereo : no finite samples (stereo distances missing)")

    # Error vs distance bins
    bins = [0, 10, 20, 30, 40, 60, 80]
    bin_indices = np.digitize(dists, bins)
    print("\n  Error by range bin (pinhole / stereo vs depth):")
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if not mask.any():
            continue
        mae_p = np.mean(np.abs(e_p[mask]))
        rmse_p = np.sqrt(np.mean(e_p[mask] ** 2))
        if np.isfinite(e_s[mask]).any():
            mae_s = np.nanmean(np.abs(e_s[mask]))
            rmse_s = np.sqrt(np.nanmean(e_s[mask] ** 2))
            extra = f" | stereo: MAE={mae_s:.3f}, RMSE={rmse_s:.3f}"
        else:
            extra = " | stereo: n/a"
        print(f"    {bins[i-1]:2d}-{bins[i]:2d} m: pinhole MAE={mae_p:.3f} m, RMSE={rmse_p:.3f} m{extra}, N={mask.sum()}")

    # Per-class MAE
    print("\n  Error by class (pinhole / stereo vs depth):")
    for cls in cls_names:
        mask = [r["cls"] == cls for r in rows]
        if not any(mask):
            continue
        ep = np.array([e_p[i] for i, m in enumerate(mask) if m])
        es = np.array([e_s[i] for i, m in enumerate(mask) if m and np.isfinite(e_s[i])])
        mae_p = np.mean(np.abs(ep)) if ep.size else float("nan")
        if es.size:
            mae_s = np.mean(np.abs(es))
            extra = f" | stereo MAE={mae_s:.3f} m (N={es.size})"
        else:
            extra = " | stereo: n/a"
        print(f"    {cls:15s}: pinhole MAE={mae_p:.3f} m (N={ep.size}){extra}")

    # Scatter plots: pinhole vs depth and stereo vs depth
    plt.figure(figsize=(6, 5))
    plt.scatter(dists, e_p, s=4, alpha=0.3, label="pinhole-depth")
    if np.isfinite(e_s).any():
        plt.scatter(dists, e_s, s=4, alpha=0.3, label="stereo-depth")
    plt.axhline(0.0, color="k", linewidth=0.5)
    plt.xlabel("depth distance s_depth [m]")
    plt.ylabel("error s_method - s_depth [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_range_error_scatter.png", dpi=200)
    plt.close()


def analyze_stereo_compare(rows, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("\nNo stereo-compare rows to analyze.")
        return

    cls_names = sorted(set(r["cls"] for r in rows))
    dists = np.array([r["s_depth_m"] for r in rows], dtype=float)
    e_s = np.array([r["s_stereo_m"] - r["s_depth_m"] for r in rows], dtype=float)
    pinhole_vals = np.array([r.get("s_pinhole_m", float("nan")) for r in rows], dtype=float)
    e_p = pinhole_vals - dists

    valid_s = np.isfinite(e_s) & np.isfinite(dists)
    valid_p = np.isfinite(e_p) & np.isfinite(dists)

    print("\n=== Stereo vs depth comparison summary ===")
    print(f"  total detections: {len(rows)}")
    if valid_s.any():
        print(f"  stereo overall MAE:  {np.mean(np.abs(e_s[valid_s])):.3f} m")
        print(f"  stereo overall RMSE: {np.sqrt(np.mean(e_s[valid_s]**2)):.3f} m")
    else:
        print("  stereo overall MAE/RMSE: n/a (no finite stereo-depth pairs)")
    if valid_p.any():
        print(f"  pinhole overall MAE: {np.mean(np.abs(e_p[valid_p])):.3f} m")
        print(f"  pinhole overall RMSE: {np.sqrt(np.mean(e_p[valid_p]**2)):.3f} m")
    else:
        print("  pinhole overall MAE/RMSE: n/a (no finite pinhole-depth pairs)")

    bins = [0, 10, 20, 30, 40, 60, 80]
    bin_indices = np.digitize(dists, bins)
    print("\n  Error by range bin (stereo & pinhole vs depth):")
    for i in range(1, len(bins)):
        mask_s = (bin_indices == i) & valid_s
        mask_p = (bin_indices == i) & valid_p
        if not mask_s.any() and not mask_p.any():
            continue
        parts = [f"    {bins[i-1]:2d}-{bins[i]:2d} m:"]
        if mask_s.any():
            mae_s = np.mean(np.abs(e_s[mask_s])); rmse_s = np.sqrt(np.mean(e_s[mask_s] ** 2))
            parts.append(f"stereo MAE={mae_s:.3f} m RMSE={rmse_s:.3f} m (N={mask_s.sum()})")
        else:
            parts.append("stereo n/a")
        if mask_p.any():
            mae_p = np.mean(np.abs(e_p[mask_p])); rmse_p = np.sqrt(np.mean(e_p[mask_p] ** 2))
            parts.append(f"pinhole MAE={mae_p:.3f} m RMSE={rmse_p:.3f} m (N={mask_p.sum()})")
        else:
            parts.append("pinhole n/a")
        print(" ".join(parts))

    print("\n  Error by class (stereo & pinhole vs depth):")
    for cls in cls_names:
        mask_cls = np.array([r["cls"] == cls for r in rows])
        mask_s = mask_cls & valid_s
        mask_p = mask_cls & valid_p
        if not mask_s.any() and not mask_p.any():
            continue
        if mask_s.any():
            es = e_s[mask_s]; msg_s = f"stereo MAE={np.mean(np.abs(es)):.3f} m (N={es.size})"
        else:
            msg_s = "stereo n/a"
        if mask_p.any():
            ep = e_p[mask_p]; msg_p = f"pinhole MAE={np.mean(np.abs(ep)):.3f} m (N={ep.size})"
        else:
            msg_p = "pinhole n/a"
        print(f"    {cls:15s}: {msg_s} | {msg_p}")

    # Scatter plot error vs distance
    plt.figure(figsize=(6, 5))
    added = False
    if valid_p.any():
        plt.scatter(dists[valid_p], e_p[valid_p], s=4, alpha=0.3, label="pinhole-depth")
        added = True
    if valid_s.any():
        plt.scatter(dists[valid_s], e_s[valid_s], s=4, alpha=0.3, label="stereo-depth")
        added = True
    plt.axhline(0.0, color="k", linewidth=0.5)
    plt.xlabel("depth distance s_depth [m]")
    plt.ylabel("range error (method - depth) [m]")
    if added:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_stereo_range_error_scatter.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Quick analysis of dynamic_brake_state logs for thesis plots.")
    ap.add_argument("--telemetry-csv", type=str, required=False,
                    help="Path to telemetry CSV produced by --telemetry-csv")
    ap.add_argument("--scenario-csv", type=str, required=False,
                    help="Path to scenario CSV produced by --scenario-csv")
    ap.add_argument("--compare-csv", type=str, required=False,
                    help="Path to range comparison CSV produced by --range-est both --compare-csv")
    ap.add_argument("--stereo-compare-csv", type=str, required=False,
                    help="Path to stereo vs depth comparison CSV produced by --range-est stereo --stereo-compare-csv")
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

    if args.stereo_compare_csv:
        s_rows = load_stereo_compare(Path(args.stereo_compare_csv))
        print(f"\nLoaded stereo-compare rows: {len(s_rows)}")
        analyze_stereo_compare(s_rows, out_dir, args.tag + "_stereo")

    print(f"\nFigures (if any) saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
