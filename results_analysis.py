import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_DT_MS = 20.0  # controller loop target period (50 Hz)


def load_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "t", "v_mps", "tau_dyn", "D_safety_dyn", "sigma_depth", "a_des", "brake",
        "lambda_max", "abs_factor", "mu_est", "loop_ms", "loop_ms_max", "detect_ms",
        "latency_ms", "a_meas", "x_rel_m", "range_est_m", "gate_hit", "false_stop_flag",
        "ttc_s", "sensor_to_control_ms", "control_to_act_ms", "sensor_to_act_ms"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_scenarios(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "mu", "v_init_mps", "s_init_m", "s_min_m", "s_init_gt_m", "s_min_gt_m",
        "t_to_stop_s", "range_margin_m", "tts_margin_s", "max_lambda", "mean_abs_factor"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bool_cols = ["stopped", "collision", "false_stop"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


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


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy()
    return np.full(len(df), np.nan)


def _mu_regime_from_value(mu: float) -> str:
    if pd.isna(mu):
        return "unknown"
    if mu > 0.8:
        return "high"
    if mu > 0.4:
        return "medium"
    return "low"


def summarize_scenarios(df: pd.DataFrame, out_dir: Path, tag: str):
    if df.empty:
        print("\nNo scenario episodes to summarize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if "range_margin_m" not in df.columns:
        if {"s_init_gt_m", "s_min_gt_m"}.issubset(df.columns):
            df["range_margin_m"] = df["s_init_gt_m"] - df["s_min_gt_m"]
        elif {"s_init_m", "s_min_m"}.issubset(df.columns):
            df["range_margin_m"] = df["s_init_m"] - df["s_min_m"]

    if "mu_regime" not in df.columns:
        if "mu" in df.columns:
            df["mu_regime"] = df["mu"].apply(_mu_regime_from_value)
        else:
            df["mu_regime"] = "unknown"

    print("\n=== Scenario summary ===")
    for scen, sub in df.groupby("scenario"):
        print(f"\nScenario: {scen} (N={len(sub)})")
        if "v_init_mps" in sub:
            print(f"  mean v_init: {sub['v_init_mps'].mean():.2f} m/s")
        if "s_init_m" in sub:
            print(f"  mean s_init: {sub['s_init_m'].mean():.2f} m")
        if "s_min_m" in sub:
            print(f"  mean s_min:  {sub['s_min_m'].mean():.2f} m")
        if "t_to_stop_s" in sub and sub.get("stopped", pd.Series(dtype=bool)).any():
            stopped = sub["stopped"].astype(bool)
            print(f"  mean t_stop (stops): {sub.loc[stopped, 't_to_stop_s'].mean():.2f} s")
        if "collision" in sub:
            print(f"  collision rate: {sub['collision'].mean() * 100:.1f}%")

    by_mu = df.groupby("mu_regime")
    table_mu = by_mu.agg({
        "range_margin_m": ["mean", "std", "min"],
        "collision": "mean",
        "false_stop": "mean",
        "t_to_stop_s": "mean"
    }).rename(columns={"mean": "mean", "std": "std", "min": "min", "collision": "collision", "false_stop": "false_stop"})
    print("\n=== Range & collision stats by μ regime ===")
    print(table_mu)
    table_mu.to_csv(out_dir / f"{tag}_mu_regime_summary.csv")

    if "t_to_stop_s" in df.columns and "v_init_mps" in df.columns:
        bins = [0, 10, 15, 20, 25, 40]
        df["v_bin"] = pd.cut(df["v_init_mps"], bins, right=False)
        pivot = df.pivot_table(index="v_bin", columns="mu_regime", values="t_to_stop_s", aggfunc=["mean", "std"])
        print("\n=== Time to stop by speed bin and μ regime ===")
        print(pivot)
        pivot.to_csv(out_dir / f"{tag}_tts_table.csv")

    if "range_margin_m" in df.columns:
        plot_margin_cdf(df["range_margin_m"].to_numpy(), out_dir, tag)

    if "mean_abs_factor" in df.columns:
        duty = (1.0 - df["mean_abs_factor"]).clip(lower=0.0)
        print("\nMean ABS duty per regime (lower is less intervention):")
        print(df.assign(abs_duty=duty).groupby("mu_regime")["abs_duty"].mean())


def plot_margin_cdf(margins: np.ndarray, out_dir: Path, tag: str):
    finite = margins[np.isfinite(margins)]
    if finite.size == 0:
        return
    finite.sort()
    y = np.linspace(0, 1, finite.size, endpoint=False)
    plt.figure(figsize=(7, 4))
    plt.plot(finite, y)
    plt.xlabel("range margin [m]")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{tag}_range_margin_cdf.png", dpi=220)
    plt.close()


def summarize_false_stops(scen: pd.DataFrame, tele: Optional[pd.DataFrame]):
    print("\n=== False stop diagnostics ===")
    if "false_stop" in scen.columns:
        rate = scen["false_stop"].mean() * 100.0
        print(f"  Scenario-level false stops: {rate:.2f}% of episodes")
        print(scen.groupby("mu_regime")["false_stop"].mean().rename("false_stop_rate"))
    else:
        print("  Scenario CSV lacks false_stop column.")
    if tele is not None and "false_stop_flag" in tele.columns:
        flag = tele["false_stop_flag"].dropna().to_numpy()
        if flag.size:
            tele_rate = 100.0 * np.mean(flag > 0.5)
            print(f"  Frame-level false_stop_flag active {tele_rate:.2f}% of the time")
        else:
            print("  false_stop_flag column has no samples.")
    else:
        print("  Telemetry missing false_stop_flag column.")


def plot_telemetry(tele: pd.DataFrame, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if tele.empty:
        print("\nNo telemetry samples to plot.")
        return

    t = _col(tele, "t")
    if not np.isfinite(t).any():
        print("\nTelemetry lacks time samples; skipping plots.")
        return

    v = _col(tele, "v_mps")
    D_safe = _col(tele, "D_safety_dyn")
    tau = _col(tele, "tau_dyn")
    brake = _col(tele, "brake")
    x_rel = _col(tele, "x_rel_m")
    range_est = _col(tele, "range_est_m")
    lam = _col(tele, "lambda_max")
    abs_factor = _col(tele, "abs_factor")
    mu_est = _col(tele, "mu_est")
    mu_regime = tele["mu_regime"].astype(str).fillna("") if "mu_regime" in tele.columns else None

    scale = max(1.0, np.nanmax(v))

    plt.figure(figsize=(9, 4))
    plt.plot(t, v, label="speed [m/s]")
    if np.isfinite(D_safe).any():
        plt.plot(t, D_safe, label="D_safety_dyn [m]")
    if np.isfinite(x_rel).any():
        plt.plot(t, x_rel, label="x_rel_gt [m]")
    if np.isfinite(range_est).any():
        plt.plot(t, range_est, label="range_est [m]")
    plt.xlabel("time [s]")
    plt.ylabel("meters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_speed_gap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(t, brake * scale, label="brake (scaled)")
    if np.isfinite(lam).any():
        plt.plot(t, lam * scale, label="lambda_max (scaled)")
    if np.isfinite(abs_factor).any():
        plt.plot(t, abs_factor * scale, label="abs_factor (scaled)")
    plt.xlabel("time [s]")
    plt.ylabel("scaled units")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_brake_slip.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4))
    if np.isfinite(mu_est).any():
        plt.plot(t, mu_est, label="mu_est")
    if mu_regime is not None:
        mapping = {"low": 0.2, "medium": 0.5, "high": 0.9, "fixed": 0.5, "off": np.nan}
        regime_vals = mu_regime.map(lambda s: mapping.get(s.lower(), np.nan)).to_numpy()
        plt.plot(t, regime_vals, label="mu_regime (encoded)")
    plt.xlabel("time [s]")
    plt.ylabel("μ estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_mu_est.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 3.5))
    if np.isfinite(tau).any():
        plt.plot(t, tau, label="tau_dyn")
    plt.xlabel("time [s]")
    plt.ylabel("tau [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_tau_dyn.png", dpi=220)
    plt.close()


def summarize_timing(tele: pd.DataFrame):
    if "loop_ms" not in tele.columns:
        print("\nTelemetry file lacks loop_ms; timing summary skipped.")
        return
    loop = tele["loop_ms"].dropna().to_numpy()
    if loop.size == 0:
        print("\nNo loop_ms samples available.")
        return
    detect = tele["detect_ms"].dropna().to_numpy() if "detect_ms" in tele.columns else np.array([])
    latency = tele["latency_ms"].dropna().to_numpy() if "latency_ms" in tele.columns else np.array([])
    miss_pct = 100.0 * np.mean(loop > TARGET_DT_MS)
    print("\n=== Real-time loop timing ===")
    print(f"  loop_ms: mean={np.mean(loop):.2f}  p95={np.percentile(loop,95):.2f}  max={np.max(loop):.2f}  deadline_miss%={miss_pct:.2f}")
    if detect.size:
        print(f"  detect_ms: mean={np.mean(detect):.2f}  p95={np.percentile(detect,95):.2f}  max={np.max(detect):.2f}")
    if latency.size:
        print(f"  latency_ms: mean={np.mean(latency):.2f}  p95={np.percentile(latency,95):.2f}  max={np.max(latency):.2f}")


def plot_latency_hist(tele: pd.DataFrame, out_dir: Path, tag: str):
    if "loop_ms" not in tele.columns:
        return
    loop = tele["loop_ms"].dropna().to_numpy()
    if loop.size == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(loop, bins=40, color="#3a7", alpha=0.8)
    plt.axvline(TARGET_DT_MS, color="r", linestyle="--", label=f"dt target {TARGET_DT_MS:.0f} ms")
    plt.xlabel("loop_ms")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_loop_ms_hist.png", dpi=220)
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

    tele_df: Optional[pd.DataFrame] = None
    if args.telemetry_csv:
        tele_df = load_telemetry(Path(args.telemetry_csv))
        print("Loaded telemetry samples:", len(tele_df))
        plot_telemetry(tele_df, out_dir, args.tag)
        summarize_timing(tele_df)
        plot_latency_hist(tele_df, out_dir, args.tag)

    if args.scenario_csv:
        scen_df = load_scenarios(Path(args.scenario_csv))
        print(f"\nLoaded scenario episodes: {len(scen_df)}")
        summarize_scenarios(scen_df, out_dir, args.tag)
        summarize_false_stops(scen_df, tele_df)

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
