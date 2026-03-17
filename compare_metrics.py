"""
compare_metrics.py
==================
Loads saved metrics JSONs produced by train_pipeline.py and generates
comparison tables + plots across all 4 approaches.
 
Expected folder layout (produced by train_pipeline.py)
-------------------------------------------------------
results/
  efficientnet/
    nctd/metrics/   ← NCTD_efficientnet_<dataset>_metrics.json
    ours/metrics/   ← ours_efficientnet_<dataset>_metrics.json
  cnn/
    nctd/metrics/   ← NCTD_nctd_cnn_<dataset>_metrics.json
    ours/metrics/   ← ours_nctd_cnn_<dataset>_metrics.json
 
All outputs go to results/comparison/
  summary_table.csv         ← mean ± std per approach × metric
  summary_heatmap.png
  rank_table.csv            ← win counts + average rank
  win_count.png
  per_dataset_<metric>.csv  ← one row per dataset, one col per approach
  boxplot_<metric>.png
  per_dataset_<metric>.png  ← grouped bar charts (key metrics)
  radar_chart.png
  logs/compare_<ts>.log
 
Usage
-----
  python compare_metrics.py
  python compare_metrics.py --results_dir /path/to/results
"""
 
import os
import sys
import json
import logging
import argparse
import datetime
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
 
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_RESULTS_DIR = "results"
_RUN_TS  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
 
APPROACHES = {
    "EfficientNet+NCTD": ("efficientnet", "nctd"),
    "EfficientNet+Ours": ("efficientnet", "ours"),
    "CNN+NCTD":          ("cnn",          "nctd"),
    "CNN+Ours":          ("cnn",          "ours"),
}
 
SCALAR_METRICS = [
    "accuracy",
    "f1",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "balanced_accuracy",
    "roc_auc",
    "mcc",
    "cohen_kappa",
]
 
METRIC_LABELS = {
    "accuracy":          "Accuracy",
    "f1":                "F1 (binary)",
    "f1_macro":          "Macro F1",
    "f1_weighted":       "Weighted F1",
    "precision":         "Precision",
    "recall":            "Recall",
    "balanced_accuracy": "Balanced Accuracy",
    "roc_auc":           "ROC-AUC",
    "mcc":               "MCC",
    "cohen_kappa":       "Cohen's Kappa",
}
 
PALETTE = {
    "EfficientNet+NCTD": "#4C72B0",
    "EfficientNet+Ours": "#DD8452",
    "CNN+NCTD":          "#55A868",
    "CNN+Ours":          "#C44E52",
}
 
# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING  (same style as train_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════
def _make_logger(results_dir: str) -> logging.Logger:
    log_dir  = os.path.join(results_dir, "comparison", "logs")
    log_file = os.path.join(log_dir, f"compare_{_RUN_TS}.log")
    os.makedirs(log_dir, exist_ok=True)
 
    logger = logging.getLogger("compare")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
 
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
 
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
 
    return logger
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if f == f else float("nan")   # filter inf → nan
    except (TypeError, ValueError):
        return float("nan")
 
 
def _dataset_stem(json_fname: str) -> str:
    """Strip combo prefix and '_metrics.json' suffix → clean dataset id."""
    stem = json_fname.replace("_metrics.json", "")
    for prefix in [
        "NCTD_efficientnet_", "NCTD_nctd_cnn_",
        "ours_efficientnet_", "ours_nctd_cnn_",
    ]:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def _load_approach(
    results_dir: str, model_key: str, method_key: str, logger: logging.Logger
) -> dict:
    """Returns {dataset_id: {metric: float}} for one approach."""
    metrics_dir = os.path.join(results_dir, model_key, method_key, "metrics")
    data = {}
 
    if not os.path.isdir(metrics_dir):
        logger.warning(f"metrics dir not found: {metrics_dir}")
        return data
 
    for fname in sorted(f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")):
        fpath = os.path.join(metrics_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"could not read {fpath}: {exc}")
            continue
        data[_dataset_stem(fname)] = {m: _safe_float(raw.get(m)) for m in SCALAR_METRICS}
 
    logger.debug(f"loaded {len(data)} datasets from {metrics_dir}")
    return data
 
 
def _build_long_df(results_dir: str, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for label, (mk, mth) in APPROACHES.items():
        logger.info(f"  Loading  {label:<22}  (results/{mk}/{mth}/metrics/)")
        for ds, row in _load_approach(results_dir, mk, mth, logger).items():
            for metric, value in row.items():
                rows.append({"dataset": ds, "approach": label, "metric": metric, "value": value})
    df = pd.DataFrame(rows)
    logger.info(
        f"  Total: {len(df)} rows | "
        f"{df['dataset'].nunique()} datasets | "
        f"{df['approach'].nunique()} approaches"
    )
    return df
 
 
def _wide(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = long_df[long_df["metric"] == metric]
    w   = sub.pivot_table(index="dataset", columns="approach", values="value")
    return w[[k for k in APPROACHES if k in w.columns]]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════
def _summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for approach in APPROACHES:
        row = {"Approach": approach}
        sub = long_df[long_df["approach"] == approach]
        for m in SCALAR_METRICS:
            vals  = sub[sub["metric"] == m]["value"].dropna().values
            label = METRIC_LABELS[m]
            if len(vals) == 0:
                row[label]        = "N/A"
                row[f"_mean_{m}"] = float("nan")
            else:
                mean = float(np.mean(vals));  std = float(np.std(vals))
                row[label]        = f"{mean:.4f} ± {std:.4f}"
                row[f"_mean_{m}"] = mean
        records.append(row)
    return pd.DataFrame(records).set_index("Approach")
 
 
def _rank_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in SCALAR_METRICS:
        ranked = _wide(long_df, metric).rank(axis=1, ascending=False, method="average")
        for approach in ranked.columns:
            for ds, rv in ranked[approach].items():
                rows.append({"approach": approach, "metric": metric,
                              "dataset": ds, "rank": rv, "is_best": int(rv == 1.0)})
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("approach")
        .agg(avg_rank=("rank", "mean"), wins=("is_best", "sum"), total=("rank", "count"))
        .reset_index().sort_values("avg_rank")
    )
    summary["win_pct"] = (summary["wins"] / summary["total"] * 100).round(2)
    return summary
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_heatmap(summary_df: pd.DataFrame, out: str, logger: logging.Logger):
    numeric = {
        METRIC_LABELS[m]: summary_df[f"_mean_{m}"].astype(float).values
        for m in SCALAR_METRICS if f"_mean_{m}" in summary_df.columns
    }
    ndf = pd.DataFrame(numeric, index=summary_df.index)
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.heatmap(ndf, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Mean Score per Approach × Metric", fontsize=13, pad=12)
    ax.set_ylabel("");  ax.tick_params(axis="x", rotation=30)
    plt.tight_layout();  plt.savefig(out, dpi=150);  plt.close()
    logger.info(f"  Saved: {out}")
 
 
def _plot_boxplots(long_df: pd.DataFrame, metric: str, out: str, logger: logging.Logger):
    sub   = long_df[long_df["metric"] == metric].dropna(subset=["value"])
    order = [o for o in APPROACHES if o in sub["approach"].unique()]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=sub, x="approach", y="value", order=order,
                palette=PALETTE, width=0.5, ax=ax,
                flierprops=dict(marker="o", markersize=4, alpha=0.5))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} – Distribution across Datasets", fontsize=12)
    ax.set_xlabel("");  ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.xticks(rotation=15, ha="right");  plt.tight_layout()
    plt.savefig(out, dpi=150);  plt.close()
    logger.debug(f"  Saved: {out}")
 
 
def _plot_per_dataset_bar(
    long_df: pd.DataFrame, metric: str, out: str, logger: logging.Logger
):
    sub      = long_df[long_df["metric"] == metric].dropna(subset=["value"])
    order    = [o for o in APPROACHES if o in sub["approach"].unique()]
    datasets = sorted(sub["dataset"].unique())
    n_ds, n_app = len(datasets), len(order)
    bar_w = 0.8 / n_app;  x = np.arange(n_ds)
 
    fig, ax = plt.subplots(figsize=(max(16, n_ds * 0.65), 5))
    for i, approach in enumerate(order):
        heights = [
            sub[(sub["dataset"] == d) & (sub["approach"] == approach)]["value"].values
            for d in datasets
        ]
        heights = [v[0] if len(v) > 0 else float("nan") for v in heights]
        ax.bar(x + i * bar_w, heights, width=bar_w,
               label=approach, color=PALETTE[approach], alpha=0.85)
 
    ax.set_xticks(x + bar_w * (n_app - 1) / 2)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} – Per Dataset Comparison", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout();  plt.savefig(out, dpi=150);  plt.close()
    logger.debug(f"  Saved: {out}")
 
 
def _plot_radar(summary_df: pd.DataFrame, out: str, logger: logging.Logger):
    selected = ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]
    labels   = [METRIC_LABELS[m] for m in selected]
    N        = len(selected)
    angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
 
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for approach in APPROACHES:
        if approach not in summary_df.index:
            continue
        vals = [_safe_float(summary_df.loc[approach, f"_mean_{m}"]) for m in selected]
        vals = [v if v == v else 0.0 for v in vals] + [vals[0] if vals[0] == vals[0] else 0.0]
        ax.plot(angles, vals, linewidth=2, color=PALETTE[approach], label=approach)
        ax.fill(angles, vals, alpha=0.10, color=PALETTE[approach])
 
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_title("Radar – Mean Scores (5 key metrics)", fontsize=11, pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), fontsize=8)
    plt.tight_layout();  plt.savefig(out, dpi=150, bbox_inches="tight");  plt.close()
    logger.info(f"  Saved: {out}")
 
 
def _plot_win_bar(rank_df: pd.DataFrame, out: str, logger: logging.Logger):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = [PALETTE.get(a, "#888888") for a in rank_df["approach"]]
    ax.barh(rank_df["approach"], rank_df["wins"], color=colors, alpha=0.85)
    ax.set_xlabel("Number of wins (ranked #1 across datasets × metrics)")
    ax.set_title("Win Count per Approach")
    for i, (w, p) in enumerate(zip(rank_df["wins"], rank_df["win_pct"])):
        ax.text(w + 0.2, i, f"{w}  ({p:.1f}%)", va="center", fontsize=9)
    plt.tight_layout();  plt.savefig(out, dpi=150);  plt.close()
    logger.info(f"  Saved: {out}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(results_dir: str):
    out_dir = os.path.join(results_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    logger  = _make_logger(results_dir)
 
    logger.info(f"{'─'*60}")
    logger.info(f"  compare_metrics.py  |  run {_RUN_TS}")
    logger.info(f"  results → {os.path.abspath(results_dir)}")
    logger.info(f"  output  → {os.path.abspath(out_dir)}")
    logger.info(f"{'─'*60}")
 
    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading metrics …")
    long_df = _build_long_df(results_dir, logger)
 
    if long_df.empty:
        logger.error("No metrics found. Run train_pipeline.py for at least one combo first.")
        return
 
    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("Building summary table …")
    summary_df   = _summary_table(long_df)
    display_cols = [METRIC_LABELS[m] for m in SCALAR_METRICS if METRIC_LABELS[m] in summary_df.columns]
    display_df   = summary_df[display_cols]
 
    logger.info("\n── Summary (mean ± std across all datasets) ──")
    for line in display_df.to_string().splitlines():
        logger.info("  " + line)
 
    csv_path = os.path.join(out_dir, "summary_table.csv")
    display_df.to_csv(csv_path)
    logger.info(f"\nSaved: {csv_path}")
 
    # ── Per-dataset wide tables ────────────────────────────────────────────────
    logger.info("Saving per-dataset tables …")
    for m in SCALAR_METRICS:
        wide = _wide(long_df, m)
        if wide.empty:
            continue
        wide_out = pd.concat([
            wide,
            wide.mean(numeric_only=True).rename("MEAN").to_frame().T,
            wide.std(numeric_only=True).rename("STD").to_frame().T,
        ])
        p = os.path.join(out_dir, f"per_dataset_{m}.csv")
        wide_out.round(5).to_csv(p)
        logger.debug(f"  Saved: {p}")
    logger.info(f"  Per-dataset CSVs → {out_dir}/")
 
    # ── Rank table ─────────────────────────────────────────────────────────────
    logger.info("Computing rank / win table …")
    rank_df = _rank_table(long_df)
    logger.info("\n── Rank Table ──")
    for line in rank_df.to_string(index=False).splitlines():
        logger.info("  " + line)
    rank_df.to_csv(os.path.join(out_dir, "rank_table.csv"), index=False)
 
    # ── Plots ──────────────────────────────────────────────────────────────────
    logger.info("Generating plots …")
    _plot_heatmap(summary_df, os.path.join(out_dir, "summary_heatmap.png"), logger)
    _plot_radar(summary_df,   os.path.join(out_dir, "radar_chart.png"),     logger)
    _plot_win_bar(rank_df,    os.path.join(out_dir, "win_count.png"),        logger)
 
    logger.info("  Box-plots …")
    for m in SCALAR_METRICS:
        _plot_boxplots(long_df, m, os.path.join(out_dir, f"boxplot_{m}.png"), logger)
 
    logger.info("  Per-dataset bar charts …")
    for m in ["accuracy", "f1", "roc_auc", "mcc", "precision", "recall"]:
        _plot_per_dataset_bar(long_df, m, os.path.join(out_dir, f"per_dataset_{m}.png"), logger)
 
    # ── Final comparison ───────────────────────────────────────────────────────
    logger.info(f"{'─'*60}")
    logger.info("  Final comparison — Accuracy")
    logger.info(f"{'─'*60}")
    for approach in APPROACHES:
        vals = (
            long_df[(long_df["approach"] == approach) & (long_df["metric"] == "accuracy")]
            ["value"].dropna().values
        )
        if len(vals):
            logger.info(
                f"  {approach:<22}  {np.mean(vals):.4f} ± {np.std(vals):.4f}"
                f"  (n={len(vals)})"
            )
        else:
            logger.info(f"  {approach:<22}  no results found")
    logger.info(f"{'─'*60}")
    logger.info(f"All outputs → {os.path.abspath(out_dir)}")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare metrics across all 4 approaches",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR,
        help="Root results directory (default: results/)",
    )
    args = parser.parse_args()
    main(args.results_dir)