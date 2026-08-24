"""
research/phase13/prophet_tuning.py — Phase 13 Step 1: Prophet hyperparameter tuning.
Trains Prophet with each config in PROPHET_TUNE_GRID, evaluates, picks best.
"""
import logging, pickle, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config.logger import get_logger
from config.settings import (
    PROCESSED_DIR, FORECAST_DATASET_FILENAME, PROPHET_TRAIN_FRACTION,
    PROPHET_TUNE_GRID, RESULTS_TUNING, RESULTS_MODELS,
)
from research.experiment_tracker import log_experiment

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logger = get_logger(__name__)
PLOTS_DIR = RESULTS_TUNING / "prophet_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _evaluate_config(cfg: dict, train_df, test_df) -> dict:
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    t0 = time.time()
    m = Prophet(
        growth="linear",
        seasonality_mode=cfg["seasonality_mode"],
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        seasonality_prior_scale=cfg["seasonality_prior_scale"],
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
        interval_width=0.95,
    )
    m.add_seasonality("daily", period=1, fourier_order=10, mode="additive")
    m.add_seasonality("weekly", period=7, fourier_order=5, mode="additive")
    m.fit(train_df)
    train_time = time.time() - t0

    total = len(train_df) + len(test_df) + 288
    future = pd.DataFrame({"ds": pd.date_range(train_df["ds"].min(), periods=total, freq="5min")})
    fc = m.predict(future)
    merged = pd.merge(test_df[["ds","y"]], fc[["ds","yhat"]], on="ds", how="inner")

    actual = merged["y"].values; pred = merged["yhat"].values
    mae  = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2   = float(r2_score(actual, pred))
    nz   = actual != 0
    mape = float(np.mean(np.abs((actual[nz]-pred[nz])/actual[nz]))*100) if nz.sum()>0 else float("nan")
    return {"mae":mae,"rmse":rmse,"mape":mape,"r2":r2,
            "train_time_s":round(train_time,2),"model":m,"config":cfg}


class ProphetTuner:

    def run(self):
        logger.info("=== Phase 13-1: Prophet Hyperparameter Tuning ===")
        df = pd.read_csv(PROCESSED_DIR / FORECAST_DATASET_FILENAME,
                         index_col="timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        n = len(df); split = int(n * PROPHET_TRAIN_FRACTION)
        prophet_df = pd.DataFrame({
            "ds": df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None),
            "y":  df["cpu_util_percent"].values,
        })
        train_df = prophet_df.iloc[:split]
        test_df  = prophet_df.iloc[split:]

        rows = []
        for i, cfg in enumerate(PROPHET_TUNE_GRID):
            logger.info("Config %d/%d: %s", i+1, len(PROPHET_TUNE_GRID), cfg)
            res = _evaluate_config(cfg, train_df, test_df)
            row = {**cfg, **{k:v for k,v in res.items() if k not in ("model","config")}}
            row["config_id"] = i + 1
            rows.append(row)
            log_experiment(phase="13-Prophet", model="Prophet",
                           dataset="Alibaba-v2018", hyperparameters=cfg,
                           training_time_s=res["train_time_s"],
                           mae=res["mae"], rmse=res["rmse"],
                           mape=res["mape"], r2=res["r2"],
                           notes=f"Grid search config {i+1}")

        results_df = pd.DataFrame(rows)
        results_df.to_csv(RESULTS_TUNING / "prophet_tuning_results.csv", index=False)

        best_idx = results_df["mae"].idxmin()
        best = results_df.iloc[best_idx]
        logger.info("Best config (MAE=%.4f): %s", best["mae"], PROPHET_TUNE_GRID[int(best["config_id"])-1])

        # Retrain best and save
        best_cfg = PROPHET_TUNE_GRID[int(best["config_id"])-1]
        best_res = _evaluate_config(best_cfg, train_df, test_df)
        model_path = RESULTS_MODELS / "best_prophet_model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(best_res["model"], fh)
        logger.info("Best Prophet model saved: %s", model_path)

        self._plot_comparison(results_df)
        self._write_report(results_df, best)
        return results_df

    def _plot_comparison(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = range(len(df))
        for ax, metric, color in zip(axes, ["mae","rmse","mape"],
                                      ["#e74c3c","#3498db","#2ecc71"]):
            bars = ax.bar(x, df[metric], color=color, edgecolor="white", alpha=0.85)
            ax.set_title(f"Prophet Tuning — {metric.upper()}")
            ax.set_xlabel("Config ID"); ax.set_ylabel(metric.upper())
            ax.set_xticks(x); ax.set_xticklabels([f"C{i+1}" for i in x])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            best = df[metric].idxmin()
            bars[best].set_edgecolor("black"); bars[best].set_linewidth(2)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "prophet_tuning_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _write_report(self, df, best):
        rows = "\n".join(
            f"| C{r['config_id']} | {r['changepoint_prior_scale']} | "
            f"{r['seasonality_prior_scale']} | {r['seasonality_mode']} | "
            f"{r['mae']:.4f} | {r['rmse']:.4f} | {r['mape']:.4f} | {r['r2']:.4f} |"
            for _, r in df.iterrows()
        )
        report = f"""# Prophet Hyperparameter Tuning Report

**Phase:** 13 — Step 1  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  
**Grid size:** {len(df)} configurations

## Grid Results

| Config | CP Scale | Season Scale | Mode | MAE | RMSE | MAPE | R² |
|--------|----------|--------------|------|-----|------|------|-----|
{rows}

## Best Configuration
- **Config:** C{int(best['config_id'])}
- **changepoint_prior_scale:** {best['changepoint_prior_scale']}
- **seasonality_prior_scale:** {best['seasonality_prior_scale']}
- **seasonality_mode:** {best['seasonality_mode']}
- **MAE:** {best['mae']:.4f} %CPU
- **RMSE:** {best['rmse']:.4f} %CPU
- **MAPE:** {best['mape']:.4f}%
- **R²:** {best['r2']:.4f}

## Recommendation
Use `changepoint_prior_scale={best['changepoint_prior_scale']}`,
`seasonality_prior_scale={best['seasonality_prior_scale']}`,
`seasonality_mode="{best['seasonality_mode']}"` for the final Prophet model.
Best model saved to `results/models/best_prophet_model.pkl`.
"""
        (RESULTS_TUNING / "prophet_tuning_report.md").write_text(report)
        logger.info("prophet_tuning_report.md written.")
