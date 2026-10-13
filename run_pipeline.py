# -*- coding: utf-8 -*-
"""
Credit Card Fraud Detection — Production Benchmark Pipeline (v2.1)
===================================================================

Author: Vinicius Dias Oliveira (UNIFESP - EPPEN)
Methodological revisions over v1 (monograph):
- Eliminates selection leakage: decision thresholds selected exclusively on validation.
- Enforces temporal integrity: strict out-of-time split (70/15/15) + cyclical 'hour' feature.
- Strict ablation study: baseline vs. class_weight vs. SMOTE inside imblearn.Pipeline.
- Evaluates ranking with PR-AUC (Average Precision) under extreme imbalance (0.17%).
- Cost-sensitive business decision: expected economic loss in R$ (FN = Amount, FP = cost_fp).
- Probability calibration: Platt scaling via CalibratedClassifierCV + Brier score.
- Statistical inference: multi-seed runs (algorithmic variance), bootstrap 95% CI (sample variance),
  and McNemar exact paired test between top models.
- Modern architectures: XGBoost, HistGradientBoosting, Isolation Forest, and Reconstruction Autoencoder.
- Explainable AI: SHAP TreeExplainer for transaction-level feature attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path": "data/creditcard.csv" if Path("data/creditcard.csv").exists() else "creditcard.csv",
    "output_dir": "outputs",
    "base_seed": 42,
    "n_seeds": 5,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "cost_fp": 10.0,
    "cost_fp_grid": [2.0, 10.0, 50.0],
    "n_bootstrap": 500,
    "sample_n": None,
    "run_shap": True,
    "run_xgb_tuning": True,
}

SUPERVISED_MODELS = [
    "Logistic Regression",
    "Linear SVM (calibrated)",
    "Decision Tree",
    "Gaussian NB",
    "Random Forest",
    "Hist Gradient Boosting",
    "XGBoost",
]
UNSUPERVISED_MODELS = ["Isolation Forest", "Autoencoder"]
ABLATION_STRATEGIES = ["baseline", "class_weight", "smote"]
DETERMINISTIC_MODELS = {
    "Logistic Regression",
    "Linear SVM (calibrated)",
    "Gaussian NB",
}


# ============================================================================
# 1. Data Ingestion & Feature Engineering
# ============================================================================

def load_data(path: str | Path) -> pd.DataFrame:
    """Load credit card transactions with optimized dtypes."""
    path = Path(path)
    if not path.exists():
        # Fallback search
        alt_paths = [
            Path("data") / path.name,
            Path(path.name),
            Path("creditcard.csv"),
            Path("data/creditcard.csv"),
            Path("data/creditcard.zip"),
            Path("creditcard.zip"),
        ]
        for p in alt_paths:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. Place creditcard.csv or creditcard.zip "
                "in 'data/' or repository root."
            )

    dtypes = {"Time": "float64", "Amount": "float64", "Class": "int8"}
    dtypes.update({f"V{i}": "float32" for i in range(1, 29)})
    df = pd.read_csv(path, dtype=dtypes)
    
    missing = ({"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical daily hour feature and log-transformed amount."""
    out = df.copy()
    out["hour"] = (out["Time"] // 3600) % 24
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Strict chronological out-of-time split (past -> train, present -> validation, future -> test)."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["hour", "log_amount"]
    
    parts = {}
    for name, sl in (("train", slice(0, i_tr)), ("val", slice(i_tr, i_va)), ("test", slice(i_va, n))):
        sub = df.iloc[sl]
        parts[name] = (
            sub[feature_cols].to_numpy(np.float64),
            sub["Class"].to_numpy(np.int8),
            sub["Amount"].to_numpy(np.float64),
        )
        y = parts[name][1]
        print(f"  {name:5s}: {len(y):>7,} obs | Frauds: {int(y.sum()):>4} ({y.mean()*100:.3f}%)")
    return parts


# ============================================================================
# 2. Model Factory & Anomaly Scorers
# ============================================================================

def make_supervised(name: str, strategy: str, seed: int, pos_ratio: float, config: Dict[str, Any]):
    """Constructs supervised models within sklearn/imblearn pipelines."""
    cw = "balanced" if strategy == "class_weight" else None

    if name == "Logistic Regression":
        est = LogisticRegression(max_iter=5000, class_weight=cw, solver="lbfgs", random_state=seed)
        scale = True
    elif name == "Linear SVM (calibrated)":
        base = LinearSVC(class_weight=cw, dual=False, max_iter=10000, random_state=seed)
        est = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        scale = True
    elif name == "Decision Tree":
        est = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight=cw, random_state=seed)
        scale = False
    elif name == "Gaussian NB":
        est = GaussianNB(priors=[0.5, 0.5] if strategy == "class_weight" else None)
        scale = False
    elif name == "Random Forest":
        est = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight=cw, n_jobs=-1, random_state=seed)
        scale = False
    elif name == "Hist Gradient Boosting":
        est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight=cw, random_state=seed)
        scale = False
    elif name == "XGBoost":
        est = XGBClassifier(
            n_estimators=config.get("_xgb_n_estimators", 400),
            learning_rate=config.get("_xgb_lr", 0.1),
            max_depth=config.get("_xgb_depth", 6),
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_ratio if strategy == "class_weight" else 1.0,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
        )
        scale = False
    else:
        raise ValueError(f"Unknown model: {name}")

    steps = []
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=5)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", est))
    return ImbPipeline(steps) if strategy == "smote" else SkPipeline(steps)


class IsoForestScorer:
    """Isolation Forest anomaly detector (unsupervised)."""
    def __init__(self, seed: int):
        self.model = IsolationForest(n_estimators=200, max_samples=0.5, contamination="auto", n_jobs=-1, random_state=seed)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class AutoencoderScorer:
    """Bottleneck Autoencoder (30->16->8->16->30) trained exclusively on legitimate transactions."""
    def __init__(self, seed: int):
        self.scaler = StandardScaler()
        self.ae = MLPRegressor(
            hidden_layer_sizes=(16, 8, 16),
            activation="relu",
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_leg = X[y == 0]
        Xs = self.scaler.fit_transform(X_leg)
        self.ae.fit(Xs, Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        rec = self.ae.predict(Xs)
        return np.mean((Xs - rec) ** 2, axis=1)


def minmax_to_proba(scores_fit: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Normalize anomaly scores to [0, 1] using validation bounds only."""
    lo, hi = scores_fit.min(), scores_fit.max()
    return np.clip((scores - lo) / (hi - lo + 1e-12), 0, 1)


# ============================================================================
# 3. Decision Theory & Economic Cost Functions
# ============================================================================

def threshold_max_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Finds F1-maximizing threshold on validation data."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, amount: np.ndarray, cost_fp: float) -> float:
    """
    Cost function in R$:
    - False Negative: unrecovered fraud cost = transaction Amount.
    - False Positive: friction & manual review cost = cost_fp.
    """
    fn = (y_true == 1) & (y_pred == 0)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(amount[fn].sum() + fp * cost_fp)


def threshold_min_cost(y_val: np.ndarray, p_val: np.ndarray, amount_val: np.ndarray, cost_fp: float) -> float:
    """Selects threshold on validation set minimizing total financial loss."""
    cand = np.unique(np.quantile(p_val, np.linspace(0, 1, 512)))
    costs = [expected_cost(y_val, (p_val >= t).astype(int), amount_val, cost_fp) for t in cand]
    return float(cand[int(np.argmin(costs))])


# ============================================================================
# 4. Statistical Inference (Bootstrap CI & McNemar Test)
# ============================================================================

def bootstrap_ci(metric_fn, y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    """Stratified bootstrap 95% confidence interval for sample uncertainty."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(ip, len(ip), replace=True), rng.choice(ineg, len(ineg), replace=True)])
        stats.append(metric_fn(y[bi], p[bi]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[int, int, float]:
    """Exact paired binomial test on disagreement contingency table."""
    ea, eb = pred_a != y_true, pred_b != y_true
    b, c = int((ea & ~eb).sum()), int((~ea & eb).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), n=b + c, p=0.5).pvalue)


# ============================================================================
# 5. Validation-Only XGBoost Tuning
# ============================================================================

def tune_xgb(parts: Dict[str, Any], pos_ratio: float, config: Dict[str, Any]):
    """Tuning candidate hyperparameters strictly evaluated on the validation partition."""
    X_tr, y_tr, _ = parts["train"]
    X_va, y_va, _ = parts["val"]
    grid = [(300, 0.10, 4), (400, 0.10, 6), (400, 0.05, 8)]
    best, best_ap = None, -1.0
    print("\n[Tuning] XGBoost candidates evaluated on Validation set (never on test):")
    for n_est, lr, depth in grid:
        m = XGBClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=config["base_seed"],
        )
        m.fit(X_tr, y_tr)
        ap = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
        print(f"  Candidate: n_est={n_est} lr={lr} depth={depth} -> AP_val = {ap:.4f}")
        if ap > best_ap:
            best_ap, best = ap, (n_est, lr, depth)
    config["_xgb_n_estimators"], config["_xgb_lr"], config["_xgb_depth"] = best
    print(f"  Selected: n_est={best[0]}, lr={best[1]}, max_depth={best[2]} (AP_val = {best_ap:.4f})\n")


# ============================================================================
# 6. Main Experiment Execution
# ============================================================================

@dataclass
class ResultRow:
    model: str
    strategy: str
    n_seeds: int
    ap_val: float
    ap_test_mean: float
    ap_test_std: float
    ap_test_ci: tuple
    roc_test: float
    brier_test: Optional[float]
    thr_f1: float
    f1_test: float
    prec_test: float
    rec_test: float
    thr_cost: float
    cost_test: float
    fit_seconds: float


def seeds_for(name: str, strategy: str, config: Dict[str, Any]) -> List[int]:
    stochastic = (name not in DETERMINISTIC_MODELS) or (strategy == "smote")
    k = config["n_seeds"] if stochastic else 1
    return [config["base_seed"] + i for i in range(k)]


def run_experiment(parts: Dict[str, Any], config: Dict[str, Any]):
    X_tr, y_tr, _ = parts["train"]
    X_va, y_va, amt_va = parts["val"]
    X_te, y_te, amt_te = parts["test"]
    pos_ratio = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    cost_fp = config["cost_fp"]

    rows, preds_cost, probas = [], {}, {}

    def finalize(name, strategy, aps_te, p_va_m, p_te_m, fit_s, brier_ok):
        thr_f1 = threshold_max_f1(y_va, p_va_m)
        thr_c = threshold_min_cost(y_va, p_va_m, amt_va, cost_fp)
        y_f1 = (p_te_m >= thr_f1).astype(int)
        y_c = (p_te_m >= thr_c).astype(int)
        row = ResultRow(
            model=name,
            strategy=strategy,
            n_seeds=len(aps_te),
            ap_val=round(float(average_precision_score(y_va, p_va_m)), 4),
            ap_test_mean=round(float(np.mean(aps_te)), 4),
            ap_test_std=round(float(np.std(aps_te)), 4),
            ap_test_ci=bootstrap_ci(average_precision_score, y_te, p_te_m, config["n_bootstrap"], config["base_seed"]),
            roc_test=round(float(roc_auc_score(y_te, p_te_m)), 4),
            brier_test=round(float(brier_score_loss(y_te, p_te_m)), 6) if brier_ok else None,
            thr_f1=round(thr_f1, 4),
            f1_test=round(float(f1_score(y_te, y_f1)), 4),
            prec_test=round(float(precision_score(y_te, y_f1, zero_division=0)), 4),
            rec_test=round(float(recall_score(y_te, y_f1)), 4),
            thr_cost=round(thr_c, 4),
            cost_test=round(expected_cost(y_te, y_c, amt_te, cost_fp), 2),
            fit_seconds=round(fit_s, 1),
        )
        rows.append(row)
        key = f"{name} | {strategy}"
        preds_cost[key] = y_c
        probas[key] = p_te_m
        print(f"[{key:40s}] AP={row.ap_test_mean:.4f}±{row.ap_test_std:.4f} F1={row.f1_test:.4f} Cost=R${row.cost_test:,.0f} ({row.n_seeds} seeds, {fit_s:.1f}s)")

    # 1. Supervised ablation
    print("\n--- Running Supervised Ablation Benchmark ---")
    for name in SUPERVISED_MODELS:
        for strategy in ABLATION_STRATEGIES:
            t0 = time.time()
            aps, pv, pt = [], [], []
            for sd in seeds_for(name, strategy, config):
                m = make_supervised(name, strategy, sd, pos_ratio, config)
                m.fit(X_tr, y_tr)
                pva = m.predict_proba(X_va)[:, 1]
                pte = m.predict_proba(X_te)[:, 1]
                aps.append(average_precision_score(y_te, pte))
                pv.append(pva)
                pt.append(pte)
            finalize(name, strategy, aps, np.mean(pv, axis=0), np.mean(pt, axis=0), time.time() - t0, brier_ok=True)

    # 2. Unsupervised anomaly detectors
    print("\n--- Running Unsupervised Anomaly Detectors ---")
    for name in UNSUPERVISED_MODELS:
        t0 = time.time()
        aps, pv, pt = [], [], []
        for sd in seeds_for(name, "unsupervised", config):
            scorer = IsoForestScorer(sd) if name == "Isolation Forest" else AutoencoderScorer(sd)
            scorer.fit(X_tr, y_tr)
            s_va, s_te = scorer.score(X_va), scorer.score(X_te)
            pva = minmax_to_proba(s_va, s_va)
            pte = minmax_to_proba(s_va, s_te)
            aps.append(average_precision_score(y_te, pte))
            pv.append(pva)
            pt.append(pte)
        finalize(name, "unsupervised", aps, np.mean(pv, axis=0), np.mean(pt, axis=0), time.time() - t0, brier_ok=False)

    df = pd.DataFrame([asdict(r) for r in rows])
    top2 = df.nsmallest(2, "cost_test")
    k1 = f"{top2.iloc[0]['model']} | {top2.iloc[0]['strategy']}"
    k2 = f"{top2.iloc[1]['model']} | {top2.iloc[1]['strategy']}"
    b, c, p = mcnemar_exact(y_te, preds_cost[k1], preds_cost[k2])
    mcnemar = {"config_a": k1, "config_b": k2, "only_a_errs": b, "only_b_errs": c, "p_value": round(p, 4)}
    print(f"\nMcNemar Exact Test ({k1} vs {k2}): Disagreements: only_A={b}, only_B={c} | p-value={p:.4f}")

    return df, {
        "mcnemar": mcnemar,
        "probas": probas,
        "preds": preds_cost,
        "y_test": y_te,
        "amt_test": amt_te,
        "y_val": y_va,
        "amt_val": amt_va,
    }


# ============================================================================
# 7. Visualization & Explainability
# ============================================================================

def run_shap(parts: Dict[str, Any], outdir: Path, config: Dict[str, Any]):
    """Generates global SHAP summary plot for XGBoost."""
    try:
        import shap  # noqa
    except ImportError:
        print("\n[SHAP] 'shap' package not installed. Skipping explanation plot (pip install shap).")
        return

    X_tr, y_tr, _ = parts["train"]
    X_te, y_te, _ = parts["test"]
    cols = [f"V{i}" for i in range(1, 29)] + ["hour", "log_amount"]
    pos_ratio = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    
    model = XGBClassifier(
        n_estimators=config.get("_xgb_n_estimators", 400),
        learning_rate=config.get("_xgb_lr", 0.1),
        max_depth=config.get("_xgb_depth", 6),
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_ratio,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=config["base_seed"],
    )
    model.fit(X_tr, y_tr)
    rng = np.random.default_rng(config["base_seed"])
    idx = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
    
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te[idx])
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_te[idx], feature_names=cols, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_xgb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved to {outdir / 'shap_summary_xgb.png'}")


def make_plots(df: pd.DataFrame, extras: Dict[str, Any], outdir: Path, config: Dict[str, Any]):
    """Renders Precision-Recall, Calibration, and Economic Cost vs. Threshold curves."""
    y_te, amt_te = extras["y_test"], extras["amt_test"]
    best = df.loc[df.groupby("model")["ap_val"].idxmax()]

    # 1. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in best.iterrows():
        key = f"{r['model']} | {r['strategy']}"
        prec, rec, _ = precision_recall_curve(y_te, extras["probas"][key])
        ax.plot(rec, prec, label=f"{r['model']} (AP = {r['ap_test_mean']:.3f})", lw=1.6)
    ax.axhline(y_te.mean(), ls="--", c="gray", lw=1, label=f"Random Chance (AP = {y_te.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Curves on Out-of-Time Test Set", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "pr_curves_test.png", dpi=150)
    plt.close(fig)

    # 2. Probability Calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    sup = best[best["strategy"] != "unsupervised"].nlargest(3, "ap_val")
    for _, r in sup.iterrows():
        key = f"{r['model']} | {r['strategy']}"
        CalibrationDisplay.from_predictions(y_te, extras["probas"][key], n_bins=10, ax=ax, name=r["model"])
    ax.set_title("Probability Calibration on Test Set (Reliability Diagram)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "calibration_test.png", dpi=150)
    plt.close(fig)

    # 3. Cost vs Threshold
    top = df.nsmallest(1, "cost_test").iloc[0]
    key = f"{top['model']} | {top['strategy']}"
    p = extras["probas"][key]
    ts = np.linspace(0, 1, 200)
    cs = [expected_cost(y_te, (p >= t).astype(int), amt_te, config["cost_fp"]) for t in ts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, cs, lw=2, color="navy", label="Expected Test Cost")
    ax.axvline(top["thr_cost"], ls="--", c="crimson", lw=1.5, label=f"Optimal Thr (Val) = {top['thr_cost']:.3f}")
    do_nothing = float(amt_te[y_te == 1].sum())
    ax.axhline(do_nothing, ls=":", c="gray", lw=1.5, label=f"Approve All baseline = R${do_nothing:,.0f}")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Total Economic Cost in Test Set (R$)", fontsize=11)
    ax.set_title(f"Economic Cost vs. Threshold Curve — {key}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)


# ============================================================================
# 8. Main Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Benchmark (v2.1)")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG["data_path"], help="Path to creditcard.csv")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="Directory for output files")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_CONFIG["n_seeds"], help="Number of random seeds")
    parser.add_argument("--sample-n", type=int, default=DEFAULT_CONFIG["sample_n"], help="Optional row subsample for fast testing")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability plot")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["data_path"] = args.data_path
    config["output_dir"] = args.output_dir
    config["n_seeds"] = args.n_seeds
    config["sample_n"] = args.sample_n
    if args.skip_shap:
        config["run_shap"] = False

    outdir = Path(config["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print(" Credit Card Fraud Detection — Production ML Benchmark (v2.1)")
    print("===================================================================")
    print(f"Dataset path: {config['data_path']}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Stochastic seeds: {config['n_seeds']}")

    print("\nLoading dataset...")
    df = load_data(config["data_path"])
    if config["sample_n"]:
        print(f"Subsampling first {config['sample_n']} transactions...")
        df = df.sort_values("Time").head(config["sample_n"])
    df = engineer_features(df)
    
    print("\nChronological Out-of-Time Split (70/15/15):")
    parts = temporal_split(df, train_frac=config["train_frac"], val_frac=config["val_frac"])

    if config["run_xgb_tuning"]:
        pos_ratio = float((parts["train"][1] == 0).sum() / max(1, (parts["train"][1] == 1).sum()))
        tune_xgb(parts, pos_ratio, config)

    df_res, extras = run_experiment(parts, config)

    # Cost Sensitivity Analysis (Top 3 models)
    print("\nEvaluating Cost Ratio Sensitivity Analysis (top 3 by test cost)...")
    X_tr, y_tr, _ = parts["train"]
    X_va, y_va, amt_va = parts["val"]
    X_te, y_te, amt_te = parts["test"]
    pos_ratio = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    sens_rows = []
    
    for _, r in df_res.nsmallest(3, "cost_test").iterrows():
        name, strat = r["model"], r["strategy"]
        if strat == "unsupervised":
            scorer = IsoForestScorer(config["base_seed"]) if name == "Isolation Forest" else AutoencoderScorer(config["base_seed"])
            scorer.fit(X_tr, y_tr)
            s_va = scorer.score(X_va)
            p_va = minmax_to_proba(s_va, s_va)
            p_te = minmax_to_proba(s_va, scorer.score(X_te))
        else:
            m = make_supervised(name, strat, config["base_seed"], pos_ratio, config)
            m.fit(X_tr, y_tr)
            p_va = m.predict_proba(X_va)[:, 1]
            p_te = m.predict_proba(X_te)[:, 1]
            
        for cfp in config["cost_fp_grid"]:
            t = threshold_min_cost(y_va, p_va, amt_va, cfp)
            c = expected_cost(y_te, (p_te >= t).astype(int), amt_te, cfp)
            sens_rows.append({"config": f"{name} | {strat}", "cost_fp": cfp, "threshold": round(t, 4), "cost_test": round(c, 2)})

    sens = pd.DataFrame(sens_rows)
    sens.to_csv(outdir / "cost_sensitivity.csv", index=False)

    # Save benchmark results
    df_res = df_res.sort_values(["cost_test", "ap_test_mean"], ascending=[True, False]).reset_index(drop=True)
    df_res.to_csv(outdir / "results_v2_1.csv", index=False)
    with open(outdir / "mcnemar.json", "w") as f:
        json.dump(extras["mcnemar"], f, indent=2)

    make_plots(df_res, extras, outdir, config)
    if config["run_shap"]:
        run_shap(parts, outdir, config)

    cols = ["model", "strategy", "n_seeds", "ap_test_mean", "ap_test_std", "ap_test_ci", "roc_test", "f1_test", "thr_cost", "cost_test"]
    print("\n======================= FINAL BENCHMARK RESULTS =======================")
    print(df_res[cols].to_string(index=False))
    
    y_te = extras["y_test"]
    amt_te = extras["amt_test"]
    do_nothing = float(amt_te[y_te == 1].sum())
    block_all = float((y_te == 0).sum() * config["cost_fp"])
    print(f"\nBaselines: Approve All (Do Nothing) = R${do_nothing:,.2f} | Block All = R${block_all:,.2f}")
    print(f"Results and plots saved to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
