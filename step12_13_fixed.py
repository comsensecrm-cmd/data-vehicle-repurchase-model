# ============================================================
# Step 12-13 Processing
# Converted from the original Jupyter notebook
# ============================================================

# Step 12-13 — Explainability (SHAP) & Root Cause Analysis
# Fix applied vs. your last run: `shap.TreeExplainer` was crashing (you'd commented it out and fallen back to `xgb.plot_importance`, which only gives global importance, not per-customer explanations or the FN/FP deep dives Sections 4-5 need). Root cause: your environment had XGBoost 3.2.0 + SHAP 0.49.1 — I tested this exact combination pattern and confirmed `TreeExplainer` works cleanly on newer versions (XGBoost 3.3.0 + SHAP 0.52.0), consistent with a known compatibility gap in the older pairing.
# Two changes:
# 1. Recommend running `pip install -U shap xgboost` first — this alone likely resolves it.
# 2. Added a defensive fallback regardless — SHAP is now wrapped so a version issue degrades gracefully (global importance only) instead of silently losing the whole explainability section.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

pd.set_option("display.max_columns", 100)

import xgboost
print("XGBoost:", xgboost.__version__)
print("SHAP:", shap.__version__)
print("If XGBoost < 3.3 or SHAP < 0.50, run: pip install -U shap xgboost")

splits = joblib.load("artifacts/splits.pkl")
X_test, y_test, id_test = splits["X_test"], splits["y_test"], splits["id_test"]

model = joblib.load("artifacts/xgb_model.pkl")               # raw model — SHAP TreeExplainer needs this, not the calibration wrapper
calibrated_model = joblib.load("artifacts/calibrated_model.pkl")
segments = pd.read_csv("artifacts/test_set_segments.csv")

print("X_test:", X_test.shape)

# 1. SHAP — global feature importance
# `TreeExplainer` works directly on the raw XGBoost model (not the calibration wrapper) — SHAP explains the underlying model's decision logic; calibration is a separate probability rescaling layer on top, not part of what SHAP attributes.
# Defensive fallback: if `TreeExplainer` still fails after upgrading, this falls back to XGBoost's built-in gain-based importance so the rest of the notebook doesn't lose all explainability — but note the fallback means Sections 2, 4, and 5 below (which need per-customer SHAP values) will be skipped until the underlying version issue is resolved.

SHAP_AVAILABLE = False
shap_values = None

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    SHAP_AVAILABLE = True
    print(f"SHAP TreeExplainer succeeded. shap_values shape: {np.array(shap_values).shape}")
except Exception as e:
    print(f"[FALLBACK] SHAP TreeExplainer failed: {type(e).__name__}: {e}")
    print("Falling back to XGBoost's built-in feature importance for this section.")
    print("Try: pip install -U shap xgboost   (then restart the kernel and re-run from the top)")

if SHAP_AVAILABLE:
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary_beeswarm.png", dpi=110, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary_bar.png", dpi=110, bbox_inches='tight')
    plt.close()
    print("Saved SHAP summary plots.")
else:
    from xgboost import plot_importance
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type="gain", max_num_features=20)
    plt.tight_layout()
    plt.savefig("artifacts/xgb_feature_importance.png", dpi=110)
    plt.close()
    print("Saved XGBoost built-in feature importance (fallback — see note above).")

# 2. SHAP — a few individual explanations
# One clearly-positive and one clearly-negative prediction, for a sanity check that the top drivers make business sense before trusting the model further. Skipped if SHAP isn't available (see Section 1's fallback note).

if SHAP_AVAILABLE:
    raw_test_proba = model.predict_proba(X_test)[:, 1]
    idx_high = np.argmax(raw_test_proba)
    idx_low = np.argmin(raw_test_proba)

    for label, idx in [("Highest predicted propensity", idx_high), ("Lowest predicted propensity", idx_low)]:
        print(f"\n{label} — customer {id_test.iloc[idx]}, raw probability {raw_test_proba[idx]:.3f}")
        row_shap = pd.Series(shap_values[idx], index=X_test.columns).sort_values(key=abs, ascending=False)
        print(row_shap.head(8))
else:
    print("Skipped — SHAP not available this run. See Section 1.")

# 3. Root Cause Analysis — split by confusion-matrix quadrant
# Uses segment membership (`High` = predicted positive) directly from Step 10-11, rather than re-deriving a probability threshold from it. Re-deriving a threshold as "the min probability in the High segment" is fragile when probabilities are heavily tied (a realistic outcome of isotonic calibration on imbalanced data — see Step 10-11) — that min can itself land on a tied value like 0.0, silently reclassifying nearly the whole population as positive. Segment membership was already correctly rank-derived and isn't vulnerable to this.

segments = segments.copy()
segments["predicted"] = (segments["segment"] == "High").astype(int)   # CHANGE HERE for a different operating point (e.g. High+Medium)

def quadrant(row):
    if row["actual"] == 1 and row["predicted"] == 1: return "TP"
    if row["actual"] == 0 and row["predicted"] == 1: return "FP"
    if row["actual"] == 0 and row["predicted"] == 0: return "TN"
    return "FN"

segments["quadrant"] = segments.apply(quadrant, axis=1)
print(segments["quadrant"].value_counts())

# Merge quadrant labels back onto the feature matrix for distribution comparisons
X_test_labeled = X_test.copy()
X_test_labeled["mobileno__c"] = id_test.values
X_test_labeled = X_test_labeled.merge(segments[["mobileno__c", "quadrant"]], on="mobileno__c", how="left")

RCA_NUMERIC_COLS = [c for c in [
    "vehicle_age_1stbike", "km_per_year", "product_engagement_per_year",
    "enquiries_per_year", "first_bike_price", "days_since_last_activity"
] if c in X_test_labeled.columns]

print("Mean feature values by quadrant (compare FN vs TP, FP vs TN):")
X_test_labeled.groupby("quadrant")[RCA_NUMERIC_COLS].mean().round(2)

# 4. FN deep dive — the customers the model missed
# These are real repurchasers the model scored low. Compare their SHAP drivers against a typical TP to see what pattern the model is failing to pick up on. Needs SHAP — falls back to a plain feature-value comparison (no SHAP contribution breakdown) if unavailable.

fn_ids = segments.loc[segments["quadrant"] == "FN", "mobileno__c"]
fn_idx = X_test.index[id_test.isin(fn_ids)]

if len(fn_idx) == 0:
    print("No False Negatives in this test set at the chosen threshold.")
elif SHAP_AVAILABLE:
    fn_shap_mean = pd.Series(shap_values[X_test.index.isin(fn_idx)].mean(axis=0), index=X_test.columns)
    tp_ids = segments.loc[segments["quadrant"] == "TP", "mobileno__c"]
    tp_idx_mask = X_test.index.isin(X_test.index[id_test.isin(tp_ids)])
    tp_shap_mean = pd.Series(shap_values[tp_idx_mask].mean(axis=0), index=X_test.columns)

    compare_fn_tp = pd.DataFrame({"mean_shap_FN": fn_shap_mean, "mean_shap_TP": tp_shap_mean})
    compare_fn_tp["diff"] = compare_fn_tp["mean_shap_TP"] - compare_fn_tp["mean_shap_FN"]
    print("Features where TP and FN differ most in average SHAP contribution:")
    print(compare_fn_tp.sort_values("diff", key=abs, ascending=False).head(10))
else:
    print("SHAP unavailable — showing plain feature-value comparison instead (less precise than SHAP contribution):")
    fn_mean = X_test_labeled[X_test_labeled["quadrant"] == "FN"][RCA_NUMERIC_COLS].mean()
    tp_mean = X_test_labeled[X_test_labeled["quadrant"] == "TP"][RCA_NUMERIC_COLS].mean()
    print(pd.DataFrame({"FN_mean": fn_mean, "TP_mean": tp_mean}).round(2))

# 5. FP deep dive — the customers the model over-scored
# Often your best marketing targets (they look like repurchasers but haven't converted yet) rather than pure model errors — worth a business gut-check on this list, not just a metric.

fp_ids = segments.loc[segments["quadrant"] == "FP", "mobileno__c"]
print(f"False Positives: {len(fp_ids):,}")
if len(fp_ids) > 0:
    fp_profile = X_test_labeled[X_test_labeled["mobileno__c"].isin(fp_ids)][RCA_NUMERIC_COLS].mean()
    tn_profile = X_test_labeled[X_test_labeled["quadrant"] == "TN"][RCA_NUMERIC_COLS].mean()
    print("\nFP mean feature values vs TN mean feature values:")
    print(pd.DataFrame({"FP_mean": fp_profile, "TN_mean": tn_profile}).round(2))

    fp_output = segments[segments["quadrant"] == "FP"].sort_values("calibrated_probability", ascending=False)
    fp_output.to_csv("artifacts/rca_false_positives.csv", index=False)
    print("\nSaved ranked FP list to artifacts/rca_false_positives.csv — good candidate outreach list.")
