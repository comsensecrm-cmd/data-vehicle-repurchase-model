# ============================================================
# Step 10-11: Calibration Segments
# Converted from the original Jupyter notebook
# ============================================================

# Step 10-11 — Probability Calibration & Propensity Segments
# Loads the model + splits saved by Step 5-9. Calibrates raw probabilities against the validation set (held out from training, never touched by CV) using both Platt scaling and isotonic regression, picks whichever is better-calibrated, then builds propensity segments (decile-based) with lift/gain — the business-facing output this whole pipeline is for.
# Why calibration matters here specifically: Step 4 sampled negatives at a ratio different from the true population rate. Raw model probabilities reflect *that sampled ratio*, not real-world likelihood — without calibration, a raw '80% probability' does NOT mean 80% of such customers actually repurchase.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

os.makedirs("artifacts", exist_ok=True)
pd.set_option("display.max_columns", 100)

splits = joblib.load("artifacts/splits.pkl")
X_train, y_train, id_train = splits["X_train"], splits["y_train"], splits["id_train"]
X_val, y_val, id_val = splits["X_val"], splits["y_val"], splits["id_val"]
X_test, y_test, id_test = splits["X_test"], splits["y_test"], splits["id_test"]

model = joblib.load("artifacts/xgb_model.pkl")
print("Loaded model and splits. X_val:", X_val.shape, "X_test:", X_test.shape)

# 1. Calibrate — fit on validation, never on train or test

raw_val_proba = model.predict_proba(X_val)[:, 1]
raw_test_proba = model.predict_proba(X_test)[:, 1]

# sklearn >=1.6 removed cv='prefit' in favor of wrapping the model in FrozenEstimator.
# This shim keeps the notebook working across sklearn versions.
try:
    from sklearn.frozen import FrozenEstimator
    frozen_model = FrozenEstimator(model)
    calibrated_sigmoid = CalibratedClassifierCV(frozen_model, method='sigmoid')
    calibrated_isotonic = CalibratedClassifierCV(frozen_model, method='isotonic')
except ImportError:
    # sklearn < 1.6
    calibrated_sigmoid = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')

calibrated_sigmoid.fit(X_val, y_val)
calibrated_isotonic.fit(X_val, y_val)

sigmoid_test_proba = calibrated_sigmoid.predict_proba(X_test)[:, 1]
isotonic_test_proba = calibrated_isotonic.predict_proba(X_test)[:, 1]

for name, proba in [("Raw (uncalibrated)", raw_test_proba),
                     ("Platt / sigmoid", sigmoid_test_proba),
                     ("Isotonic", isotonic_test_proba)]:
    brier = brier_score_loss(y_test, proba)
    print(f"{name:22s} Brier score (lower is better): {brier:.4f}")

fig, ax = plt.subplots(figsize=(8, 8))
for name, proba in [("Raw", raw_test_proba), ("Platt/sigmoid", sigmoid_test_proba), ("Isotonic", isotonic_test_proba)]:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy='quantile')
    ax.plot(mean_pred, frac_pos, marker='o', label=name)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of actual positives")
ax.set_title("Calibration curve — closer to the diagonal is better")
ax.legend()
plt.tight_layout()
plt.savefig("artifacts/calibration_curve.png", dpi=110)
plt.close()

# CHANGE HERE if isotonic doesn't win on your real data — pick whichever had the lower Brier score above.
# Isotonic is more flexible (good with enough validation data) but can overfit with a small val set;
# sigmoid is more constrained and usually safer if X_val is not very large.
FINAL_CALIBRATED_MODEL = calibrated_isotonic   # CHANGE HERE based on the Brier scores printed above

joblib.dump(FINAL_CALIBRATED_MODEL, "artifacts/calibrated_model.pkl")
print("Calibrated model saved to artifacts/calibrated_model.pkl")

frac_pos, mean_pred = calibration_curve(y_test, isotonic_test_proba, n_bins=10, strategy='quantile')
# also print how many test rows actually fall in each bin
bin_counts = pd.qcut(isotonic_test_proba, 10, duplicates='drop').value_counts().sort_index()
print(bin_counts)

from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, isotonic_test_proba)
print("Brier Score:", brier)

# 2. Propensity segments (deciles) with lift and gain
# The business-facing output: rank every test customer by calibrated probability, bucket into deciles, and show how much better than random targeting each decile performs — this is what actually gets handed to the marketing/CRM team, not a single threshold.

calibrated_test_proba = FINAL_CALIBRATED_MODEL.predict_proba(X_test)[:, 1]

decile_df = pd.DataFrame({
    "mobileno__c": id_test.values,
    "actual": y_test.values,
    "calibrated_probability": calibrated_test_proba,
})

# Rank first, then bin the RANK (not the raw probability) into deciles. Isotonic calibration
# commonly produces many tied values (e.g. a flat region at 0 when validation data is imbalanced),
# and pd.qcut on raw values with heavy ties silently collapses bins via duplicates='drop' —
# ranking first guarantees exactly 10 real, non-empty, roughly-equal-sized deciles regardless of ties.
decile_df["_rank"] = decile_df["calibrated_probability"].rank(method="first", ascending=False)
decile_df["decile"] = pd.qcut(decile_df["_rank"], 10, labels=False)   # decile 0 = highest propensity, directly
decile_df = decile_df.drop(columns=["_rank"])

print("Rows per decile (should be roughly equal, none empty):")
print(decile_df["decile"].value_counts().sort_index())

overall_rate = decile_df["actual"].mean()

lift_table = decile_df.groupby("decile").agg(
    customers=("actual", "size"),
    actual_positives=("actual", "sum"),
    positive_rate=("actual", "mean"),
    mean_calibrated_prob=("calibrated_probability", "mean"),
).reset_index()
lift_table["lift"] = (lift_table["positive_rate"] / overall_rate).round(2)
lift_table["cumulative_positives"] = lift_table.sort_values("decile")["actual_positives"].cumsum()
lift_table["cumulative_gain_pct"] = (lift_table["cumulative_positives"] / decile_df["actual"].sum() * 100).round(1)

print(f"Overall test-set positive rate: {overall_rate*100:.2f}%")
lift_table.sort_values("decile")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(lift_table["decile"].astype(str), lift_table["lift"])
axes[0].axhline(1, linestyle='--', color='gray')
axes[0].set_title("Lift by decile (0 = highest propensity)")
axes[0].set_xlabel("Decile"); axes[0].set_ylabel("Lift over random targeting")

axes[1].plot(lift_table["decile"], lift_table["cumulative_gain_pct"], marker='o')
axes[1].plot([0, 9], [10, 100], linestyle='--', color='gray', label='Random targeting')
axes[1].set_title("Cumulative gain")
axes[1].set_xlabel("Decile (cumulative, best-first)"); axes[1].set_ylabel("% of all positives captured")
axes[1].legend()

plt.tight_layout()
plt.savefig("artifacts/lift_gain_chart.png", dpi=110)
plt.close()

# 3. Map deciles to business segments (High / Medium / Low)

# CHANGE HERE — adjust the decile cutoffs to match business capacity
# (e.g. "how many customers can the outreach team actually contact this month")
def decile_to_segment(d):
    if d <= 1:
        return "High"
    elif d <= 4:
        return "Medium"
    return "Low"

decile_df["segment"] = decile_df["decile"].apply(decile_to_segment)

segment_summary = decile_df.groupby("segment").agg(
    customers=("actual", "size"),
    positive_rate=("actual", "mean"),
).reindex(["High", "Medium", "Low"])
segment_summary["lift"] = (segment_summary["positive_rate"] / overall_rate).round(2)
print(segment_summary)

decile_df.to_csv("artifacts/test_set_segments.csv", index=False)
print("\nSaved per-customer segment assignments to artifacts/test_set_segments.csv")
