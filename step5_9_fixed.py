# ============================================================
# Step 5-9 Processing
# Converted from the original Jupyter notebook
# ============================================================

# Step 5-9 — Column Spec, Split, Encoding, CV Training, Evaluation
# Runs on `modeling_dataset_v2.csv` from the ratio-fixed `build_step3_v2.py`.
# Fixes applied vs. your last run:
# 1. `num_features` restored to the rate-based/safe list — your last run used the raw, leak-prone versions (`Km_driven_by_first_bike`, `total_amount_spent`, `Frequency_gma`, `Frequency_apparel`, `product_count`) that were supposed to be replaced.
# 2. Removed `ratio_actual_expected_service` — it no longer exists (dropped by Step 3 v2) and was silently absent from every run so far.
# 3. `cat_features` restored to the full safe list (was down to just `State`).
# 4. Full test-set evaluation section added back (ROC curve, confusion matrix, classification report, probability distribution) — your last run only had the threshold sweep table, which already showed a real CV-to-test gap (0.89 -> 0.60 precision at threshold 0.5) worth seeing in full before trusting the model.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pltssss
import seaborn as sns
import joblib
import os
from scipy import stats

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score,
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

os.makedirs("artifacts", exist_ok=True)
pd.set_option("display.max_columns", 100)

pd.set_option('display.float_format', '{:.2f}'.format)

# CHANGE HERE — path to the ratio-fixed modeling dataset from Step 3 v2
DF_PATH = r"C:\Users\ArpitaDhumal\Desktop\Retrain_RE_vehicle_upgrade_repurchase\Outputs\step3_v2_outputs\modeling_dataset_v2.csv"
df = pd.read_csv(DF_PATH, low_memory=False)
df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"], errors="coerce")
print(df.shape)

df['label'].value_counts()

df.drop(columns=['first_chassis_number','used_soft_product_flag',
                'first_enquiry_date', 'second_enquiry_date',
                'second_enquiry_type',
                'second_enquired_model_mapped', 'second_enquired_bike_category',
                'First_purchased_model(UCE_or_J-series)','km_was_corrected',
                'service_count_was_corrected', 'service_amount_was_corrected',
                'reference_date',
                'has_second_enquiry',
                'last_activity_date', 'last_purchase_date_service',
                ],inplace=True)

df = df.rename(columns={
    'isfinanced__c': 'isfinanced',
    'Final_ride_done': 'Ride_done_binary',
    'KM_before_2nd_bike': 'Km_driven_by_first_bike',
    'enquiry_count_between_purchases': 'Enquiry_for_2nd_bike',
    'first_purchase_month': 'first_bike_purchase_month',
    'first_purchase_quarter': 'first_bike_purchase_quarter',
    'first_purchased_model_mapped': 'First_purchased_model_family',
    'first_enquired_model_mapped': 'first_enquired_model_family'
})

df.shape

df.columns

# feature selection

# ---------------------------------------------------------------------------
# Feature diagnostic — run BEFORE finalizing num_features / cat_features
# ---------------------------------------------------------------------------

CANDIDATE_NUM_COLS = [c for c in [
    # Main features
    'isfinanced', 'Km_driven_by_first_bike','Ride_done_binary',
    'vehicle_age_1stbike', 'NPS_Rating',
    'total_service_amount', 'service_count', 'used_soft_product_flag',
    'enquiry_count_between_purchases','Frequency_gma','product_count',
    'Frequency_apparel', 'product_engagement_score',
    'total_amount_spent', 'amount_per_service', 'ratio_actual_expected_service',

    # purchased bike date features
    'purchased_in_festival_season',
    'first_bike_price', 'first_bike_purchase_month', 'first_bike_purchase_quarter',

    # per yearly features
    'km_per_year', 'services_per_year', 'service_spend_per_year','enquiries_per_year', 'product_engagement_per_year',

    # days since features
    'days_since_last_activity', 'days_since_last_enquiry', 'days_since_last_service','is_service_overdue',
    'cltv', 'has_NPS_rating',

    # other features
    'km_was_corrected','service_count_was_corrected', 'service_amount_was_corrected',
] if c in df.columns]

CANDIDATE_CAT_COLS = [c for c in [
    'First_purchased_model_family', 'first_bike_category',
    'first_enquired_model_family', 'first_enquired_bike_category',
    'State', 'Segment', 'first_enquiry_type',
] if c in df.columns]

# --- Numeric features: point-biserial correlation + Mann-Whitney U significance + effect size ---
numeric_results = []
for col in CANDIDATE_NUM_COLS:
    vals = df[col].dropna()
    y_aligned = df.loc[vals.index, 'label']
    if vals.nunique() < 2 or y_aligned.nunique() < 2:
        continue

    corr = np.corrcoef(vals, y_aligned)[0, 1]

    pos = vals[y_aligned == 1]
    neg = vals[y_aligned == 0]
    # Mann-Whitney U — doesn't assume normal distribution, appropriate for skewed features like these
    try:
        u_stat, p_value = stats.mannwhitneyu(pos, neg, alternative='two-sided')
    except ValueError:
        p_value = np.nan

    pooled_std = np.sqrt(((len(pos)-1)*pos.std()**2 + (len(neg)-1)*neg.std()**2) / (len(pos)+len(neg)-2))
    cohens_d = (pos.mean() - neg.mean()) / pooled_std if pooled_std > 0 else np.nan

    numeric_results.append({
        'feature': col, 'correlation_with_target': round(corr, 4),
        'p_value': p_value, 'cohens_d': round(cohens_d, 4) if pd.notna(cohens_d) else np.nan,
        'significant_at_0.05': p_value < 0.05 if pd.notna(p_value) else None,
    })

numeric_diag = pd.DataFrame(numeric_results).sort_values('cohens_d', key=abs, ascending=False)
print("===== Numeric feature diagnostic =====")
print(numeric_diag.to_string(index=False))

# --- Categorical features: chi-square test of independence + Cramér's V ---
categorical_results = []
for col in CANDIDATE_CAT_COLS:
    contingency = pd.crosstab(df[col].fillna('Missing'), df['label'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    categorical_results.append({
        'feature': col, 'chi2': round(chi2, 2), 'p_value': p_value,
        'cramers_v': round(cramers_v, 4), 'significant_at_0.05': p_value < 0.05,
    })

categorical_diag = pd.DataFrame(categorical_results).sort_values('cramers_v', ascending=False)
print("\n===== Categorical feature diagnostic =====")
print(categorical_diag.to_string(index=False))

# --- Multicollinearity check among numeric features (helps spot redundant pairs) ---
corr_matrix = df[CANDIDATE_NUM_COLS].corr()
high_corr_pairs = []
for i in range(len(CANDIDATE_NUM_COLS)):
    for j in range(i+1, len(CANDIDATE_NUM_COLS)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.7:
            high_corr_pairs.append((CANDIDATE_NUM_COLS[i], CANDIDATE_NUM_COLS[j], round(c, 3)))

print("\n===== Highly correlated numeric feature pairs (|r| > 0.7 — candidates for dropping one) =====")
for a, b, c in high_corr_pairs:
    print(f"  {a} <-> {b}: r = {c}")
if not high_corr_pairs:
    print("  None found.")

# 1. Column specification
# Restored to the rate-based / leak-safe list. If you deliberately want to compare against the raw-feature version again for diagnostic purposes, keep both lists around, but don't train the model you intend to deploy on the raw one.

df.notna().sum()/df.shape[0]

import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include="number").columns.drop("label")

for col in num_cols:
    plt.figure(figsize=(6,4))
    df[col].hist(bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

summary = df[num_cols].describe().T

summary["missing"] = df[num_cols].isnull().sum()

summary["skewness"] = df[num_cols].skew()

summary["kurtosis"] = df[num_cols].kurt()

summary

import matplotlib.pyplot as plt

for col in num_cols:

    plt.figure(figsize=(6,4))

    df[df["label"] == 0][col].hist(
        bins=30,
        alpha=0.5,
        label="Target 0"
    )

    df[df["label"] == 1][col].hist(
        bins=30,
        alpha=0.5,
        label="Target 1"
    )

    plt.title(col)
    plt.legend()
    plt.show()

for col in num_cols:

    plt.figure(figsize=(6,4))

    df.boxplot(column=col, by="label")

    plt.title(col)

    plt.suptitle("")

    plt.show()

cat_cols = df.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:

    print("=" * 50)

    print(col)

    print(df[col].value_counts(normalize=True))

for col in cat_cols:

    cross_tab = pd.crosstab(
        df[col],
        df["label"],
        normalize="index"
    )

    print("=" * 60)

    print(col)

    print(cross_tab)

corr = df[num_cols.tolist() + ["label"]].corr(numeric_only=True)

corr["label"].sort_values(ascending=False)

# Does missingness in first_enquiry_type differ systematically by class or by cohort?
print(df.groupby('label')['first_enquiry_type'].apply(lambda s: s.isna().mean()))
print(df.groupby(df['first_purchase_date'].dt.year)['first_enquiry_type'].apply(lambda s: s.isna().mean()))

num_features = [
        # Main features
        'isfinanced', 'Ride_done_binary',
        'vehicle_age_1stbike', 'NPS_Rating',

        # 'enquiry_count_between_purchases',
        'total_amount_spent', 'ratio_actual_expected_service',
        'Frequency_gma','product_count','Frequency_apparel',
        # 'product_engagement_score', 'total_service_amount', 'service_count',
        # 'Km_driven_by_first_bike','amount_per_service',

        # purchased bike date features
        # 'purchased_in_festival_season',
        # 'first_bike_price', 'first_bike_purchase_month', 'first_bike_purchase_quarter',

        # per yearly features
        'km_per_year', 'services_per_year', 'service_spend_per_year','enquiries_per_year'
        #   'product_engagement_per_year'

        # days since features
        # 'days_since_last_activity' ,
        #  'days_since_last_service','is_service_overdue'
        # 'days_since_last_enquiry'
        # 'cltv', 'has_NPS_rating',

        # other features
        # 'km_was_corrected','service_count_was_corrected', 'service_amount_was_corrected',
]

cat_features = [
    'First_purchased_model_family'
    # 'first_bike_category',
    # 'first_enquired_model_family',
    # 'first_enquired_bike_category'
    # 'State'
    # 'Segment',
    # 'first_enquiry_type',

    # second_enquiry_type / second_enquired_model_mapped / second_enquired_bike_category are
    # also safe now (nulled out by Step 3 v2 whenever the enquiry itself is after the cutoff) —
    # CHANGE HERE to add them in if you want the extra signal.
    # First_purchased_model(UCE_or_J-series) still excluded pending the classification bug fix.
]

target = 'label'
date_col = 'first_purchase_date'

missing_num = [c for c in num_features if c not in df.columns]
missing_cat = [c for c in cat_features if c not in df.columns]

if missing_num or missing_cat:
    print("[WARNING] Columns listed but not found in df:")
    print("  num:", missing_num)
    print("  cat:", missing_cat)
else:
    print("All listed columns found.")

print(f"\n{len(num_features)} numeric + {len(cat_features)} categorical features, target = '{target}'")

# Does the 'Other' rate differ wildly by class?
print(df.groupby('label')['first_enquired_bike_category'].apply(lambda s: (s == 'Other').mean()))

# Does it track sensibly with purchase year, or jump abruptly (pipeline artifact signature)?
print(df.groupby(df['first_purchase_date'].dt.year)['first_enquired_bike_category'].apply(lambda s: (s == 'Other').mean()))

# Is this literally the same population as the missing first_enquiry_type from your last run?
print(pd.crosstab(df['first_enquiry_type'].isna(), df['first_enquired_bike_category'] == 'Other'))

# 2. Time-based test, stratified-random train/val

FILTER_TO_CUTOFF = True
REAL_CUTOFF_DATE = "2025-10-31"   # CHANGE HERE — your confirmed true cutoff

_check_date = pd.to_datetime(df['first_purchase_date'], errors='coerce')
print("Max first_purchase_date in modeling_dataset:", _check_date.max().date())
if 'reference_date' in df.columns:
    _check_ref = pd.to_datetime(df['reference_date'], errors='coerce')
    print("Max reference_date in modeling_dataset:", _check_ref.max().date())

if FILTER_TO_CUTOFF:
    before = len(df)
    df = df[pd.to_datetime(df['first_purchase_date'], errors='coerce') <= pd.Timestamp(REAL_CUTOFF_DATE)]
    print(f"Filtered to <= {REAL_CUTOFF_DATE}: {before:,} -> {len(df):,} rows")

def split_data_time_based_stratified(
    df, target_col, date_col='first_purchase_date',
    test_ratio=0.2, val_ratio_within_train=0.15, drop_cols=None
):
    print("\n--- Performing Time-Based + Stratified Split ---")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    n_total = len(df)
    test_size = int(n_total * test_ratio)
    df_trainval = df.iloc[:-test_size].reset_index(drop=True)
    df_test = df.iloc[-test_size:].reset_index(drop=True)

    drop_cols = (drop_cols or []) + [target_col, date_col]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X_trainval = df_trainval.drop(columns=drop_cols, errors='ignore')
    y_trainval = df_trainval[target_col]
    id_trainval = df_trainval['mobileno__c'].reset_index(drop=True)

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_trainval, y_trainval, id_trainval,
        test_size=val_ratio_within_train, stratify=y_trainval, random_state=42, shuffle=True
    )

    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test = df_test[target_col]
    id_test = df_test['mobileno__c'].reset_index(drop=True)

    def class_dist(y): return dict(y.value_counts(normalize=True).round(3))

    print(f"\nTotal Samples: {n_total}")
    print("\nSplit Details:")
    print(f"Train:      {len(X_train)} | Class dist: {class_dist(y_train)}")
    print(f"Validation: {len(X_val)} | Class dist: {class_dist(y_val)}")
    print(f"Test (latest): {len(X_test)} | Class dist: {class_dist(y_test)}")
    print("\nDate Ranges:")
    print(f"  Train/Val block: {df_trainval[date_col].min().date()} -> {df_trainval[date_col].max().date()}")
    print(f"  Test block     : {df_test[date_col].min().date()} -> {df_test[date_col].max().date()}")

    pos_rate_train = class_dist(y_train).get(1, 0)
    if pos_rate_train > 0.5:
        print("\n[FLAG] Positive class is the MAJORITY in the training set "
              f"({pos_rate_train*100:.1f}%). Revisit Step 3 v2 sampling before trusting downstream metrics.")

    return X_train, y_train, X_val, y_val, X_test, y_test, id_train, id_val, id_test

ID_AND_LEAKAGE_COLS = [
    'mobileno__c', 'first_chassis_number', 'reference_date',
    'second_purchase_date', 'second_purchased_model_mapped',
    'second_bike_category', 'second_purchased_model(UCE_or_J-series)',
    'is_mature',
]

X_train, y_train, X_val, y_val, X_test, y_test, id_train, id_val, id_test = split_data_time_based_stratified(
    df, target_col=target, date_col=date_col,
    test_ratio=0.2, val_ratio_within_train=0.15,
    drop_cols=[c for c in df.columns if c not in num_features + cat_features + [target, date_col]]
)

# 3. Encoding — fit on train only

for X in (X_train, X_val, X_test):
    X[cat_features] = X[cat_features].fillna('Missing')

drop_list = None
encoder = OneHotEncoder(handle_unknown='ignore', drop=drop_list, sparse_output=False)
encoder.fit(X_train[cat_features])

ENCODER_PATH = 'artifacts/onehot_encoder.pkl'
joblib.dump(encoder, ENCODER_PATH)
print(f"Encoder saved at {ENCODER_PATH}")

encoded_cols = encoder.get_feature_names_out(cat_features)

def apply_encoder(X):
    encoded = encoder.transform(X[cat_features])
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    return pd.concat([X.drop(columns=cat_features), encoded_df], axis=1)

X_train = apply_encoder(X_train)
X_val = apply_encoder(X_val)
X_test = apply_encoder(X_test)

print(X_train.shape, X_val.shape, X_test.shape)

# 4. Cross-validated training (XGBoost, StratifiedKFold)

def train_XGBoost_model(X, y, n_splits=5):
    print("\n--- Training Model (XGBoost) ---")
    num_positive = (y == 1).sum()
    num_negative = (y == 0).sum()
    scale_pos_weight = num_negative / num_positive
    print("scale_pos_weight:", scale_pos_weight)
    if scale_pos_weight < 1:
        print("[FLAG] scale_pos_weight < 1 means positives currently OUTNUMBER negatives.")

    base_model = XGBClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        eval_metric='auc', reg_alpha=1, reg_lambda=1, min_child_weight=5, gamma=1
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_auc_list, val_pr_list = [], []
    val_precision_list, val_recall_list, val_f1_list = [], [], []

    print("\n========= CROSS VALIDATION =========")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(base_model)
        model.fit(X_train_fold, y_train_fold)

        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        val_auc = roc_auc_score(y_val_fold, y_val_proba)
        val_pr = average_precision_score(y_val_fold, y_val_proba)
        val_precision = precision_score(y_val_fold, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val_fold, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)

        val_auc_list.append(val_auc); val_pr_list.append(val_pr)
        val_precision_list.append(val_precision); val_recall_list.append(val_recall); val_f1_list.append(val_f1)

        print(f"\nFold {fold}")
        print(f"AUC: {val_auc:.4f} | PR-AUC: {val_pr:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    print("\n========= CV SUMMARY =========")
    print(f"Mean Val AUC:       {np.mean(val_auc_list):.4f}")
    print(f"Mean Val PR-AUC:    {np.mean(val_pr_list):.4f}")
    print(f"Mean Val Precision: {np.mean(val_precision_list):.4f}")
    print(f"Mean Val Recall:    {np.mean(val_recall_list):.4f}")
    print(f"Mean Val F1:        {np.mean(val_f1_list):.4f}")

    print("\nTraining final model on full CV data...")
    final_model = clone(base_model)
    final_model.fit(X, y)
    print("Done.")

    cv_metrics = {
        "CV AUC": np.mean(val_auc_list), "CV PR-AUC": np.mean(val_pr_list),
        "CV Precision": np.mean(val_precision_list), "CV Recall": np.mean(val_recall_list),
        "CV F1": np.mean(val_f1_list),
    }
    return final_model, cv_metrics

X_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_cv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

joblib.dump(X_cv.columns.tolist(), 'artifacts/model_columns.pkl')
print("Model columns saved. X_cv shape:", X_cv.shape)

xgb_model, cv_metrics = train_XGBoost_model(X_cv, y_cv)

# 5. Evaluation on the held-out test set — RESTORED (was missing from your last run)

def evaluate_model(model, X_test, y_test, custom_threshold=0.4):
    print("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_proba[y_test == 1], color='red', label='Class 1 (Positive)', stat='density', kde=True)
    sns.histplot(y_pred_proba[y_test == 0], color='blue', label='Class 0 (Negative)', stat='density', kde=True)
    plt.legend(); plt.xlabel("Predicted Probability"); plt.ylabel("Density")
    plt.title("Probability Distribution by Actual Class")
    plt.savefig("artifacts/eval_probability_distribution.png", dpi=110); plt.close()

    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.grid(True)
    plt.savefig("artifacts/eval_roc_curve.png", dpi=110); plt.close()

    print("\n--- Test Set Performance (0.5 Threshold) ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nClassification Report (0.5 Threshold):")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix (0.5 Threshold)'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig("artifacts/eval_confusion_matrix_05.png", dpi=110); plt.close()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    threshold_df = pd.DataFrame({'Threshold': thresholds, 'Precision': precision[:-1],
                                  'Recall': recall[:-1], 'F1-Score': f1_scores[:-1]})
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.grid(True)
    plt.savefig("artifacts/eval_pr_curve.png", dpi=110); plt.close()

    best_idx = threshold_df['F1-Score'].idxmax()
    best_threshold = threshold_df.loc[best_idx, 'Threshold']
    print(f"\nBest F1-Score ({threshold_df.loc[best_idx, 'F1-Score']:.4f}) at threshold: {best_threshold:.4f}")

    y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)
    print(f"\nClassification Report (Threshold = {custom_threshold}):")
    print(classification_report(y_test, y_pred_custom, zero_division=0))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        idx = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[idx], y=feature_names[idx])
        plt.title("Feature Importance")
        plt.savefig("artifacts/eval_feature_importance.png", dpi=110); plt.close()

    return {"auc": auc_score, "best_threshold": best_threshold, "threshold_df": threshold_df}

eval_results = evaluate_model(xgb_model, X_test, y_test, custom_threshold=0.4)

# 6. Precision/recall at fixed thresholds

def precision_recall_at_thresholds(model, X, y, feature_cols, start=0.15, end=0.80, step=0.01):
    y_proba = model.predict_proba(X[feature_cols])[:, 1]
    thresholds = np.arange(start, end + step, step)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y, y_pred, zero_division=0)
        r = recall_score(y, y_pred, zero_division=0)
        rows.append((t, p, r))
    return pd.DataFrame(rows, columns=["Threshold", "Precision", "Recall"])

pr_fixed_df = precision_recall_at_thresholds(xgb_model, X_test, y_test, X_test.columns, start=0.3, end=0.9, step=0.05)
pr_fixed_df.round(3)

pr_full = precision_recall_at_thresholds(xgb_model, X_test, y_test, X_test.columns,
                                          start=0.30, end=0.99, step=0.005)

candidates = pr_full[(pr_full['Precision'] >= 0.60) & (pr_full['Recall'] >= 0.85)]
print("Thresholds hitting precision >= 0.60 AND recall >= 0.85:")
print(candidates)

if candidates.empty:
    print("\nNo threshold satisfies both targets simultaneously with the current model.")
    print("Closest tradeoffs (sorted by recall, descending):")
    print(pr_full[pr_full['Precision'] >= 0.60].sort_values('Recall', ascending=False).head(10))

joblib.dump(xgb_model, "artifacts/xgb_model.pkl")
print("Model saved to artifacts/xgb_model.pkl")

# 7. Persist splits for downstream steps

joblib.dump(
    {
        "X_train": X_train, "y_train": y_train, "id_train": id_train,
        "X_val": X_val, "y_val": y_val, "id_val": id_val,
        "X_test": X_test, "y_test": y_test, "id_test": id_test,
        "num_features": num_features, "cat_features": cat_features,
        "encoded_cols": list(encoded_cols),
    },
    "artifacts/splits.pkl"
)
print("Splits + feature lists saved to artifacts/splits.pkl")
