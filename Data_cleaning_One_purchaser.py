# ============================================================
# Data Cleaning - One Purchaser
# Converted from the original Jupyter notebook
# ============================================================

# One purchaser — Cleaning & Feature Engineering
# Mirrors the two-purchaser cleaning notebook, adapted for the one-purchaser population.
# Key differences from the two-purchaser pipeline (read before running):
# - No `second_purchase_date` / `second_purchased_model` — tenure (`vehicle_age_1stbike`) is computed against `CUTOFF_DATE`, not a second purchase.
# - This notebook computes and saves the shared imputation constants (medians, outlier caps) to `cleaning_constants.json`, since the one-purchaser population is the large, representative one. The two-purchaser notebook should be updated to load these instead of computing its own — see the note in Section 9 below for exactly what to change there.
# - Every business-rule correction now writes a companion `..._was_corrected` flag column *before* overwriting the value, so corrected vs. observed values stay distinguishable.

# 0. Setup

import gc
gc.collect()

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os
os.makedirs('Data_files', exist_ok=True)

# CHANGE HERE: path to your one-purchaser export
df = pd.read_csv(r"one_purchase_data\one_purchase_data_oct25.csv")

# CONFIRM: this must be the TRUE max first_purchase_date in your export — see the cutoff-mismatch
# discussion from earlier. Set it explicitly here so every downstream tenure calc is consistent.
CUTOFF_DATE = pd.Timestamp("2025-10-31")  # CHANGE HERE

df.columns

print(df.shape)
df.head()

df.describe()

df.notna().sum() / df.shape[0]

df.info()

# 1. Drop columns not needed / not applicable to one-purchaser
# Mirrors the two-purchaser drop list, minus anything second-bike-specific (which doesn't exist here).

drop_cols = [
    'total_transactions', 'dealercode__c',
    'Frequency_softProduct', 'Frequency_apparel_store',
    'Frequency_service',
    'vehicle_age_days_1stbike','vehicle_age_1stbike',
    'Expected_Service_Visits_1stbike', 'Actual_Service_Visits_1stbike',
    'First Bike Category'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
df.columns

# 2. Change Datatypes

print(df['Final_ride_done'].value_counts(dropna=False))
df['Final_ride_done'] = df['Final_ride_done'].eq('Yes').astype(int)
print(df['Final_ride_done'].value_counts())

df.dtypes

mobile_cols = ['mobileno__c']

for col in mobile_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype('Int64')
            .astype('string')
        )

int_cols = [
    'isfinanced__c',
    'KM_before_2nd_bike',
    'NPS_Rating',
    'enquiry_count_between_purchases',
    'Frequency_gma', 'Frequency_apparel', 'product_count',
    'Final_ride_done'
]

for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

date_cols = [
    'first_purchase_date',
    'first_enquiry_date', 'second_enquiry_date',
    'last_purchase_date_service',
    'last_purchase_date_gma',
    'last_purchase_date_softProduct',
    'last_purchase_date_apparel',
    'last_purchase_date_apparel_store'
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Filter records from 2012 to oct 2025

df.shape

# Filter records from 2012 to October 2025 (inclusive)
df = df[
    df['first_purchase_date'] < '2025-10-01'
].copy()

df.shape

df.dtypes

# 4. Basic cleaning

# Missing primary key / chassis — these rows are unusable, not "duplicates" (see note from last review)
print("Rows with missing mobileno__c:", df['mobileno__c'].isna().sum())
print("Rows with missing first_chassis_number:", df['first_chassis_number'].isna().sum())
df.dropna(subset=['mobileno__c'], inplace=True)
df.dropna(subset=['first_chassis_number'], inplace=True)

# Explicit duplicate check on the stated primary key (this was missing before — added per review)
print("Duplicate mobileno__c after dropna:", df['mobileno__c'].duplicated().sum())

# Sentinel/placeholder date check
print(df[df['first_purchase_date'] == '1900-01-01'].shape)
df = df[df['first_purchase_date'] != '1900-01-01']

duplicate_chassis = df[
    df.duplicated(subset="first_chassis_number", keep=False)
].sort_values("first_chassis_number")
duplicate_chassis.head()

# Dedup by chassis, keep earliest first_purchase_date (same logic as two-purchaser)
df_sorted = df.sort_values(by=['first_chassis_number', 'first_purchase_date'], ascending=True)
df_cleaned = df_sorted.drop_duplicates(subset=['first_chassis_number'], keep='first')

print(df_sorted.shape)
print(df_cleaned.shape)

# Re-check PK uniqueness post-dedup
print("Duplicate mobileno__c in df_cleaned:", df_cleaned['mobileno__c'].duplicated().sum())

# Cross-check: same mobile number attached to multiple distinct chassis numbers?
multi_chassis = df_cleaned.groupby('mobileno__c')['first_chassis_number'].nunique()
print("mobileno__c values linked to >1 distinct first_chassis_number:", (multi_chassis > 1).sum())

# Also drop the exact overlap customers found earlier (present in BOTH one- and two-purchaser exports)
# CHANGE HERE: point this at the overlap diagnostic CSV generated during EDA, if you still want to exclude them
# overlap_ids = pd.read_csv("00_overlap_customers_diagnostic.csv")['mobileno__c'].astype(str).tolist()
# df_cleaned = df_cleaned[~df_cleaned['mobileno__c'].isin(overlap_ids)].copy()
# print(f"Rows after removing known two-purchaser overlaps: {len(df_cleaned):,}")

# 5. Model name → model family

def clean_model(text):
    if pd.isna(text):
        return np.nan
    text = str(text).upper().strip()
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# NOTE: no 'second_purchased_model' here — one-purchaser has no second bike.
# 'second_enquired_model' DOES exist (someone can enquire about a 2nd bike without buying it).
model_cols = [
    'first_purchased_model',
    'first_enquired_model',
    'second_enquired_model',
]

for col in model_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].apply(clean_model)

def map_model(model):
    if pd.isna(model):
        return np.nan
    model = model.upper()

    # -------- 650 CC --------
    if 'CLASSIC' in model and '650' in model: return 'Classic 650'
    if 'SHOTGUN' in model: return 'Shotgun 650'
    if 'SUPER METEOR' in model: return 'Super Meteor 650'
    if 'INTERCEPTOR' in model or 'INT650' in model: return 'Interceptor 650'
    if 'CONTINENTAL' in model: return 'Continental GT 650'

    # -------- 500 CC --------
    if 'CLASSIC' in model and '500' in model: return 'Classic 500'
    if 'BULLET' in model and '500' in model: return 'Bullet 500'
    if 'THUNDERBIRD' in model and '500' in model: return 'Thunderbird 500'
    if 'LIGHTNING 535' in model or 'LIGHTNING 500' in model: return 'Lightning'
    if 'CL EFI 500' in model: return 'CL EFI 500'

    # -------- ADV --------
    if 'HIMALAYAN' in model: return 'Himalayan'
    if 'SCRAM' in model: return 'Scram 411'   # FIX applied per prior review: matches category_map key below
    if 'GUERRILLA' in model: return 'Guerrilla'
    if 'U4J' in model: return 'U4J'
    if 'CKDKIT P1E' in model: return 'CKDKIT P1E'

    # -------- 350 CC --------
    if 'METEOR' in model: return 'Meteor 350'
    if 'HUNTER' in model or 'HNTR' in model: return 'Hunter 350'
    if 'THUNDERBIRD' in model: return 'Thunderbird 350'
    if 'ELECTRA' in model: return 'Electra 350'
    if 'BULLET' in model: return 'Bullet 350'
    if 'CLASSIC' in model: return 'Classic 350'
    if 'RUMBLER' in model: return 'Rumbler'
    if 'SIGNALS 350' in model: return 'SIGNALS 350'

    return 'Other'

df_cleaned['first_purchased_model_mapped'] = df_cleaned['first_purchased_model'].apply(map_model)
if 'first_enquired_model' in df_cleaned.columns:
    df_cleaned['first_enquired_model_mapped'] = df_cleaned['first_enquired_model'].apply(map_model)
if 'second_enquired_model' in df_cleaned.columns:
    df_cleaned['second_enquired_model_mapped'] = df_cleaned['second_enquired_model'].apply(map_model)

df_cleaned['first_purchased_model_mapped'].value_counts()

# What's still falling through to 'Other'? Worth a periodic check as new models launch.
df_cleaned.loc[
    df_cleaned['first_purchased_model_mapped'] == 'Other', 'first_purchased_model'
].value_counts().head(20)

# 6. Model category
# CONFIRM per prior review: added `'Classic 650': 'Premium'`; `'Scram 411'` key now matches `map_model`'s output exactly (was a silent mismatch before). `Guerrilla`, `Rumbler`, `U4J`, `CKDKIT P1E`, `SIGNALS 350`, `Lightning`, `CL EFI 500` still fall to `'Other'` — tell me the right bucket if these aren't negligible volume.

category_map = {
    # -------- City Commuter --------
    'Classic 350': 'City Commuter',
    'Bullet 350': 'City Commuter',
    'Meteor 350': 'City Commuter',
    'Hunter 350': 'City Commuter',
    'Thunderbird 350': 'City Commuter',
    'Electra 350': 'City Commuter',

    # -------- Adventure --------
    'Himalayan': 'Adventure',
    'Scram 411': 'Adventure',

    # -------- Premium --------
    'Super Meteor 650': 'Premium',
    'Continental GT 650': 'Premium',
    'Interceptor 650': 'Premium',
    'Shotgun 650': 'Premium',
    'Classic 500': 'Premium',
    'Bullet 500': 'Premium',
    'Thunderbird 500': 'Premium',
    'Classic 650': 'Premium',
}

df_cleaned['first_bike_category'] = df_cleaned['first_purchased_model_mapped'].map(category_map)
df_cleaned['first_bike_category'] = df_cleaned['first_bike_category'].fillna('Other')

if 'first_enquired_model_mapped' in df_cleaned.columns:
    df_cleaned['first_enquired_bike_category'] = df_cleaned['first_enquired_model_mapped'].map(category_map).fillna('Other')
if 'second_enquired_model_mapped' in df_cleaned.columns:
    df_cleaned['second_enquired_bike_category'] = df_cleaned['second_enquired_model_mapped'].map(category_map).fillna('Other')

# 7. Engine type (UCE vs J-series)
# Fix applied per prior review: models that map to `'Other'` now return `'Unknown'` instead of silently defaulting to `'J-series'`.

def classify_engine(row, model_col, date_col):
    model = row[model_col]
    date = pd.to_datetime(row[date_col], errors='coerce')

    if pd.isna(date):
        return 'Unknown'

    classic_cutoff = pd.Timestamp('2021-09-30')
    bullet_cutoff = pd.Timestamp('2023-09-30')

    if model in ['Classic 350', 'Classic 500']:
        return 'UCE' if date < classic_cutoff else 'J-series'

    if model in ['Bullet 350', 'Bullet 500', 'Electra 350']:
        return 'UCE' if date < bullet_cutoff else 'J-series'

    if model in ['SIGNALS 350', 'Lightning', 'CKDKIT P1E', 'U4J', 'CL EFI 500',
                 'Thunderbird 350', 'Thunderbird 500', 'Rumbler']:
        return 'UCE'

    if model == 'Other' or pd.isna(model):
        return 'Unknown'   # FIX: was silently defaulting to 'J-series'

    return 'J-series'

df_cleaned['First_purchased_model(UCE_or_J-series)'] = df_cleaned.apply(
    lambda r: classify_engine(r, 'first_purchased_model_mapped', 'first_purchase_date'), axis=1
)
df_cleaned['First_purchased_model(UCE_or_J-series)'].value_counts(dropna=False)

# 8. State tier

tier1 = ['MAHARASHTRA', 'KARNATAKA', 'TAMIL NADU', 'GUJARAT', 'TELANGANA', 'DELHI', 'NEW DELHI','MUMBAI']
tier2 = ['ANDHRA PRADESH', 'UTTAR PRADESH', 'WEST BENGAL', 'HARYANA', 'PUNJAB',
         'RAJASTHAN', 'KERALA', 'ODISHA', 'MADHYA PRADESH', 'CHHATTISGARH',
         'GOA', 'UTTARAKHAND', 'CHANDIGARH', 'PUDUCHERRY', 'DAMAN & DIU',
         'DADRA & NAGAR HAVELI', 'DADRA & NAGAR HAVELI AND DAMAN AND DIU']
tier3 = ['BIHAR', 'JHARKHAND', 'JAMMU & KASHMIR', 'HIMACHAL PRADESH', 'ASSAM', 'SIKKIM',
         'MEGHALAYA', 'MANIPUR', 'NAGALAND', 'MIZORAM', 'ARUNACHAL PRADESH', 'TRIPURA',
         'LADAKH', 'ANDAMAN & NICOBAR ISLANDS', 'LAKSHADWEEP']

def map_tier(state):
    if state in tier1: return "Tier 1"
    if state in tier2: return "Tier 2"
    if state in tier3: return "Tier 3"
    return "Unknown"

df_cleaned["state_clean"] = df_cleaned["state__c"].astype(str).str.upper().str.strip()
df_cleaned["state_clean"] = df_cleaned["state_clean"].replace({"NEW DELHI": "DELHI"})
df_cleaned["State"] = df_cleaned["state_clean"].apply(map_tier)

print(df_cleaned.loc[df_cleaned["State"] == "Unknown", "state_clean"].value_counts(dropna=False))

# 9. Vehicle age — the critical difference from the two-purchaser notebook
# Two-purchaser used `second_purchase_date − first_purchase_date`. There is no second purchase here, so tenure is measured against the confirmed `CUTOFF_DATE` instead. Get this cell right — every downstream ratio (km/year, services/year, Expected_Service_Visits, Ratio Actual/Expected) depends on it.

df_cleaned['vehicle_age_days_1stbike'] = (CUTOFF_DATE - df_cleaned['first_purchase_date']).dt.days
df_cleaned['vehicle_age_1stbike'] = (df_cleaned['vehicle_age_days_1stbike'] / 365.25).round(2)

# Guard: should never be negative if CUTOFF_DATE is set correctly. If this prints >0, CUTOFF_DATE is wrong.
print("Rows with negative tenure (CUTOFF_DATE is likely wrong if >0):", (df_cleaned['vehicle_age_1stbike'] < 0).sum())

df_cleaned['vehicle_age_1stbike'].value_counts().sort_index()

df_cleaned.drop(columns=[c for c in ['first_purchased_model', 'first_enquired_model', 'second_enquired_model']
                          if c in df_cleaned.columns], inplace=True)

df_cleaned.columns

# 10. Data validation & correction — KM, service count, service amount
# Same domain rules as the two-purchaser notebook, but every correction now writes a `..._was_corrected` flag *before* overwriting, so the model and any later RCA can tell real values from estimated ones.
# This notebook also computes the shared constants (medians, outlier caps) used here — they get saved to `cleaning_constants.json` at the end so the two-purchaser notebook can reuse the exact same numbers instead of computing its own (see Section 13 for why, and the note at the bottom of this notebook for what to change in the two-purchaser file).

df_cleaned["km_per_year"] = (
    df_cleaned["KM_before_2nd_bike"] / df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
)
df_cleaned["services_per_year"] = (
    df_cleaned["service_count"] / df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
)
df_cleaned["cost_per_service"] = (
    df_cleaned["total_service_amount"] / df_cleaned["service_count"].replace(0, np.nan)
)

print(df_cleaned["km_per_year"].describe())
print(df_cleaned["services_per_year"].describe())
print(df_cleaned["cost_per_service"].describe())

# 10a. Old bike but zero KM

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) & (df_cleaned["KM_before_2nd_bike"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['KM_before_2nd_bike','vehicle_age_1stbike']].head())

mask = (
    (df_cleaned["vehicle_age_1stbike"] > 5) & (df_cleaned["KM_before_2nd_bike"].fillna(0) == 0)
)

print(mask.dtype)
print(mask.value_counts(dropna=False))

# NEW: flag BEFORE overwriting
df_cleaned["km_was_corrected"] = mask.astype(int)

median_km_year = df_cleaned.loc[df_cleaned["km_per_year"] > 0, "km_per_year"].median()

df_cleaned.loc[mask, "KM_before_2nd_bike"] = (
    df_cleaned.loc[mask, "vehicle_age_1stbike"] * median_km_year
).round()

print("Corrected rows:", mask.sum(), f"({mask.mean()*100:.2f}% of dataset)")

# 10b. Old bike but zero services

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) & (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)

mask = (
    (df_cleaned["vehicle_age_1stbike"] > 5) & (df_cleaned["service_count"] == 0)
)

# NEW: flag BEFORE overwriting. Combine with the KM flag if a row was already flagged for that.
df_cleaned["service_count_was_corrected"] = mask.astype(int)

median_services_per_year = df_cleaned.loc[df_cleaned["services_per_year"] > 0, "services_per_year"].median()
print("median_services_per_year:", median_services_per_year)

df_cleaned.loc[mask, "service_count"] = (
    df_cleaned.loc[mask, "vehicle_age_1stbike"] * median_services_per_year
).round().astype(int)

print("Corrected rows:", mask.sum(), f"({mask.mean()*100:.2f}% of dataset)")

# 10c. High KM but zero service count

records_to_check = df_cleaned[
    (df_cleaned["KM_before_2nd_bike"] > 100000) & (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)

median_km_per_service = (
    df_cleaned.loc[df_cleaned["service_count"] > 0, "KM_before_2nd_bike"] /
    df_cleaned.loc[df_cleaned["service_count"] > 0, "service_count"]
).median()

print("median km_per_service:", median_km_per_service)

mask = (
    (df_cleaned["KM_before_2nd_bike"].fillna(0) > 100000) & (df_cleaned["service_count"] == 0)
)

# NEW: OR this into the existing flag rather than overwrite it, since a row could be
# touched by more than one correction rule.
df_cleaned["service_count_was_corrected"] = (
    (df_cleaned["service_count_was_corrected"] == 1) | mask
).astype(int)

df_cleaned.loc[mask, "service_count"] = (
    df_cleaned.loc[mask, "KM_before_2nd_bike"] / median_km_per_service
).round().astype(int)

print("Corrected rows (this rule):", mask.sum())

# 10d. High service count but zero amount

records_to_check = df_cleaned[
    (df_cleaned["service_count"] >= 5) & (df_cleaned["total_service_amount"] == 0)
]
print(records_to_check.shape)

median_cost_per_service = (
    df_cleaned.loc[
        (df_cleaned["service_count"] > 0) & (df_cleaned["total_service_amount"] > 0),
        "cost_per_service"
    ]
).median()
print("median_cost_per_service:", median_cost_per_service)

mask = (
    (df_cleaned["service_count"] >= 5) & (df_cleaned["total_service_amount"] == 0)
)

# NEW: flag before overwrite
df_cleaned["service_amount_was_corrected"] = mask.astype(int)

df_cleaned.loc[mask, "total_service_amount"] = (
    df_cleaned.loc[mask, "service_count"] * median_cost_per_service
)

print("Corrected rows:", mask.sum())

# 10E. Zero services but non-zero amount

records_to_check = df_cleaned[
    (df_cleaned["service_count"] == 0) &
    (df_cleaned["total_service_amount"] > 0)
]
print(records_to_check.shape)
print(records_to_check[['service_count','total_service_amount']].head())

# mask = (
#     (df_cleaned["service_count"] == 0) &
#     (df_cleaned["total_service_amount"] > 0)
# )

# df_cleaned.loc[mask, "service_count"] = (
#     df_cleaned.loc[mask, "total_service_amount"] /
#     median_cost_per_service
# ).round().astype(int)

# records_to_check = df_cleaned[
#     (df_cleaned["service_count"] == 0) &
#     (df_cleaned["total_service_amount"] > 0)
# ]
# print(records_to_check.shape)
# print(records_to_check[['service_count','total_service_amount']].head())

df_cleaned["km_per_year"] = (
    df_cleaned["KM_before_2nd_bike"] /
    df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
)

df_cleaned["services_per_year"] = (
    df_cleaned["service_count"] /
    df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
)

df_cleaned["cost_per_service"] = (
    df_cleaned["total_service_amount"] /
    df_cleaned["service_count"].replace(0, np.nan)
)
print(df_cleaned['km_per_year'].describe())
print(df_cleaned['services_per_year'].describe())
print(df_cleaned['cost_per_service'].describe())

# 11. Missing value treatment

df_cleaned.notna().sum() / df_cleaned.shape[0]

# Structural zero-fills — NaN genuinely means "never happened". Safe to fill the same way
# regardless of which dataset it's computed on (see explanation in chat).
cols_zero_fill = [c for c in [
    'Frequency_gma', 'Frequency_apparel', 'product_count',
    'isfinanced__c', 'enquiry_count_between_purchases'
] if c in df_cleaned.columns]

df_cleaned[cols_zero_fill] = df_cleaned[cols_zero_fill].fillna(0)

if 'Segment' in df_cleaned.columns:
    df_cleaned['Segment'] = df_cleaned['Segment'].fillna('Unknown')
df_cleaned['State'] = df_cleaned['State'].fillna('Unknown')

# NPS_Rating: ~98% missing. Per review — add an explicit has-rating flag rather than relying on the
# sentinel value alone, since a tree model could otherwise treat -1 as an ordinal extreme rather than
# "no response given".
# df_cleaned['has_NPS_rating'] = df_cleaned['NPS_Rating'].notna().astype(int)
df_cleaned['NPS_Rating'] = df_cleaned['NPS_Rating'].fillna(-1)

# Remaining total_service_amount gaps: use the SAME median_cost_per_service computed in 10d above,
# rather than a separate hardcoded constant — this reconciles the "cell 68 vs 84" inconsistency flagged
# in the last review (two different per-service cost assumptions for the same kind of gap).
mask = df_cleaned['total_service_amount'].isna()
df_cleaned['service_amount_was_corrected'] = (
    (df_cleaned['service_amount_was_corrected'] == 1) | mask
).astype(int)

df_cleaned.loc[mask, 'total_service_amount'] = df_cleaned.loc[mask, 'service_count'] * median_cost_per_service

# Remaining KM gaps
mask = df_cleaned['KM_before_2nd_bike'].isna()
df_cleaned['km_was_corrected'] = (
    (df_cleaned['km_was_corrected'] == 1) | mask
).astype(int)
df_cleaned.loc[mask, 'KM_before_2nd_bike'] = (
    df_cleaned.loc[mask, 'service_count'] * median_km_per_service
).round()

# 12. Cap extreme outliers
# These caps are computed HERE (one-purchaser = large, representative population) and saved to `cleaning_constants.json` — the two-purchaser notebook will load and reuse these exact values rather than computing its own (see the closing note).

cols = ["KM_before_2nd_bike", "service_count", "total_service_amount", "Frequency_gma"]
df_cleaned[cols].describe(percentiles=[0.90, 0.95, 0.99, 0.995, 0.999])

col = ["enquiry_count_between_purchases", "total_cdp_amount", "total_gma_amount", "Frequency_gma","Frequency_apparel","Monetary_apparel"]
df_cleaned[col].describe(percentiles=[0.90, 0.95, 0.99, 0.995, 0.999])

col=['total_cdp_amount']
df_cleaned[col].describe(
    percentiles=[
        0.001,
        0.005,
        0.01,
        0.05,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999
    ]
)

# Check negative values
negative_cdp = df_cleaned[
    df_cleaned["total_cdp_amount"] < 0
]

print("Total negative records:", len(negative_cdp))

print(
    "Percentage negative:",
    len(negative_cdp) / len(df_cleaned) * 100
)

# Check extremely negative values
print(
    "Below -1750:",
    (df_cleaned["total_cdp_amount"] < -1750).sum()
)

# Display the most negative records
print(
    df_cleaned[
        ["total_cdp_amount"]
    ]
    .sort_values("total_cdp_amount")
    .head(20)
)

# ============================================================
# 1. Define columns for each capping percentile
# ============================================================

# Columns to cap at 99.5 percentile
cols_995 = [
    "KM_before_2nd_bike",
    "service_count",
    "total_service_amount",
    "Frequency_gma"
]

# Columns to cap at 99.9 percentile
cols_999 = [
    "enquiry_count_between_purchases",
    "total_cdp_amount",
    "total_gma_amount",
    "Frequency_apparel",
    "Monetary_apparel"
]

# ============================================================
# 2. Create one dictionary to store ALL caps
# ============================================================
outlier_caps = {}
# ============================================================
# 3. Function to calculate and apply cap
# ============================================================

def apply_outlier_cap(df, columns, percentile, dataset_name):

    for c in columns:

        if c not in df.columns:
            print(f"WARNING: {c} not found in {dataset_name}")
            continue

        # Calculate cap
        if pd.api.types.is_integer_dtype(df[c]):
            cap = df[c].quantile(
                percentile,
                interpolation="nearest"
            )
        else:
            cap = df[c].quantile(percentile)

        print(
            f"{dataset_name} -> {c}: "
            f"{percentile * 100:.1f}th percentile cap = {cap}"
        )

        # Save cap in SAME dictionary
        outlier_caps[f"{dataset_name}_{c}"] = float(cap)

        # Apply cap
        df[c] = df[c].clip(upper=cap)

# ============================================================
# 4. Apply 99.5% capping to df_cleaned
# ============================================================

apply_outlier_cap(
    df=df_cleaned,
    columns=cols_995,
    percentile=0.995,
    dataset_name="df_cleaned"
)

# ============================================================
# 5. Apply 99.9% capping to df
# ============================================================

apply_outlier_cap(
    df=df_cleaned,
    columns=cols_999,
    percentile=0.999,
    dataset_name="df_cleaned"
)

negative_count = (df_cleaned["total_cdp_amount"] < 0).sum()

print("Negative records before correction:", negative_count)

df_cleaned["total_cdp_amount"] = df_cleaned[
    "total_cdp_amount"
].clip(lower=0)

# 13. Final derived features

# 13.a Rebuild `total_amount_spent`
# Drop the raw field, reconstruct from components (price-by-model + service + cdp + gma + mag + apparel).

if 'total_amount_spent' in df_cleaned.columns:
    df_cleaned.drop(columns=['total_amount_spent'], inplace=True)

def calculate_price(model_name):
    if pd.isna(model_name): return 150000
    model = str(model_name).upper()
    if 'SUPER METEOR 650' in model: return 348900
    elif 'THUNDERBIRD 500' in model: return 214421
    elif 'NEW CLASSIC 350' in model: return 199499
    elif 'HIMALAYAN' in model and 'NEW' not in model: return 285000
    elif 'NEW HIMALAYAN' in model: return 350000
    elif 'BULLET 350' in model and 'NEW' not in model: return 173562
    elif 'NEW BULLET' in model: return 173562
    elif 'BULLET 500' in model: return 188803
    elif 'BULLET X' in model: return 188803
    elif 'BULLET ELECTRA' in model: return 169000
    elif 'THUNDERBIRD X 350' in model: return 163842
    elif 'THUNDERBIRD X 500' in model: return 206747
    elif 'THUNDERBIRD 350' in model and 'X' not in model: return 156496
    elif 'THUNDERBIRD 500' in model and 'X' not in model: return 206747
    elif 'METEOR 350' in model: return 205528
    elif 'GUERRILLA' in model: return 239000
    elif 'CONTINENTAL GT' in model or 'CONTINENTALGT' in model: return 318418
    elif 'INTERCEPTOR' in model: return 302418
    elif 'CLASSIC 350' in model and 'NEW' not in model: return 178000
    elif 'HUNTER 350' in model: return 149900
    elif 'SHOTGUN' in model: return 359430
    elif 'CLASSIC 500' in model: return 162000
    elif 'BEAR 650' in model: return 343000
    elif 'CLASSIC 650' in model: return 337000
    elif 'SCRAM' in model: return 215000
    elif 'BOBBER' in model: return 180000
    elif 'SIGNALS' in model: return 164095
    else: return 150000

df_cleaned['first_bike_price'] = df_cleaned['first_purchased_model_mapped'].apply(calculate_price)

cols = ['first_bike_price', 'total_service_amount', 'total_cdp_amount',
        'total_gma_amount', 'total_mag_amount', 'Monetary_apparel']
for c in cols:
    if c not in df_cleaned.columns:
        df_cleaned[c] = 0
df_cleaned[cols] = df_cleaned[cols].fillna(0)

df_cleaned['total_amount_spent'] = df_cleaned[cols].sum(axis=1)

df_cleaned.drop(columns=['total_cdp_amount',
                          'total_gma_amount', 'total_mag_amount', 'Monetary_apparel',
                          'state__c', 'state_clean'], inplace=True)

# 13.b Last activity date

date_cols = [
    'first_enquiry_date', 'second_enquiry_date',
    'last_purchase_date_service', 'last_purchase_date_gma',
    'last_purchase_date_softProduct', 'last_purchase_date_apparel',
    'last_purchase_date_apparel_store',
]
date_cols = [c for c in date_cols if c in df_cleaned.columns]

df_cleaned["last_activity_date"] = df_cleaned[date_cols].max(axis=1)

drop_cols = [
    'last_purchase_date_gma',
    'last_purchase_date_softProduct', 'last_purchase_date_apparel',
    'last_purchase_date_apparel_store',
    'km_per_year', 'services_per_year', 'cost_per_service',
]
df_cleaned.drop(columns=[c for c in drop_cols if c in df_cleaned.columns], inplace=True)

# 13.c Amount per service

df_cleaned['amount_per_service'] = np.where(
    df_cleaned['service_count'] > 0,
    df_cleaned['total_service_amount'].fillna(0) / df_cleaned['service_count'].fillna(0),
    0
)

# 13.d Ratio(Actual service visits/ Expected Service visits)

df_cleaned['Expected_Service_Visits_1stbike'] = (df_cleaned['vehicle_age_1stbike'] * 2 + 2).astype('int64')

# Renamed from 'Ratio ( Actual/ Expected)' per review — spaces/parentheses in column names
# can break some modeling libraries (XGBoost/LightGBM feature-name validation).
df_cleaned['ratio_actual_expected_service'] = np.where(
    df_cleaned['Expected_Service_Visits_1stbike'] == 0, 0,
    df_cleaned['service_count'].astype(float) / df_cleaned['Expected_Service_Visits_1stbike'].astype(float)
)

# 13.e  Other feature engineering

df_cleaned['reference_date'] = CUTOFF_DATE
# CHANGE HERE if your festival-season definition differs
FESTIVAL_MONTHS = [9, 10, 11]

def engineer_features(df, km_col):
    df = df.copy()
    ref = df['reference_date']

    # ---- Recency ----
    enquiry_dates = df[['first_enquiry_date', 'second_enquiry_date']]
    df['days_since_last_enquiry'] = (ref - enquiry_dates.max(axis=1)).dt.days
    df['days_since_last_activity'] = (ref - df['last_activity_date']).dt.days
    df['days_since_last_service'] = (ref - df['last_purchase_date_service']).dt.days
    # df['recency_ratio_service'] = np.where(
    #     df['vehicle_age_days_1stbike'] > 0,
    #     df['days_since_last_service'] / df['vehicle_age_days_1stbike'],
    #     np.nan
    # )

    # ---- Enquiry funnel ----
    df['purchase_to_enquiry_for_2nd_bike'] = (df['first_enquiry_date'] - df['first_purchase_date']).dt.days
    df['has_second_enquiry'] = df['second_enquiry_date'].notna().astype(int)
    # df['enquiries_per_year'] = np.where(
    #     df['vehicle_age_1stbike'] > 0,
    #     df['enquiry_count_between_purchases'] / df['vehicle_age_1stbike'],
    #     0
    # )

    # ---- Engagement composite ----
    engagement_cols = [c for c in
                        ['Frequency_gma', 'Frequency_apparel', 'product_count']
                        if c in df.columns]
    df['product_engagement_score'] = df[engagement_cols].fillna(0).sum(axis=1)

    # ---- Financial intensity ----
    # df['first_bike_price'] = df['first_purchased_model_mapped'].apply(calculate_price)
    # df['spend_per_year'] = np.where(
    #     df['vehicle_age_1stbike'] > 0,
    #     df['total_amount_spent'] / df['vehicle_age_1stbike'],
    #     df['total_amount_spent']
    # )

    df['km_per_year'] = np.where(
        df['vehicle_age_1stbike'] > 0,
        df[km_col] / df['vehicle_age_1stbike'],
        np.nan
    )

    # ---- Seasonality ----
    df['first_purchase_month'] = df['first_purchase_date'].dt.month
    df['first_purchase_quarter'] = df['first_purchase_date'].dt.quarter
    df['purchased_in_festival_season'] = df['first_purchase_month'].isin(FESTIVAL_MONTHS).astype(int)

    # ---- Service health ----
    expected_gap_days = np.where(
        df['Expected_Service_Visits_1stbike'] > 0,
        df['vehicle_age_days_1stbike'] / df['Expected_Service_Visits_1stbike'],
        np.nan
    )
    df['is_service_overdue'] = (df['days_since_last_service'] > expected_gap_days * 1.5).astype(int)

    return df

df_cleaned = engineer_features(df_cleaned, km_col='KM_before_2nd_bike')   # CHANGE HERE if your column name differs

print("one_feat:", df_cleaned.shape)

new_cols = [
    'days_since_last_enquiry', 'days_since_last_service',
    'purchase_to_enquiry_for_2nd_bike', 'has_second_enquiry',
    'product_engagement_score', 'km_per_year',
    'first_purchase_month', 'first_purchase_quarter', 'purchased_in_festival_season',
    'is_service_overdue'
]
df_cleaned[new_cols].describe(include='all').T

# 15. Save cleaned constants for reuse in the two-purchaser notebook and future scoring data
# This is the file the two-purchaser notebook (and later, the live scoring pipeline) should load instead of recomputing its own medians/caps — see the closing note below for exact code to paste into the two-purchaser notebook.

cleaning_constants = {
    "CUTOFF_DATE": CUTOFF_DATE.strftime("%Y-%m-%d"),
    "median_km_per_year": float(median_km_year),
    "median_services_per_year": float(median_services_per_year),
    "median_km_per_service": float(median_km_per_service),
    "median_cost_per_service": float(median_cost_per_service),
    "outlier_caps": outlier_caps,   # {"KM_before_2nd_bike": ..., "service_count": ..., "total_service_amount": ...}
}

with open("cleaning_constants.json", "w") as f:
    json.dump(cleaning_constants, f, indent=2)

cleaning_constants

df_cleaned.dtypes

df_cleaned.columns

df_cleaned.to_csv("Data_files/one_purchaser_cleaned_feature_engineered.csv", index=False)

df_cleaned.shape

df_cleaned.columns

# What to change in the two-purchaser notebook to use these saved values
# Add near the top, after imports:
# ```python
# import json
# with open("cleaning_constants.json") as f:
# C = json.load(f)
# ```
# Then replace each self-computed constant with the loaded one:
# | Two-purchaser notebook (old) | Replace with |
# |---|---|
# | `median_km_year = df_cleaned.loc[df_cleaned["km_per_year"] > 0, "km_per_year"].median()` (cell 57) | `median_km_year = C["median_km_per_year"]` |
# | `median_services_per_year = ...median()` (cell 62) | `median_services_per_year = C["median_services_per_year"]` |
# | `km_per_service = ...median()` (cell 71) | `km_per_service = C["median_km_per_service"]` |
# | `median_cost = ...median()` (cell 68) | `median_cost_per_service = C["median_cost_per_service"]` (also use this same variable in cell 84's fill, instead of the hardcoded `2500`, to resolve the two-different-constants inconsistency flagged earlier) |
# | `cap = df_cleaned[col].quantile(0.995, ...)` (cell 91, computed per column) | `cap = C["outlier_caps"][col]` |
# Also add the same `..._was_corrected` flag lines immediately before each `.loc[mask, ...] = ...` overwrite in the two-purchaser notebook (cells 57, 62, 68, 71), exactly as done in Section 10 above, so both datasets carry the same audit columns before you concatenate them for training.
# One nuance for the two-purchaser notebook specifically: its `vehicle_age_1stbike` is still `second_purchase_date − first_purchase_date` (that stays as-is, don't change it to `CUTOFF_DATE` — only the one-purchaser notebook uses the cutoff). The constants above are still valid to reuse for it, since they're per-year/per-service *rates*, not absolute values tied to a specific date.
