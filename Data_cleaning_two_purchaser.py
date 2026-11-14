# ============================================================
# Data Cleaning - Two Purchaser
# Converted from the original Jupyter notebook
# ============================================================

# Two purchaser

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

with open("cleaning_constants.json") as f:
    C = json.load(f)

df=pd.read_csv(r"two_purchase_data_101 (1).csv")

df.shape

df.columns

df.head()

df.describe()

df.notna().sum()/df.shape[0]

df.info()

# 1. drop columns not needed / not applicable to one-purchaser

df.drop(columns=['second_chassis_number','total_transactions','dealercode__c',
                 'Frequency_softProduct','Frequency_apparel_store',
                 'vehicle_age_days_1stbike', 'vehicle_age_1stbike',
                 'vehicle_age_days_2ndbike','vehicle_age_2ndbike',
                 'Expected_Service_Visits_1stbike','Actual_Service_Visits_1stbike',
                 'Expected_Service_Visits_2ndbike','Actual_Service_Visits_2ndbike',
                 'Second Bike Category',
                 'Frequency_service'], inplace=True)

df.columns

df.head()

# 2. Change Datatype

print(df['Final_ride_done'].value_counts())
df['Final_ride_done'] = df['Final_ride_done'].eq('Yes').astype(int)
print(df['Final_ride_done'].value_counts())

mobile_cols = ['mobileno__c']

for col in mobile_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype('Int64')      # Removes .0 while preserving NaN
            .astype('string')     # Convert to string
        )

int_cols = [
    'isfinanced__c',
    'KM_before_2nd_bike',
    'NPS_Rating',
    'enquiry_count_between_purchases',
    'Frequency_gma','Frequency_apparel','product_count',
    'Final_ride_done'
]

for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

date_cols = [
    'first_purchase_date', 'second_purchase_date',
    'first_enquiry_date', 'second_enquiry_date',
    'last_purchase_date_service',
    'last_purchase_date_gma',
    'last_purchase_date_softProduct',
    'last_purchase_date_apparel',
    'last_purchase_date_apparel_store',
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df.dtypes

# 3. Basic Cleaning

# Missing primary key / chassis — these rows are unusable, not "duplicates" (see note from last review)
print("Rows with missing mobileno__c:", df['mobileno__c'].isna().sum())
print("Rows with missing first_chassis_number:", df['first_chassis_number'].isna().sum())

df.dropna(subset=['mobileno__c'],inplace=True) # remove duplicate mobile numbers
df.dropna(subset=['first_chassis_number'],inplace=True) # remove duplicate chassis numbers

# Explicit duplicate check on the stated primary key (this was missing before — added per review)
print("Duplicate mobileno__c after dropna:", df['mobileno__c'].duplicated().sum())
print("Duplicate first_chassis_number after dropna:", df['first_chassis_number'].duplicated().sum())

print(df[df['first_purchase_date']=='1900-01-01'])
# Filtering the dataframe to keep everything EXCEPT the target date
df = df[df['first_purchase_date'] != '1900-01-01']

# This puts the MIN date (earliest) at the top for each chassis
df_sorted = df.sort_values(by=['first_chassis_number', 'first_purchase_date'], ascending=True)

# 3. Drop duplicates, keeping only the 'first' record (which is the min date)
df_cleaned = df_sorted.drop_duplicates(subset=['first_chassis_number'], keep='first')

# Now df_cleaned contains only the records with the earliest date per chassis
print(df_sorted.shape)
print(df_cleaned.shape)

# Re-check PK uniqueness post-dedup
print("Duplicate mobileno__c in df_cleaned:", df_cleaned['mobileno__c'].duplicated().sum())

# Cross-check: same mobile number attached to multiple distinct chassis numbers?
multi_chassis = df_cleaned.groupby('mobileno__c')['first_chassis_number'].nunique()
print("mobileno__c values linked to >1 distinct first_chassis_number:", (multi_chassis > 1).sum())

df_cleaned.columns

# 4. Model name into model family

def clean_model(text):

    # Preserve missing values
    if pd.isna(text):
        return np.nan

    text = str(text).upper().strip()

    # Remove special characters (keep letters, numbers and spaces)
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

model_cols = [
    'first_purchased_model',
    'second_purchased_model',
    'first_enquired_model',
    'second_enquired_model'
]

for col in model_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].apply(clean_model)

df_cleaned.dtypes

def map_model(model):

    # Preserve missing values
    if pd.isna(model):
        return np.nan

    model = model.upper()

    # -------- 650 CC --------
    if 'CLASSIC' in model and '650' in model:
        return 'Classic 650'
    if 'SHOTGUN' in model:
        return 'Shotgun 650'
    if 'SUPER METEOR' in model:
        return 'Super Meteor 650'
    if 'INTERCEPTOR' in model or 'INT650' in model:
        return 'Interceptor 650'
    if 'CONTINENTAL' in model:
        return 'Continental GT 650'

    # -------- 500 CC --------
    if 'CLASSIC' in model and '500' in model:
        return 'Classic 500'
    if 'BULLET' in model and '500' in model:
        return 'Bullet 500'
    if 'THUNDERBIRD' in model and '500' in model:
        return 'Thunderbird 500'
    if 'LIGHTNING 535' in model or 'LIGHTNING 500' in model:
        return 'Lightning'
    if 'CL EFI 500' in model:
        return 'CL EFI 500'

    # -------- ADV --------
    if 'HIMALAYAN' in model:
        return 'Himalayan'
    if 'SCRAM' in model:
        return 'Scram'
    if 'GUERRILLA' in model:
        return 'Guerrilla'
    if 'U4J' in model:
        return 'U4J'
    if 'CKDKIT P1E' in model:
        return 'CKDKIT P1E'

    # -------- 350 CC --------
    if 'METEOR' in model:
        return 'Meteor 350'
    if 'HUNTER' in model or 'HNTR' in model:
        return 'Hunter 350'
    if 'THUNDERBIRD' in model:
        return 'Thunderbird 350'
    if 'ELECTRA' in model:
        return 'Electra 350'
    if 'BULLET' in model:
        return 'Bullet 350'
    if 'CLASSIC' in model:
        return 'Classic 350'
    if 'RUMBLER' in model:
        return 'Rumbler'
    if 'SIGNALS 350' in model:
        return 'SIGNALS 350'

    return 'Other'

df_cleaned['first_purchased_model_mapped'] = df_cleaned['first_purchased_model'].apply(map_model)
df_cleaned['second_purchased_model_mapped'] = df_cleaned['second_purchased_model'].apply(map_model)
df_cleaned['first_enquired_model_mapped'] = df_cleaned['first_enquired_model'].apply(map_model)
df_cleaned['second_enquired_model_mapped'] = df_cleaned['second_enquired_model'].apply(map_model)

print("Not converted model names in first_purchased_model:")
print(df_cleaned.loc[
    df_cleaned['first_purchased_model_mapped'] == 'Other',
    'first_purchased_model'
].value_counts())

print("\n")
print("Not converted model names in second_purchased_model:")
print(df_cleaned.loc[
    df_cleaned['second_purchased_model_mapped'] == 'Other',
    'second_purchased_model'
].value_counts())

print("\n")
print("Not converted model names in first_enquired_model:")
print(df_cleaned.loc[
    df_cleaned['first_enquired_model_mapped'] == 'Other',
    'first_enquired_model'
].value_counts())

print("\n")
print("Not converted model names in second_enquired_model:")
print(df_cleaned.loc[
    df_cleaned['second_enquired_model_mapped'] == 'Other',
    'second_enquired_model'
].value_counts())

df_cleaned['first_purchased_model_mapped'].value_counts()

# 5. Model Category

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
}

df_cleaned['first_bike_category'] = df_cleaned['first_purchased_model_mapped'].map(category_map)
df_cleaned['second_bike_category'] = df_cleaned['second_purchased_model_mapped'].map(category_map)
df_cleaned['first_enquired_bike_category'] = df_cleaned['first_enquired_model_mapped'].map(category_map)
df_cleaned['second_enquired_bike_category'] = df_cleaned['second_enquired_model_mapped'].map(category_map)

df_cleaned['first_bike_category'] = df_cleaned['first_bike_category'].fillna('Other')
df_cleaned['second_bike_category'] = df_cleaned['second_bike_category'].fillna('Other')
df_cleaned['first_enquired_bike_category'] = df_cleaned['first_enquired_bike_category'].fillna('Other')
df_cleaned['second_enquired_bike_category'] = df_cleaned['second_enquired_bike_category'].fillna('Other')

df_cleaned.columns

# 6. Model engine

def classify_engine(row):
    model = row['first_purchased_model_mapped']   # <-- replace with your actual column name
    date = pd.to_datetime(row['first_purchase_date'], errors='coerce')

    # Safety check
    if pd.isna(date):
        return 'Unknown'

    # Cutoff dates
    classic_cutoff = pd.Timestamp('2021-09-30')
    bullet_cutoff = pd.Timestamp('2023-09-30')

   # -------- CLASSIC --------
    if model in ['Classic 350', 'Classic 500']:
        if date < classic_cutoff:
            return 'UCE'
        else:
            return 'J-series'

    # -------- BULLET --------
    if model in ['Bullet 350', 'Bullet 500','Electra 350']:
        if date < bullet_cutoff:
            return 'UCE'
        else:
            return 'J-series'

    if model in ['SIGNALS 350','Lightning 535','CKDKIT P1E', 'U4J', 'CL EFI 500','Thunderbird 350', 'Thunderbird 500','Rumbler']:
        return 'UCE'

    # -------- DEFAULT (OTHER MODELS) --------
    return 'J-series'

df_cleaned['First_purchased_model(UCE_or_J-series)'] = df_cleaned.apply(classify_engine, axis=1)

def classify_engine(row):
    model = row['second_purchased_model_mapped']   # <-- replace with your actual column name
    date = pd.to_datetime(row['second_purchase_date'], errors='coerce')

    # Safety check
    if pd.isna(date):
        return 'Unknown'

    # Cutoff dates
    classic_cutoff = pd.Timestamp('2021-09-30')
    bullet_cutoff = pd.Timestamp('2023-09-30')

   # -------- CLASSIC --------
    if model in ['Classic 350', 'Classic 500']:
        if date < classic_cutoff:
            return 'UCE'
        else:
            return 'J-series'

    # -------- BULLET --------
    if model in ['Bullet 350', 'Bullet 500','Electra 350']:
        if date < bullet_cutoff:
            return 'UCE'
        else:
            return 'J-series'

    if model in ['SIGNALS 350','Lightning 535','CKDKIT P1E', 'U4J', 'CL EFI 500','Thunderbird 350', 'Thunderbird 500','Rumbler']:
        return 'UCE'

    # -------- DEFAULT (OTHER MODELS) --------
    return 'J-series'

df_cleaned['second_purchased_model(UCE_or_J-series)'] = df_cleaned.apply(classify_engine, axis=1)

df_cleaned['state__c'].unique()

# 7. State tier wise

# Define tier mappings (uppercase to match normalized data)
tier1 = ['MAHARASHTRA', 'KARNATAKA', 'TAMIL NADU', 'GUJARAT', 'TELANGANA', 'DELHI', 'NEW DELHI']
tier2 = ['ANDHRA PRADESH', 'UTTAR PRADESH', 'WEST BENGAL', 'HARYANA', 'PUNJAB',
         'RAJASTHAN', 'KERALA', 'ODISHA', 'MADHYA PRADESH', 'CHHATTISGARH',
         'GOA', 'UTTARAKHAND', 'CHANDIGARH', 'PUDUCHERRY', 'DAMAN & DIU',
         'DADRA & NAGAR HAVELI', 'DADRA & NAGAR HAVELI AND DAMAN AND DIU']
tier3 = ['BIHAR', 'JHARKHAND', 'JAMMU & KASHMIR', 'HIMACHAL PRADESH', 'ASSAM', 'SIKKIM',
         'MEGHALAYA', 'MANIPUR', 'NAGALAND', 'MIZORAM', 'ARUNACHAL PRADESH', 'TRIPURA',
         'LADAKH', 'ANDAMAN & NICOBAR ISLANDS', 'LAKSHADWEEP']

# Map states to tier
def map_tier(state):
    if state in tier1:
        return "Tier 1"
    elif state in tier2:
        return "Tier 2"
    elif state in tier3:
        return "Tier 3"
    else:
        return "Unknown"

df_cleaned["state_clean"] = df_cleaned["state__c"].str.upper().str.strip()
df_cleaned["state_clean"] = df_cleaned["state_clean"].replace({
    "NEW DELHI": "DELHI",
})

# Replace values in-place in state__c
df_cleaned["State"] = df_cleaned["state_clean"].apply(map_tier)

print(
    df_cleaned.loc[
        df_cleaned["State"] == "Unknown",
        "state_clean"
    ].value_counts(dropna=False)
)

# 8. vehicle age feature

df_cleaned['vehicle_age_days_1stbike'] = (
    df_cleaned['second_purchase_date'] - df_cleaned['first_purchase_date']
).dt.days

df_cleaned['vehicle_age_1stbike'] = (
    df_cleaned['vehicle_age_days_1stbike'] / 365.25
).round(2)

df_cleaned.columns

df_cleaned.drop(columns=[
                 'first_purchased_model','second_purchased_model',
                 'first_enquired_model','second_enquired_model'
                 ],inplace=True)

# 9. Data validation and correction for service count, amount, km driven and purchased date
# '''Higher KM driven → More services,
# More services → higher service amount,
# Older bike → generally higher KM & service count'''
# '''Avg KM per year: 8,000 – 15,000,
# Service frequency: 2–4 per year,
# Cost per service: ₹1,500 – ₹4,000'''

df_cleaned.notna().sum()/df_cleaned.shape[0]

# df_cleaned["km_per_year"] = (
#     df_cleaned["KM_before_2nd_bike"] /
#     df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
# )

# df_cleaned["services_per_year"] = (
#     df_cleaned["service_count"] /
#     df_cleaned["vehicle_age_1stbike"].replace(0, np.nan)
# )

# df_cleaned["cost_per_service"] = (
#     df_cleaned["total_service_amount"] /
#     df_cleaned["service_count"].replace(0, np.nan)
# )

# print(df_cleaned["km_per_year"].describe())
# print(df_cleaned["services_per_year"].describe())
# print(df_cleaned["cost_per_service"].describe())

df_cleaned.columns

bins = [0, 1, 2, 3, 4, 5, 100]
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']

df_cleaned['age_bucket'] = pd.cut(
    df_cleaned['vehicle_age_1stbike'],
    bins=bins,
    labels=labels,
    include_lowest=True
)
df_cleaned['service_count_bucket'] = pd.cut(
    df_cleaned['service_count'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

bins = [0, 10000, 20000, 30000, 40000, 50000, 100000]
labels = ['0-10000', '10001-20000', '20001-30000', '30001-40000', '40001-50000', '50001+']

df_cleaned['total_service_amount_bucket'] = pd.cut(
    df_cleaned['total_service_amount'],
    bins=bins,
    labels=labels,
    include_lowest=True
)
df_cleaned['KM_before_2nd_bike_bucket'] = pd.cut(
    df_cleaned['KM_before_2nd_bike'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(df_cleaned['KM_before_2nd_bike_bucket'].value_counts(dropna=False))
print(df_cleaned['service_count_bucket'].value_counts(dropna=False))
print(df_cleaned['total_service_amount_bucket'].value_counts(dropna=False))
print(df_cleaned['age_bucket'].value_counts(dropna=False))

# 9a. Old bike but zero KM

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["KM_before_2nd_bike"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['KM_before_2nd_bike','vehicle_age_1stbike']].head())

print(df_cleaned["vehicle_age_1stbike"].isna().sum())
print(df_cleaned["KM_before_2nd_bike"].isna().sum())

mask = (
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["KM_before_2nd_bike"].fillna(0)== 0)
)

print(mask.dtype)
print(len(mask))
print(mask.value_counts(dropna=False))

df["Km_was_corrected"] = mask.astype(int)

median_km_year=C['median_km_per_year']
print("median_km_year:", median_km_year)

df_cleaned.loc[mask, "KM_before_2nd_bike"] = (
    df_cleaned.loc[mask, "vehicle_age_1stbike"] *
    median_km_year
).round()

print("Corrected rows:", mask.sum(), f"({mask.mean()*100:.2f}% of dataset)")

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["KM_before_2nd_bike"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['KM_before_2nd_bike','vehicle_age_1stbike']].head())

# 9b. Old bike but zero services

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['service_count','vehicle_age_1stbike']].head())

mask = (
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["service_count"] == 0)
)

print(mask.sum())

median_services_per_year=C['median_services_per_year']
print("median_services_per_year:", median_services_per_year)

df_cleaned["service_count_was_corrected"] = mask.astype(int)

df_cleaned.loc[mask, "service_count"] = (
    df_cleaned.loc[mask, "vehicle_age_1stbike"] *
    median_services_per_year
).round().astype(int)

print("Corrected rows:", mask.sum(), f"({mask.mean()*100:.2f}% of dataset)")

records_to_check = df_cleaned[
    (df_cleaned["vehicle_age_1stbike"] > 5) &
    (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['service_count','vehicle_age_1stbike']].head())

# 9c. High KM but zero service count

print(df_cleaned['service_count'].notna().sum()/df_cleaned['service_count'].shape[0])
records_to_check = df_cleaned[
    (df_cleaned["KM_before_2nd_bike"] > 100000) &
    (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['service_count','KM_before_2nd_bike']].head())

# 9d. High service count but zero amount

print(df_cleaned['total_service_amount'].notna().sum()/df_cleaned['total_service_amount'].shape[0])
records_to_check= df_cleaned[
    (df_cleaned["service_count"] >= 5) &
    (df_cleaned["total_service_amount"] == 0)]
print(records_to_check.shape)
print(records_to_check[['service_count','total_service_amount']].head())

median_cost_per_service=C['median_cost_per_service']
print("median_cost_per_service:", median_cost_per_service)

mask = (
    (df_cleaned["service_count"] >= 5) &
    (df_cleaned["total_service_amount"] == 0)
)

print(mask.sum())

df_cleaned["service_amount_was_corrected"] = mask.astype(int)

df_cleaned.loc[mask, "total_service_amount"] = (
    df_cleaned.loc[mask, "service_count"] * median_cost_per_service
)

print("Corrected rows:", mask.sum())

print(df_cleaned['total_service_amount'].notna().sum()/df_cleaned['total_service_amount'].shape[0])
records_to_check= df_cleaned[
    (df_cleaned["service_count"] >= 5) &
    (df_cleaned["total_service_amount"] == 0)]
print(records_to_check.shape)
print(records_to_check.head())

print(df_cleaned['service_count'].notna().sum()/df_cleaned['service_count'].shape[0])
records_to_check = df_cleaned[
    (df_cleaned["KM_before_2nd_bike"] > 100000) &
    (df_cleaned["service_count"] == 0)
]
print(records_to_check.shape)
print(records_to_check[['service_count','KM_before_2nd_bike']].head())

# 9e. Zero services but non-zero amount

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
# median_cost_per_service=C['median_cost_per_service']

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

# 10. Missing value

df_cleaned.notna().sum()/df_cleaned.shape[0]

cols_zero_fill = ['Frequency_gma',
                  'Frequency_apparel',
                  'product_count',
                  'isfinanced__c',
                  'enquiry_count_between_purchases'
]

df_cleaned[cols_zero_fill] = df_cleaned[cols_zero_fill].fillna(0)

df_cleaned['Segment'] = df_cleaned['Segment'].fillna('Unknown')
df_cleaned['State'] = df_cleaned['State'].fillna('Unknown')

# df_cleaned['NPS_missing_flag'] = df_cleaned['NPS_Rating'].isna().astype(int)
# df_cleaned['has_NPS_rating'] = df_cleaned['NPS_Rating'].notna().astype(int)
df_cleaned['NPS_Rating'] = df_cleaned['NPS_Rating'].fillna(-1)

mask = df_cleaned['total_service_amount'].isna()

df_cleaned['service_amount_was_corrected'] = (
    (df_cleaned['service_amount_was_corrected'] == 1) | mask
).astype(int)

df_cleaned.loc[mask, 'total_service_amount'] = df_cleaned.loc[mask, 'service_count'] * median_cost_per_service

# Remaining KM gaps — use the constant derived from the one-purchaser population,
# not a locally-computed median (same reasoning as the capping fix)
mask = df_cleaned['KM_before_2nd_bike'].isna()

# NEW: flag before overwrite, consistent with the one-purchaser notebook
df_cleaned['km_was_corrected'] = (
    (df_cleaned.get('km_was_corrected', 0) == 1) | mask
).astype(int)

df_cleaned.loc[mask, 'KM_before_2nd_bike'] = (
    df_cleaned.loc[mask, 'service_count'] * C["median_km_per_service"]
).round()

df_cleaned['Final_ride_done'].value_counts()

# 11. Cap extreme outliers

check_records=df_cleaned[(df_cleaned['service_count']==0) & (df_cleaned['total_service_amount']==0)]
bins = [0, 1, 2, 3, 4, 5, 100]
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']

check_records['age_bucket'] = pd.cut(
    check_records['vehicle_age_1stbike'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(check_records['age_bucket'].value_counts().sort_index())

cols = [
    "KM_before_2nd_bike",
    "service_count",
    "total_service_amount",
    'Frequency_gma'
]

df_cleaned[cols].describe(percentiles=[0.90, 0.95, 0.99, 0.995, 0.999])

col = ["enquiry_count_between_purchases", "total_cdp_amount", "total_gma_amount", "Frequency_gma","Frequency_apparel","Monetary_apparel"]
df_cleaned[col].describe(percentiles=[0.90, 0.95, 0.99, 0.995, 0.999])

len(df_cleaned[df_cleaned['Frequency_gma']>21])

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
# Apply saved caps to df_cleaned
# ============================================================

# Apply saved caps for 99.5% columns
for col in cols_995:

    # Get previously calculated cap from cleaning_constants
    cap = C["outlier_caps"][f"df_cleaned_{col}"]

    print(f"{col}: Capping at saved 99.5% cap = {cap}")

    df_cleaned[col] = df_cleaned[col].clip(upper=cap)

# Apply saved caps for 99.9% columns
for col in cols_999:

    # Get previously calculated cap from cleaning_constants
    cap = C["outlier_caps"][f"df_cleaned_{col}"]

    print(f"{col}: Capping at saved 99.9% cap = {cap}")

    df_cleaned[col] = df_cleaned[col].clip(upper=cap)

df_cleaned["Frequency_gma"].describe(
    percentiles=[0.90, 0.95, 0.99, 0.995, 0.999]
)

negative_count = (df_cleaned["total_cdp_amount"] < 0).sum()

print("Negative records before correction:", negative_count)

df_cleaned["total_cdp_amount"] = df_cleaned[
    "total_cdp_amount"
].clip(lower=0)

df_cleaned.notna().sum()/df.shape[0]

df_cleaned.columns

# 12. Final_derived_features

# 12.a Rebuild total amount spent

df_cleaned.drop(columns=['total_amount_spent'],inplace=True)

def calculate_price(model_name):
    # 1. Handle missing values (NaN) - SQL ELSE defaults to 150000
    if pd.isna(model_name):
        return 150000

    # 2. Normalize to uppercase for case-insensitive matching (SQL UPPER)
    model = str(model_name).upper()

    # 3. Sequential Logic (Matches SQL Order)
    if 'SUPER METEOR 650' in model:
        return 348900

    # Note: In your SQL, this rule appears early. It catches ALL 'THUNDERBIRD 500' strings.
    # This implies the later rule for "THUNDERBIRD 500 (NOT X)" might be shadowed by this one.
    elif 'THUNDERBIRD 500' in model:
        return 214421

    elif 'NEW CLASSIC 350' in model:
        return 199499

    elif 'HIMALAYAN' in model and 'NEW' not in model:
        return 285000

    elif 'NEW HIMALAYAN' in model:
        return 350000

    elif 'BULLET 350' in model and 'NEW' not in model:
        return 173562

    elif 'NEW BULLET' in model:
        return 173562

    elif 'BULLET 500' in model:
        return 188803

    elif 'BULLET X' in model:
        return 188803

    elif 'BULLET ELECTRA' in model:
        return 169000

    elif 'THUNDERBIRD X 350' in model:
        return 163842

    elif 'THUNDERBIRD X 500' in model:
        return 206747

    elif 'THUNDERBIRD 350' in model and 'X' not in model:
        return 156496

    # This rule is logically unreachable if 'THUNDERBIRD 500' (Rule #2) is active,
    # but included here to match your SQL snippet exactly.
    elif 'THUNDERBIRD 500' in model and 'X' not in model:
        return 206747

    elif 'METEOR 350' in model:
        return 205528

    elif 'GUERRILLA' in model:
        return 239000

    elif 'CONTINENTAL GT' in model or 'CONTINENTALGT' in model:
        return 318418

    elif 'INTERCEPTOR' in model:
        return 302418

    elif 'CLASSIC 350' in model and 'NEW' not in model:
        return 178000

    elif 'HUNTER 350' in model:
        return 149900

    elif 'SHOTGUN' in model:
        return 359430

    elif 'CLASSIC 500' in model:
        return 162000

    elif 'BEAR 650' in model:
        return 343000

    elif 'CLASSIC 650' in model:
        return 337000

    # Python shortcut: if 'SCRAM 411' is in model, 'SCRAM' is also in model,
    # so we can just check for 'SCRAM' to cover both cases.
    elif 'SCRAM' in model:
        return 215000

    elif 'BOBBER' in model:
        return 180000

    elif 'SIGNALS' in model:
        return 164095

    # SQL ELSE
    else:
        return 150000

# Assuming your data is in a DataFrame called df
# Apply the function to create the new column
df_cleaned['first_bike_price'] = df_cleaned['first_purchased_model_mapped'].apply(calculate_price)

# Example Verification
test_models = ['HUNTER 350 DAPPER GREY', 'THUNDERBIRD 500 STONE', 'nan', 'Classic 350 black']
print(f"{'Model':<30} | {'Price'}")
print("-" * 45)
for m in test_models:
    # Simulating the nan float value from your list
    val = np.nan if m == 'nan' else m
    print(f"{str(val):<30} | {calculate_price(val)}")

cols = [
    'first_bike_price',
    'total_service_amount',
    'total_cdp_amount',
    'total_gma_amount',
    'total_mag_amount',
    'Monetary_apparel'
]

df_cleaned[cols] = df_cleaned[cols].fillna(0)

df_cleaned['total_amount_spent'] = df_cleaned[cols].sum(axis=1)

df_cleaned['total_cdp_amount'].describe()

df_cleaned.drop(columns=[
     'total_cdp_amount',
     'total_gma_amount',
     'total_mag_amount',
     'Monetary_apparel',
     'state__c',
     'state_clean'],inplace=True)

# 12.b. Last activity date

#### Create the last activity date of the customer based on the last purchase date across all categories

date_cols = [
    'first_enquiry_date', 'second_enquiry_date',
    'last_purchase_date_service','last_purchase_date_gma',
    'last_purchase_date_softProduct','last_purchase_date_apparel',
    'last_purchase_date_apparel_store',
]
date_cols = [c for c in date_cols if c in df_cleaned.columns]
df_cleaned["last_activity_date"] = df_cleaned[date_cols].max(axis=1)

df_cleaned.drop(columns=['last_purchase_date_gma',
    'last_purchase_date_softProduct','last_purchase_date_apparel',
    'last_purchase_date_apparel_store',
    'km_per_year', 'services_per_year',
    'cost_per_service', 'age_bucket', 'service_count_bucket',
    'total_service_amount_bucket', 'KM_before_2nd_bike_bucket'],inplace=True)

df_cleaned.columns

# 12.c Amount per service

# Amount per service
df_cleaned['amount_per_service'] = np.where(
    df_cleaned['service_count'] > 0,
    df_cleaned['total_service_amount'].fillna(0) / df_cleaned['service_count'].fillna(0),
    0
)

# 12.d Ratio( Actual service visits/ expected service visits)

# Ratio= Actual service visits/ Expected service visits
df_cleaned['Expected_Service_Visits_1stbike'] = (df_cleaned['vehicle_age_1stbike'] * 2 + 2).astype('int64')

df_cleaned['ratio_actual_expected_service'] =np.where(
    df_cleaned['Expected_Service_Visits_1stbike'] == 0,
    0,
    df_cleaned['service_count'].astype(float) / df_cleaned['Expected_Service_Visits_1stbike'].astype(float))

df_cleaned.notna().sum()/df_cleaned.shape[0]

# 12.e Other feature creation

df_cleaned['reference_date'] = df_cleaned['second_purchase_date']
print("Any missing reference_date in two?", df_cleaned['reference_date'].isna().sum())
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

df_cleaned = engineer_features(df_cleaned, km_col='KM_before_2nd_bike')       # CHANGE HERE if your column name differs
print("two_feat:", df_cleaned.shape)

new_cols = [
    'days_since_last_enquiry', 'days_since_last_service',
    'purchase_to_enquiry_for_2nd_bike', 'has_second_enquiry',
    'product_engagement_score', 'km_per_year',
    'first_purchase_month', 'first_purchase_quarter', 'purchased_in_festival_season',
    'is_service_overdue'
    # 'spend_per_year',
]
df_cleaned[new_cols].describe(include='all').T

df_cleaned.to_csv("Data_files/two_purchaser_cleaned_feature_engineered.csv", index=False)

df_cleaned.columns
