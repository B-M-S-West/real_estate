import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    return date, pd, pl


@app.cell
def _():
    from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import Ridge

    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    return (GroupShuffleSplit,)


@app.cell
def _(pl):
    # Load parquet of dataset downloaded from Kaggle https://www.kaggle.com/datasets/jakewright/house-price-data
    real_estate_data = pl.read_parquet("data/kaggle_london_house_price_data.parquet")
    return (real_estate_data,)


@app.cell
def _(real_estate_data):
    real_estate_data.glimpse()
    return


@app.cell
def _(pl, real_estate_data):
    # Make sure the numeric columns are in the expected format Float64
    numeric_cols = [
        "bathrooms","bedrooms","floorAreaSqM","livingRooms",
        "latitude","longitude",
        "rentEstimate_lowerPrice","rentEstimate_currentPrice","rentEstimate_upperPrice",
        "saleEstimate_lowerPrice","saleEstimate_currentPrice","saleEstimate_upperPrice",
        "saleEstimate_valueChange.numericChange","saleEstimate_valueChange.percentageChange",
        "history_price","history_percentageChange","history_numericChange"
    ]
    for c in numeric_cols:
        if c in real_estate_data.columns:
            real_estate_data_1 = real_estate_data.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    # # Make sure the date columns are in the expected format
    date_cols = [
        "saleEstimate_valueChange.saleDate",
        "saleEstimate_ingestedAt",
        "history_date",
    ]
    for c in date_cols:
        if c in real_estate_data_1.columns:
            # Try parsing as date or datetime
            real_estate_data_1 = real_estate_data_1.with_columns(
                pl.col(c).str.strptime(pl.Date, strict=False, format="%F").fill_null(
                    pl.col(c).str.strptime(pl.Datetime, strict=False, format="%F").dt.date()
                )
            )

    # Strings to keep as strings
    str_cols = [
        "fullAddress","postcode","country","outcode",
        "tenure","propertyType","currentEnergyRating","saleEstimate_confidenceLevel"
    ]
    for c in str_cols:
        if c in real_estate_data_1.columns:
            real_estate_data_1 = real_estate_data_1.with_columns(pl.col(c).cast(pl.Utf8, strict=False))

    # Quick schema and null counts
    schema = real_estate_data_1.schema
    nulls = real_estate_data_1.select([pl.col(c).is_null().sum().alias(c) for c in real_estate_data_1.columns])

    real_estate_data_1.head(), schema, nulls
    return (real_estate_data_1,)


@app.cell
def _(pl, real_estate_data_1):
    # Basic cleaning and target definition
    TARGET = "rentEstimate_currentPrice"
    real_estate_data_2 = real_estate_data_1

    # Drop duplicates based upon address key
    subset_key = [c for c in ["fullAddress","postcode"] if c in real_estate_data_1.columns]
    if subset_key:
        real_estate_data_2 = real_estate_data_2.unique(subset=subset_key, keep="first")

    # Remove rows without target
    if TARGET in real_estate_data_2.columns:
        real_estate_data_2 = real_estate_data_2.filter(pl.col(TARGET).is_not_null())

    # Cap target outliers (e.g., 1st and 99th percentile)
    q = real_estate_data_2.select([
        pl.col(TARGET).quantile(0.01).alias("q01"),
        pl.col(TARGET).quantile(0.99).alias("q99")
    ]).row(0)
    q01, q99 = q[0], q[1]
    real_estate_data_2 = real_estate_data_2.with_columns(
        pl.when(pl.col(TARGET) < q01).then(q01)
         .when(pl.col(TARGET) > q99).then(q99)
         .otherwise(pl.col(TARGET))
         .alias(TARGET)
    )

    # Treat floorAreaSqM == 0 as null (if any)
    if "floorAreaSqM" in real_estate_data_2.columns:
        real_estate_data_2 = real_estate_data_2.with_columns(
            pl.when(pl.col("floorAreaSqM") <= 0).then(None).otherwise(pl.col("floorAreaSqM")).alias("floorAreaSqM")
        )

    real_estate_data_2.shape, real_estate_data_2.head()
    return TARGET, real_estate_data_2


@app.cell
def _(date, pl, real_estate_data_2):
    # Feature engineering
    today =date.today()

    real_estate_data_3 = real_estate_data_2

    # months_since_last_sale from saleEstimate_valueChange.saleDate
    if "saleEstimate_valueChange.saleDate" in real_estate_data_3.columns:
        real_estate_data_3 = real_estate_data_3.with_columns(
            (
                (pl.lit(today) - pl.col("saleEstimate_valueChange.saleDate")).dt.total_days() / 30.4375
            ).alias("months_since_last_sale")
        )
    else:
        real_estate_data_3 = real_estate_data_3.with_columns(pl.lit(None).alias("months_since_last_sale"))

    # Appreciation ratio: saleEstimate_currentPrice / history_price (if both exist)
    if ("saleEstimate_currentPrice" in real_estate_data_3.columns) and ("history_price" in real_estate_data_3.columns):
        real_estate_data_3 = real_estate_data_3.with_columns(
            (pl.col("saleEstimate_currentPrice") / pl.col("history_price")).alias("sale_to_hist_ratio")
        )
    else:
        real_estate_data_3 = real_estate_data_3.with_columns(pl.lit(None).alias("sale_to_hist_ratio"))

    # Clean energy rating to uppercase single-letter buckets (keep None)
    if "currentEnergyRating" in real_estate_data_3.columns:
        real_estate_data_3 = real_estate_data_3.with_columns(
            pl.col("currentEnergyRating").str.to_uppercase().alias("currentEnergyRating")
        )

    # Build postcode lookup: median lat/long per postcode
    lookup_df = real_estate_data_3.filter(
        pl.col("postcode").is_not_null() & pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
    ).group_by("postcode").agg([
        pl.median("latitude").alias("postcode_lat"),
        pl.median("longitude").alias("postcode_lng"),
        pl.first("outcode").alias("postcode_outcode"),
    ])

    postcode_lookup = {
        r["postcode"]: {
            "lat": r["postcode_lat"],
            "lng": r["postcode_lng"],
            "outcode": r["postcode_outcode"],
        }
        for r in lookup_df.iter_rows(named=True)
    }

    real_estate_data_3.head(3), len(postcode_lookup), list(real_estate_data_3.columns)
    return (real_estate_data_3,)


@app.cell
def _(GroupShuffleSplit, TARGET, pd, pl, real_estate_data_3):
    # Select features and split into train/valid/test with group-aware 
    # Select my numerical features and categorical features to use
    numeric_features = [
            "bathrooms","bedrooms","floorAreaSqM","livingRooms",
            "latitude","longitude",
            "months_since_last_sale","sale_to_hist_ratio"
    ]
    categorical_features = [
        "tenure","propertyType","currentEnergyRating","outcode","outcode_prefix","saleEstimate_confidenceLevel"
    ]

    # Filter to existing columns
    numeric_features = [c for c in numeric_features if c in real_estate_data_3.columns]
    categorical_features = [c for c in categorical_features if c in real_estate_data_3.columns]

    feature_cols = numeric_features + categorical_features + [TARGET]
    data = real_estate_data_3.select([c for c in feature_cols if c in real_estate_data_3.columns])

    # Drop rows with missing target (already mostly handled)
    data = data.filter(pl.col(TARGET).is_not_null())

    # Convert to pandas for scikit-learn (Can't split data by indices in polars easily) .take depreciated
    pdf = data.to_pandas()

    # Groups = outcode to avoid leakage
    groups = pdf["outcode"] if "outcode" in pdf.columns else pd.Series(["ALL"] * len(pdf))

    # First, train vs temp split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(gss.split(pdf, groups=groups))

    train_df = pdf.iloc[train_idx].reset_index(drop=True)
    temp_df = pdf.iloc[temp_idx].reset_index(drop=True)

    # Then, temp -> valid/test split (50/50 of temp)
    groups_temp = temp_df["outcode"] if "outcode" in temp_df.columns else pd.Series(["ALL"] * len(temp_df))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    valid_idx, test_idx = next(gss2.split(temp_df, groups=groups_temp))

    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    train_df.shape, valid_df.shape, test_df.shape, numeric_features, categorical_features
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
