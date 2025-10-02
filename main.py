import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, date
    return (pl,)


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
    return


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
    return


@app.cell
def _():
    # Feature engineering
    return


if __name__ == "__main__":
    app.run()
