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
    # Coerce likely numeric columns
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

    # Coerce date-like columns
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
