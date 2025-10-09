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
    import os
    import joblib
    return date, joblib, mo, np, os, pd, pl


@app.cell
def _():
    from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import Ridge
    from sklearn.utils.validation import check_is_fitted

    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    return (
        ColumnTransformer,
        GroupShuffleSplit,
        LGBMRegressor,
        OneHotEncoder,
        Pipeline,
        RandomizedSearchCV,
        Ridge,
        SimpleImputer,
        TransformedTargetRegressor,
        XGBRegressor,
        mean_absolute_error,
        r2_score,
        root_mean_squared_error,
    )


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
    return categorical_features, numeric_features, test_df, train_df, valid_df


@app.cell
def _(
    ColumnTransformer,
    HAS_LGBM,
    LGBMRegressor,
    OneHotEncoder,
    Pipeline,
    Ridge,
    SimpleImputer,
    TransformedTargetRegressor,
    XGBRegressor,
    categorical_features,
    np,
    numeric_features,
    test_df,
    train_df,
    valid_df,
):
    # Build preprocessing + model pipeline
    # Column transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
        # Trees don't need scaling. Add StandardScaler if using linear
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    # Choose model
    if 'HAS_LGBM' in globals() and HAS_LGBM:
        model = LGBMRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=-1, 
            num_leaves=31, 
            subsample=0.9, 
            colsample_bytree=0.9, 
            random_state=42
        )
    else:
        try:
            model = XGBRegressor(
                n_estimators=800, 
                learning_rate=0.05, 
                max_depth=8, 
                subsample=0.9, 
                colsample_bytree=0.9, 
                random_state=42, 
                tree_method="hist"
            )
        except Exception:
            model = Ridge(alpha=1.0, random_state=42)

    # Pipeline + target transform
    pipeline = Pipeline(steps=[
        ("pre", preprocessor), 
        ("reg", model)
    ])

    # Log-transform the target
    regr = TransformedTargetRegressor(
        regressor=pipeline, 
        func=np.log1p, 
        inverse_func=np.expm1,
        check_inverse=False
    )

    # Split X/y
    _TARGET = "rentEstimate_currentPrice"
    X_train = train_df.drop(columns=[_TARGET])
    y_train = train_df[_TARGET].values
    X_valid = valid_df.drop(columns=[_TARGET])
    y_valid = valid_df[_TARGET].values
    X_test  = test_df.drop(columns=[_TARGET])
    y_test  = test_df[_TARGET].values

    regr, X_train, y_train, X_valid, y_valid, X_test, y_test
    return X_train, X_valid, regr, y_train, y_valid


@app.cell
def _(
    RandomizedSearchCV,
    X_train,
    X_valid,
    mean_absolute_error,
    r2_score,
    regr,
    root_mean_squared_error,
    y_train,
    y_valid,
):
    # Hyperparameter tuning (quick randomized search)
    # Build param grid depending on model type
    base = regr.regressor.named_steps["reg"]
    if hasattr(base, "get_params") and "num_leaves" in base.get_params():
        # LightGBM
        param_distributions = {
            "regressor__reg__n_estimators": [600, 800, 1000, 1200],
            "regressor__reg__learning_rate": [0.03, 0.05, 0.08],
            "regressor__reg__num_leaves": [15, 31, 63], 
            "regressor__reg__max_depth": [-1, 8, 10, 12], 
            "regressor__reg__min_child_samples": [10, 20, 50],
            "regressor__reg__subsample": [0.8,0.9, 1.0], 
            "regressor__reg__colsample_bytree": [0.8, 0.9, 1.0]
        }
    elif hasattr(base, "get_params") and "max_depth" in base.get_params():
        # XGBoost
        param_distributions = {
            "regressor__reg__n_estimators": [500, 800, 1200], 
            "regressor__reg__learning_rate": [0.03, 0.05, 0.08], 
            "regressor__reg__max_depth": [6, 8, 10], 
            "regressor__reg__subsample": [0.8, 0.9, 1.0],
            "regressor__reg__colsample_bytree": [0.8, 0.9, 1.0], 
        }
    else:
        # Ridge fallback
        param_distributions = {
            "regressor__reg__alpha": [0.1, 1.0, 10.0, 50.0]
        }

    # Randomized search over a few configs to keep it fast
    rs = RandomizedSearchCV(
        estimator=regr, 
        param_distributions=param_distributions, 
        n_iter=12, 
        scoring="neg_mean_absolute_error", 
        random_state=42, 
        n_jobs=-1, 
        cv=3, 
        verbose=1
    )
    rs.fit(X_train, y_train)

    # Evaluate best on validation
    best = rs.best_estimator_
    val_pred = best.predict(X_valid)
    mae = mean_absolute_error(y_valid, val_pred)
    rmse = root_mean_squared_error(y_valid, val_pred)
    r2 = r2_score(y_valid, val_pred)

    {"best_params": rs.best_params_, "val_mae": mae, "val_rmse": rmse, "val_r2": r2}, best 
    return (best,)


@app.cell
def _():
    import sys, inspect
    import sklearn, sklearn.metrics as sm
    print("Python:", sys.executable)
    print("sklearn version:", sklearn.__version__)
    print("mse module:", sm.mean_squared_error.__module__)
    print("mse signature:", inspect.signature(sm.mean_squared_error))
    return


@app.cell
def _(
    best,
    date,
    joblib,
    mean_absolute_error,
    np,
    os,
    pd,
    r2_score,
    root_mean_squared_error,
    test_df,
    train_df,
    valid_df,
):
    # Train final model on train+valid and evaluate on test
    _TARGET = "rentEstimate_currentPrice"

    # Combine train + valid
    train_valid = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_trv = train_valid.drop(columns=[_TARGET])
    y_trv = train_valid[_TARGET].values

    best.fit(X_trv, y_trv)

    # Test evaluation
    X_test_final = test_df.drop(columns=[_TARGET])
    y_test_final = test_df[_TARGET].values
    y_pred_final = best.predict(X_test_final)

    mae_final = mean_absolute_error(y_test_final, y_pred_final)
    rmse_final = root_mean_squared_error(y_test_final, y_pred_final)
    r2_final = r2_score(y_test_final, y_pred_final)
    mape_final = np.mean(np.abs((y_test_final - y_pred_final) / np.maximum(1e-9, y_test_final))) * 100.0

    # Residual std for a simple interval
    resid_std = float(np.std(y_test_final - y_pred_final))

    # Save artifacts
    ART_DIR = "artifacts"
    os.makedirs(ART_DIR, exist_ok=True)
    model_path = os.path.join(ART_DIR, "rent_model.joblib")
    meta_path = os.path.join(ART_DIR, "rent_model_meta.joblib")

    joblib.dump(best, model_path)
    joblib.dump({"resid_std": resid_std, "trained_on": str(date.today())}, meta_path)

    {
        "test_mae": mae_final,
        "test_rmse": rmse_final,
        "test_r2": r2_final, 
        "test_mape_percent": mape_final, 
        "model_path": model_path, 
        "meta_path": meta_path, 
        "resid_std": resid_std
    }
    return


@app.cell
def _(mo):
    # Create UI widgets
    postcode = mo.ui.text(label="Postcode", placeholder="e.g., SW1P 2TA")
    bathrooms = mo.ui.number(label="Bathrooms", value=1, start=0, step=1)
    bedrooms = mo.ui.number(label="Bedrooms", value=2, start=0, step=1)
    floor_area = mo.ui.number(label="Floor area (sq m)", value=60.0, start=0.0, step=1.0)
    living_rooms = mo.ui.number(label="Living rooms", value=1, start=0, step=1)

    tenure = mo.ui.dropdown(
        label="Tenure", 
        options=["Leasehold", "Freehold", "Unknown"], 
        value="Leasehold"
    )
    property_type = mo.ui.dropdown(
        label="Property type", 
        options=["Purpose Built Flat", "Flat/Maisonette", "Mid Terrace House", "End Terrace House", "Terrace Property", "Unknown"], 
        value="Flat/Maisonette"
    )
    energy = mo.ui.dropdown(
        label="Energy rating", 
        options=["A","B","C","D","E","F","G","Unknown"], 
        value="D"
    )

    history_price = mo.ui.number(label="Last history price (£)", value=None, start=0, step=1000)
    history_date = mo.ui.number(label="Last history date", value=None)

    # Manual lat/lon if postcode not found
    manual_lat = mo.ui.number(label="Latitude (optional override)", value=None, step=0.0001)
    manual_lng = mo.ui.number(label="Longitude (optional override", value=None, step=0.0001)

    # Confidence
    conf_level = mo.ui.dropdown(
        label="Sale estimate confidence (optional)", 
        options=["HIGH","MEDIUM","LOW","Unknown"], 
        value="HIGH"
    )

    # Predict button
    predict_btn = mo.ui.button(label="Predict rent")

    widgets = {
            "postcode": postcode,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "floor_area": floor_area,
            "living_rooms": living_rooms,
            "tenure": tenure,
            "property_type": property_type,
            "energy": energy,
            "history_price": history_price,
            "history_date": history_date,
            "manual_lat": manual_lat,
            "manual_lng": manual_lng,
            "conf_level": conf_level,
            "predict_btn": predict_btn
        }

    widgets
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
