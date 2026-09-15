import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os
import glob
import gc

DATA_DIR = r"C:/proect"
test_pattern = os.path.join(DATA_DIR, "test.parquet")
pre_train_pattern = os.path.join(DATA_DIR, "pretrain_part_*.parquet")
model_path = "antifraud_v7.cbm"

model = CatBoostClassifier().load_model(model_path)

pre_train_stats = (
    pl.scan_parquet(pre_train_pattern)
    .group_by("customer_id")
    .agg([
        pl.col("operaton_amt").mean().alias("hist_avg_amt"),
        pl.col("operaton_amt").max().alias("hist_max_amt"),
        pl.col("operaton_amt").count().alias("hist_trans_count")
    ]).collect().with_columns(pl.col(pl.Float64).cast(pl.Float32))
)

test_files = sorted(glob.glob(test_pattern))
all_predictions = []

for file_path in test_files:
    print(f"Обработка {os.path.basename(file_path)}...")
    test_df = pl.read_parquet(file_path).with_columns([
        pl.col("event_dttm").str.to_datetime(),
        pl.col("operaton_amt").cast(pl.Float32)
    ]).sort(["customer_id", "event_dttm"])
    
    test_df = test_df.with_columns(pl.col("event_dttm").set_sorted())

    c_10m = test_df.rolling(index_column="event_dttm", period="10m", group_by="customer_id", closed="left").agg(pl.len().alias("cnt_10m"))
    s_1h = test_df.rolling(index_column="event_dttm", period="1h", group_by="customer_id", closed="left").agg(pl.col("operaton_amt").sum().alias("sum_1h"))

    test_df = (
        test_df
        .join(c_10m, on=["customer_id", "event_dttm"], how="left")
        .join(s_1h, on=["customer_id", "event_dttm"], how="left")
        .join(pre_train_stats, on="customer_id", how="left")
        .with_columns([
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            (pl.col("event_dttm").diff().over("customer_id").dt.total_seconds().fill_null(99999)).alias("diff_sec"),
            (pl.col("operaton_amt") / (pl.col("sum_1h").fill_null(0) + 1.0)).alias("amt_ratio_1h")
        ])
    )

    X_test = test_df.drop(["event_id", "customer_id", "event_dttm", "event_desc", "session_id"]).to_pandas()
    
    X_test = X_test[model.feature_names_]
    
    # Очистка NaN
    num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test[num_cols] = X_test[num_cols].fillna(0)
    cat_cols = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        X_test[col] = X_test[col].astype(str).replace(['None', 'nan', 'NaN'], 'UNKNOWN').fillna('UNKNOWN')

    preds = model.predict_proba(X_test)[:, 1]
    all_predictions.append(pd.DataFrame({"event_id": test_df["event_id"], "predict": preds}))

final_sub = pd.concat(all_predictions).drop_duplicates(subset=["event_id"])
final_sub.to_csv("submission_v7.csv", index=False)
print("Готово! Пробуй заливать v7.")
