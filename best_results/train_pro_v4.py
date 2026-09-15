import polars as pl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import gc
import os
import glob

# --- 1. Настройки путей ---
DATA_DIR = r"C:/proect"
train_pattern = os.path.join(DATA_DIR, "train_part_*.parquet")
pre_train_pattern = os.path.join(DATA_DIR, "pretrain_part_*.parquet") 
labels_path = os.path.join(DATA_DIR, "train_labels.parquet")

print("1. Сбор глобальной статистики...")
pre_train_stats = (
    pl.scan_parquet(pre_train_pattern)
    .group_by("customer_id")
    .agg([
        pl.col("operaton_amt").mean().alias("hist_avg_amt"),
        pl.col("operaton_amt").max().alias("hist_max_amt"),
        pl.col("operaton_amt").count().alias("hist_trans_count")
    ])
    .collect()
    .with_columns(pl.col(pl.Float64).cast(pl.Float32))
)

# --- 2. Цикл обработки ---
train_files = sorted(glob.glob(train_pattern))
processed_parts = []
labels = pl.read_parquet(labels_path).select(["event_id", "target"])

for i, file_path in enumerate(train_files):
    print(f"\n>>> Файл {i+1}/{len(train_files)}: {os.path.basename(file_path)}")
    
    part = pl.read_parquet(file_path).drop(["event_desc", "session_id"]).with_columns([
        pl.col("event_dttm").str.to_datetime(),
        pl.col("operaton_amt").cast(pl.Float32)
    ]).sort(["customer_id", "event_dttm"])
    
    part = part.with_columns(pl.col("event_dttm").set_sorted())

    print("   Генерация простых признаков...")
    # Используем closed="left", это честное окно без текущей строки
    c_10m = part.rolling(index_column="event_dttm", period="10m", group_by="customer_id", closed="left").agg(pl.len().alias("cnt_10m"))
    s_1h = part.rolling(index_column="event_dttm", period="1h", group_by="customer_id", closed="left").agg(pl.col("operaton_amt").sum().alias("sum_1h"))

    part = (
        part
        .join(c_10m, on=["customer_id", "event_dttm"], how="left")
        .join(s_1h, on=["customer_id", "event_dttm"], how="left")
        .join(labels, on="event_id", how="left")
        .join(pre_train_stats, on="customer_id", how="left")
        .with_columns([
            pl.col("target").fill_null(-1),
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            # Ключевой признак: время с прошлой транзакции
            (pl.col("event_dttm").diff().over("customer_id").dt.total_seconds().fill_null(99999)).alias("diff_sec"),
            # Отношение текущей суммы к сумме за час
            (pl.col("operaton_amt") / (pl.col("sum_1h").fill_null(0) + 1.0)).alias("amt_ratio_1h")
        ])
    )

    fraud_data = part.filter(pl.col("target").is_in([0, 1]))
    # Берем чуть больше чистых данных (10%) для стабильности
    clean_data = part.filter(pl.col("target") == -1).sample(fraction=0.1, seed=42)
    processed_parts.append(pl.concat([fraud_data, clean_data]))
    del part, c_10m, s_1h; gc.collect()

train_df = pl.concat(processed_parts).with_columns(
    pl.when(pl.col("target") == -1).then(0).otherwise(pl.col("target")).alias("target")
)

# --- 3. Обучение ---
X = train_df.drop(["event_id", "customer_id", "event_dttm", "target"]).to_pandas()
y = train_df["target"].to_pandas()

# Очистка NaN и категорий
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_cols] = X[numeric_cols].fillna(0)
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if 'mcc_code' in X.columns: cat_cols.append('mcc_code')
for col in cat_cols:
    X[col] = X[col].astype(str).replace(['None', 'nan', 'NaN'], 'UNKNOWN').fillna('UNKNOWN')

print("6. Запуск CatBoost (Balanced Mode)...")
model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.03,
    depth=6,             # Оптимальная глубина
    l2_leaf_reg=5,       # Умеренная регуляризация
    task_type="GPU",
    eval_metric='PRAUC',
    early_stopping_rounds=300,
    cat_features=cat_cols,
    verbose=100
)

model.fit(X, y)
model.save_model("antifraud_v7.cbm")