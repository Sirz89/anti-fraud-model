# 🛡 Anti-Fraud Core — Data Fusion Contest

Anti-fraud scoring solution: classifying bank transactions that were not confirmed by clients.

**Result:** PR-AUC = **0.0931**

---

## 🎯 Task

The bank needs to automatically classify transactions that clients **did not confirm** (🔴 — target class, fraud). The input consists of the transaction history of 100,000 clients over 1.5 years, divided into 4 time periods.

### Task Details

| Feature | Value |
|-------------|----------|
| **Data Volume** | > 200 million transactions |
| **Target Class (🔴)** | ~51,000 transactions (**~0.025%**) — extreme imbalance |
| **Yellow Light (🟡)** | ~36,000 transactions — suspicious but **confirmed** by clients. **Not the target class.** |
| **Green Light (🟢)** | All other transactions — confirmed |
| **Metric** | **PR-AUC** (average precision) |
| **Validation** | Time-based (no data leakage) |
| **Submission Format** | One transaction per client (last day) |

### Data Periods

| Period | Dates | Labels | Purpose |
|--------|------|----------|------------|
| **Pre-train** | 2023-10-01 — 2024-09-30 | ❌ None | Pre-training, historical feature extraction |
| **Train** | 2024-10-01 — 2025-05-31 | 🔴 / 🟡 / 🟢 | Model training |
| **Pre-test** | 2025-06-01 — 2025-08-09 | ❌ None | Feature extraction for the test set |
| **Test** | 2025-06-01 — 2025-08-09 | ❓ | Classification (one day per client) |

---

## 🛠 Technology Stack

| Component | Technology | Why |
|-----------|-----------|--------|
| Language | **Python 3.11** | Unified language for the entire pipeline |
| Data processing | **Polars** | 5–10x faster than Pandas on 200M rows; lazy API; efficient rolling windows |
| ML model | **CatBoost** | Best performance on tabular data with categorical features (MCC codes) |
| Parquet processing | **PyArrow** | Fast read/write |
| Metric | **scikit-learn** (`average_precision_score`) | PR-AUC |
| DataFrame | **Pandas** | Final submission assembly |
| Numerical operations | **NumPy** | Array operations |

> 💡 **Why Polars instead of Pandas?**
> Volume > 200M rows. Pandas loads everything into memory and lacks lazy evaluation. Polars uses an Arrow backend, supports streaming, and is significantly faster for rolling operations (`rolling`, `group_by`, `over`).

---

## 🏗 Approach

### 1. Class Handling

The original labeling contains three categories:
- 🔴 **target = 1** — unconfirmed transactions (fraud).
- 🟢 **target = 0** — confirmed transactions.
- 🟡 **target = -1** — suspicious, yet confirmed by clients. **Non-target class.**

**Sampling strategy:**

```python
fraud_data = part.filter(pl.col("target").is_in([0, 1]))       # all 🔴 and 🟢
clean_data = part.filter(pl.col("target") == -1).sample(
fraction=0.1, seed=42                                       # 10% of the "gray mass"
)
