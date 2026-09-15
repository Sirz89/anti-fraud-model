# 🛡 Anti-Fraud Core — Data Fusion Contest

Anti-fraud scoring solution for classifying bank transactions that were not confirmed by clients.

**Result:** PR-AUC = **0.0931**  
**Competition Rank:** **224th place**

---

## 🎯 Task

The bank needs to automatically classify transactions that clients **did not confirm** (🔴 — target class, fraud).

The input consists of the transaction history of approximately 100,000 clients over 1.5 years, divided into several time periods.

### Task Details

| Feature | Value |
|-------------|----------|
| **Data Volume** | > 200 million transactions |
| **Target Class (🔴)** | ~51,000 transactions (**~0.025%**) — extreme imbalance |
| **Yellow Light (🟡)** | ~36,000 transactions — suspicious but **confirmed** by clients. **Not the target class.** |
| **Green Light (🟢)** | All other transactions — confirmed |
| **Metric** | **PR-AUC** (Average Precision) |
| **Validation** | Time-based / time-aware validation |
| **Submission Format** | One transaction per client (last day) |

### Data Periods

| Period | Dates | Labels | Purpose |
|--------|------|------|----------|
| **Pre-train** | 2023-10-01 — 2024-09-30 | ❌ None | Historical data for feature extraction |
| **Train** | 2024-10-01 — 2025-05-31 | 🔴 / 🟡 / 🟢 | Model training |
| **Pre-test** | 2025-06-01 — 2025-08-09 | ❌ None | Historical data for test feature extraction |
| **Test** | 2025-06-01 — 2025-08-09 | ❓ | Final classification |

---

## 🛠 Technology Stack

| Component | Technology | Why |
|-----------|-----------|--------|
| Language | **Python 3.11** | Main development language |
| Data processing | **Polars** | Efficient processing of very large datasets |
| ML model | **CatBoost** | Strong performance on tabular data and categorical features |
| Parquet processing | **PyArrow** | Efficient columnar data processing |
| Metric | **scikit-learn** | PR-AUC / Average Precision |
| DataFrame | **Pandas** | Model input and submission assembly |
| Numerical operations | **NumPy** | Numerical processing |

> 💡 **Why Polars instead of Pandas?**
>
> The dataset contains more than 200M transactions, so memory-efficient data processing is critical.
>
> Polars provides efficient columnar operations, lazy evaluation, grouping, and rolling-window operations that are useful for large-scale feature engineering.

---

## 🏗 Approach

The solution focuses on **customer behavior and temporal transaction patterns** rather than relying only on individual transaction attributes.

The main pipeline:

```text
Raw Parquet Data
       ↓
Large-scale preprocessing
       ↓
Historical customer statistics
       ↓
Rolling time-window features
       ↓
Temporal features
       ↓
Behavioral features
       ↓
Class handling
       ↓
CatBoost
       ↓
Fraud probability
       ↓
Submission
