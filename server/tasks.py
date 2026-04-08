# server/tasks.py
"""
Three data pipeline debugging tasks with increasing difficulty.
Each broken pipeline RUNS but produces WRONG output (not crashes).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import io
import contextlib
import traceback


# ─────────────────────────────────────────────
# TASK 1 — EASY: Fix dtype bugs
# ─────────────────────────────────────────────

TASK_EASY_DESCRIPTION = """You are given a sales dataset with columns:
  - order_id (string)
  - quantity (string → should be int)
  - unit_price (string → should be float)
  - order_date (string → should be datetime)
  - discount_pct (string like "10%" → should be float 0.0–1.0)

The pipeline must:
1. Cast `quantity` to int
2. Cast `unit_price` to float
3. Parse `order_date` to datetime
4. Convert `discount_pct` from "X%" string to float ratio (e.g. "10%" → 0.10)
5. Compute `total_price = quantity * unit_price * (1 - discount_pct)`

The broken pipeline has 5 bugs — find and fix ALL of them.
Function signature: fix_pipeline(df: pd.DataFrame) -> pd.DataFrame"""

TASK_EASY_BROKEN_CODE = """\
import pandas as pd

def fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # BUG 1: quantity cast to float instead of int
    df['quantity'] = df['quantity'].astype(float).astype(float)
    # BUG 2: unit_price truncated to int (loses decimals)
    df['unit_price'] = df['unit_price'].astype(float).astype(int).astype(float)
    # BUG 3: order_date left as string, not parsed
    # df['order_date'] = pd.to_datetime(df['order_date'])
    # BUG 4: discount_pct not stripped of '%' — coerces to NaN, then filled with 0
    df['discount_pct'] = pd.to_numeric(df['discount_pct'], errors='coerce').fillna(0.0)
    # BUG 5: total_price formula ignores discount entirely
    df['total_price'] = df['quantity'] * df['unit_price']
    return df
"""


def get_easy_input_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "order_id":    [f"ORD-{i:04d}" for i in range(n)],
        "quantity":    [str(q) for q in np.random.randint(1, 50, n)],
        "unit_price":  [f"{p:.2f}" for p in np.random.uniform(5.0, 200.0, n)],
        "order_date":  pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist(),
        "discount_pct":[f"{d}%" for d in np.random.choice([0, 5, 10, 15, 20], n)],
    })


def get_easy_expected_df() -> pd.DataFrame:
    df = get_easy_input_df().copy()
    df["quantity"]     = df["quantity"].astype(int)
    df["unit_price"]   = df["unit_price"].astype(float)
    df["order_date"]   = pd.to_datetime(df["order_date"])
    df["discount_pct"] = df["discount_pct"].str.replace("%", "").astype(float) / 100.0
    df["total_price"]  = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"])
    return df


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM: Fix null handling + aggregation
# ─────────────────────────────────────────────

TASK_MEDIUM_DESCRIPTION = """You are given a customer transactions dataset with columns:
  - customer_id (string)
  - transaction_amount (float, some NaN)
  - category (string: "food"/"electronics"/"clothing", some NaN)
  - transaction_date (string, parseable as datetime)
  - is_refund (int 0/1, with -1 used as sentinel for missing)

The pipeline must:
1. Fill NaN in `transaction_amount` with the column MEDIAN (not mean)
2. Fill NaN in `category` with "unknown" (don't drop rows)
3. Replace -1 sentinel in `is_refund` with NaN, then fill with 0
4. Parse `transaction_date` to datetime; extract integer `month` column
5. Group by `customer_id` + `month`:
   - `total_spent`: sum of transaction_amount WHERE is_refund == 0
   - `num_transactions`: count of rows WHERE is_refund == 0
   - `refund_count`: total sum of is_refund across ALL rows
6. Return aggregated DataFrame sorted by customer_id, month (reset index)

The broken pipeline has 6 bugs — find and fix ALL of them.
Function signature: fix_pipeline(df: pd.DataFrame) -> pd.DataFrame"""

TASK_MEDIUM_BROKEN_CODE = """\
import pandas as pd
import numpy as np

def fix_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # BUG 1: fills with MEAN instead of MEDIAN
    df['transaction_amount'] = df['transaction_amount'].fillna(df['transaction_amount'].mean())
    # BUG 2: DROPS rows with NaN category instead of filling with 'unknown'
    df = df.dropna(subset=['category'])
    # BUG 3: doesn't handle -1 sentinel (treats it as valid is_refund value)
    df['is_refund'] = df['is_refund'].fillna(0)
    # BUG 4: doesn't parse date; BUG 5: hardcodes month = 1
    df['month'] = 1
    # BUG 6: doesn't filter refunds for total_spent/num_transactions
    agg = df.groupby(['customer_id', 'month']).agg(
        total_spent=('transaction_amount', 'sum'),
        num_transactions=('transaction_amount', 'count'),
        refund_count=('is_refund', 'sum')
    ).reset_index()
    return agg
"""


def get_medium_input_df() -> pd.DataFrame:
    np.random.seed(123)
    n = 60
    amounts = np.random.uniform(10.0, 500.0, n)
    amounts[np.random.choice(n, 8, replace=False)] = np.nan
    categories = np.random.choice(["food", "electronics", "clothing"], n).tolist()
    for i in np.random.choice(n, 5, replace=False):
        categories[i] = None
    is_refund = np.random.choice([0, 1], n, p=[0.85, 0.15]).tolist()
    for i in np.random.choice(n, 6, replace=False):
        is_refund[i] = -1
    return pd.DataFrame({
        "customer_id":       [f"CUST-{i % 10:02d}" for i in range(n)],
        "transaction_amount": amounts,
        "category":          categories,
        "transaction_date":  pd.date_range("2024-01-01", periods=n, freq="5D").strftime("%Y-%m-%d").tolist(),
        "is_refund":         is_refund,
    })


def get_medium_expected_df() -> pd.DataFrame:
    df = get_medium_input_df().copy()
    df["transaction_amount"] = df["transaction_amount"].fillna(df["transaction_amount"].median())
    df["category"]           = df["category"].fillna("unknown")
    df["is_refund"]          = df["is_refund"].replace(-1, np.nan).fillna(0).astype(int)
    df["transaction_date"]   = pd.to_datetime(df["transaction_date"])
    df["month"]              = df["transaction_date"].dt.month
    non_refund = df[df["is_refund"] == 0]
    total_spent   = non_refund.groupby(["customer_id","month"])["transaction_amount"].sum().rename("total_spent")
    num_tx        = non_refund.groupby(["customer_id","month"])["transaction_amount"].count().rename("num_transactions")
    refund_count  = df.groupby(["customer_id","month"])["is_refund"].sum().rename("refund_count")
    agg = pd.concat([total_spent, num_tx, refund_count], axis=1).fillna(0).reset_index()
    agg["num_transactions"] = agg["num_transactions"].astype(int)
    agg["refund_count"]     = agg["refund_count"].astype(int)
    return agg.sort_values(["customer_id","month"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# TASK 3 — HARD: Multi-table join + reshape
# ─────────────────────────────────────────────

TASK_HARD_DESCRIPTION = """You are given a dict with TWO DataFrames:
  - df['orders']:   order_id, customer_id, product_id, quantity (str), order_date (str)
  - df['products']: product_id, product_name, category, unit_price (str), cost_price (str)

The pipeline must:
1. Cast orders.quantity to int; cast products.unit_price and cost_price to float
2. Parse orders.order_date to datetime; extract integer year and month columns
3. LEFT JOIN orders onto products on product_id (keep all orders)
4. Compute revenue = quantity * unit_price
5. Compute profit  = quantity * (unit_price - cost_price)
6. Group by [category, year, month]:
   - total_revenue: sum of revenue (round to 2 decimals)
   - total_profit:  sum of profit  (round to 2 decimals)
   - order_count:   number of orders
   - avg_quantity:  mean quantity   (round to 2 decimals)
7. Sort by year, month, category. Reset index.

The broken pipeline has 7 bugs — find and fix ALL of them.
NOTE: input is a dict with keys 'orders' and 'products', not a single DataFrame.
Function signature: fix_pipeline(df: dict) -> pd.DataFrame"""

TASK_HARD_BROKEN_CODE = """\
import pandas as pd

def fix_pipeline(df: dict) -> pd.DataFrame:
    orders   = df['orders'].copy()
    products = df['products'].copy()

    # BUG 1: quantity not cast to int (stays as string)
    # BUG 2: prices not cast to float (stay as strings)
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['year'] = orders['order_date'].dt.year
    # BUG 3: month never extracted

    # BUG 4: INNER join drops unmatched orders (should be LEFT)
    merged = orders.merge(products, on='product_id', how='inner')

    # BUG 5: revenue is string concat because quantity/unit_price are strings
    merged['revenue'] = merged['quantity'].astype(str) + merged['unit_price'].astype(str)
    # BUG 6: profit never computed

    # BUG 7: groupby missing 'month' + missing profit/avg_quantity columns
    agg = merged.groupby(['category', 'year']).agg(
        total_revenue=('revenue', 'count'),
        order_count=('order_id', 'count'),
    ).reset_index()
    return agg
"""


def get_hard_input_df() -> dict:
    np.random.seed(7)
    n_products = 15
    n_orders   = 80
    products = pd.DataFrame({
        "product_id":   [f"PROD-{i:03d}" for i in range(n_products)],
        "product_name": [f"Product {i}"  for i in range(n_products)],
        "category":     np.random.choice(["Electronics","Clothing","Food","Home"], n_products).tolist(),
        "unit_price":   [f"{p:.2f}" for p in np.random.uniform(10.0, 300.0, n_products)],
        "cost_price":   [f"{p:.2f}" for p in np.random.uniform(5.0,  150.0, n_products)],
    })
    orders = pd.DataFrame({
        "order_id":    [f"ORD-{i:04d}" for i in range(n_orders)],
        "customer_id": [f"CUST-{i % 20:02d}" for i in range(n_orders)],
        "product_id":  np.random.choice(products["product_id"].tolist(), n_orders).tolist(),
        "quantity":    [str(q) for q in np.random.randint(1, 20, n_orders)],
        "order_date":  pd.date_range("2024-01-01", periods=n_orders, freq="4D").strftime("%Y-%m-%d").tolist(),
    })
    return {"orders": orders, "products": products}


def get_hard_expected_df() -> pd.DataFrame:
    data     = get_hard_input_df()
    orders   = data["orders"].copy()
    products = data["products"].copy()
    orders["quantity"]       = orders["quantity"].astype(int)
    products["unit_price"]   = products["unit_price"].astype(float)
    products["cost_price"]   = products["cost_price"].astype(float)
    orders["order_date"]     = pd.to_datetime(orders["order_date"])
    orders["year"]           = orders["order_date"].dt.year
    orders["month"]          = orders["order_date"].dt.month
    merged                   = orders.merge(products, on="product_id", how="left")
    merged["revenue"]        = merged["quantity"] * merged["unit_price"]
    merged["profit"]         = merged["quantity"] * (merged["unit_price"] - merged["cost_price"])
    agg = merged.groupby(["category","year","month"]).agg(
        total_revenue=("revenue",   "sum"),
        total_profit= ("profit",    "sum"),
        order_count=  ("order_id",  "count"),
        avg_quantity= ("quantity",  "mean"),
    ).reset_index()
    agg["total_revenue"] = agg["total_revenue"].round(2)
    agg["total_profit"]  = agg["total_profit"].round(2)
    agg["avg_quantity"]  = agg["avg_quantity"].round(2)
    return agg.sort_values(["year","month","category"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# UNIVERSAL GRADER
# ─────────────────────────────────────────────

def grade_output(
    result_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    task_id: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Grades result_df against expected_df.
    Returns (total_score 0.0–1.0, breakdown dict).

    Weights: schema 25%, row_count 15%, dtype 20%, value 40%
    """
    breakdown = {"schema_match": 0.0, "row_count_match": 0.0, "dtype_match": 0.0, "value_match": 0.0}

    if result_df is None or not isinstance(result_df, pd.DataFrame):
        return 0.0, breakdown

    expected_cols = set(expected_df.columns)
    result_cols   = set(result_df.columns)
    col_overlap   = len(expected_cols & result_cols) / max(len(expected_cols), 1)
    breakdown["schema_match"] = round(col_overlap, 4)

    if col_overlap == 0:
        return 0.0, breakdown

    # Row count (within tolerance)
    exp_rows = len(expected_df)
    res_rows = len(result_df)
    if exp_rows > 0:
        breakdown["row_count_match"] = round(min(res_rows, exp_rows) / exp_rows, 4)

    common_cols = list(expected_cols & result_cols)
    exp = expected_df[common_cols].reset_index(drop=True)
    res = result_df[common_cols].reset_index(drop=True)
    min_rows = min(len(exp), len(res))

    if min_rows == 0:
        # Can't compare values on empty DataFrames
        breakdown["dtype_match"] = 1.0 if set(result_df.dtypes[common_cols]) == set(expected_df.dtypes[common_cols]) else 0.5
        breakdown["value_match"] = 0.0
        weights = {"schema_match": 0.25, "row_count_match": 0.15, "dtype_match": 0.20, "value_match": 0.40}
        total = round(min(sum(breakdown[k] * weights[k] for k in breakdown), 1.0), 4)
        return total, breakdown

    exp = exp.iloc[:min_rows]
    res = res.iloc[:min_rows]

    # Dtype match for numeric cols
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(expected_df[c])]
    if numeric_cols:
        breakdown["dtype_match"] = round(
            sum(1.0 if pd.api.types.is_numeric_dtype(res[c]) else 0.0 for c in numeric_cols)
            / len(numeric_cols), 4
        )
    else:
        breakdown["dtype_match"] = 1.0

    # Value match
    value_scores = []
    for col in common_cols:
        try:
            if pd.api.types.is_numeric_dtype(expected_df[col]):
                ev = pd.to_numeric(exp[col], errors="coerce").fillna(0)
                rv = pd.to_numeric(res[col], errors="coerce").fillna(0)
                value_scores.append(float(np.isclose(ev, rv, rtol=0.01, atol=0.01).mean()))
            elif pd.api.types.is_datetime64_any_dtype(expected_df[col]):
                value_scores.append(float((pd.to_datetime(exp[col], errors="coerce") ==
                                           pd.to_datetime(res[col], errors="coerce")).mean()))
            else:
                value_scores.append(float(
                    (exp[col].astype(str).str.strip().str.lower() ==
                     res[col].astype(str).str.strip().str.lower()).mean()
                ))
        except Exception:
            value_scores.append(0.0)

    breakdown["value_match"] = round(sum(value_scores) / len(value_scores), 4) if value_scores else 0.0

    weights = {"schema_match": 0.25, "row_count_match": 0.15, "dtype_match": 0.20, "value_match": 0.40}
    total   = round(min(sum(breakdown[k] * weights[k] for k in breakdown), 1.0), 4)
    return total, breakdown


def safe_execute_code(code: str, input_data, task_id: str):
    """
    Safely execute agent code.
    Returns (result_df | None, error_msg | None).
    """
    local_ns = {"pd": pd, "np": np}
    try:
        exec(compile(code, "<agent_code>", "exec"), local_ns)   # noqa: S102
    except Exception:
        return None, f"Compilation error:\n{traceback.format_exc()}"

    fix_fn = local_ns.get("fix_pipeline")
    if fix_fn is None:
        return None, "Error: `fix_pipeline` function not defined in submitted code."

    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fix_fn(input_data)
        if not isinstance(result, pd.DataFrame):
            return None, f"Error: fix_pipeline must return pd.DataFrame, got {type(result).__name__}"
        return result, None
    except Exception:
        return None, f"Runtime error:\n{traceback.format_exc()}"


# ─────────────────────────────────────────────
# TASK REGISTRY — 3 built-in tasks for hackathon evaluation
# ─────────────────────────────────────────────

TASKS = {
    "easy": {
        "description": TASK_EASY_DESCRIPTION,
        "broken_code": TASK_EASY_BROKEN_CODE,
        "get_input": get_easy_input_df,
        "get_expected": get_easy_expected_df,
        "input_is_dict": False,
        "max_steps": 10,
    },
    "medium": {
        "description": TASK_MEDIUM_DESCRIPTION,
        "broken_code": TASK_MEDIUM_BROKEN_CODE,
        "get_input": get_medium_input_df,
        "get_expected": get_medium_expected_df,
        "input_is_dict": False,
        "max_steps": 15,
    },
    "hard": {
        "description": TASK_HARD_DESCRIPTION,
        "broken_code": TASK_HARD_BROKEN_CODE,
        "get_input": get_hard_input_df,
        "get_expected": get_hard_expected_df,
        "input_is_dict": True,
        "max_steps": 20,
    },
}
