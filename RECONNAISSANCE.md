# RECONNAISSANCE.md: jaffle_shop Manual Analysis

## Codebase: dbt-labs/jaffle_shop
**Repository**: https://github.com/dbt-labs/jaffle_shop

---

## 1. Primary Data Ingestion Path
The project uses **dbt seeds** as its primary ingestion mechanism. Raw data is stored as CSV files in the `seeds/` directory:
- `raw_customers.csv`
- `raw_orders.csv`
- `raw_payments.csv`

These are loaded into the database and then referenced by the **staging models** (`stg_customers.sql`, `stg_orders.sql`, `stg_payments.sql`) using the `{{ ref() }}` function.

## 2. Critical Output Datasets
The core output datasets are the final transformed models located in the `models/` root:
- **`customers`**: A dimension table containing customer-level attributes and aggregated order history.
- **`orders`**: A fact table containing order details, including payment information.

## 3. Blast Radius Analysis
- **High Impact**: Failure in `stg_orders` or `stg_payments`. Both `orders` and `customers` final models depend on these.
- **Medium Impact**: Failure in `stg_customers`. Affects the `customers` model, but the `orders` model (primarily sales/payments) remains partially functional.

## 4. Business Logic Concentration
Business logic is heavily **concentrated within the models** via SQL CTEs.
- Staging models handle light cleaning.
- Core models (`customers.sql`, `orders.sql`) handle complex aggregations (e.g., life-time value, number of orders).
- The `macros/` directory exists but is not used for core transformations in this basic project.

## 5. Git Velocity (90 Days)
**Activity**: Zero velocity.
- The repo was archived on February 10, 2025.
- Last significant logic changes were in April 2024.

---

## Difficulty Analysis: The Manual Bottleneck
- **What was hardest to figure out?** Tracing "blast radius". I had to open multiple files (`orders.sql`, `customers.sql`) and look for `{{ ref() }}` calls to see what depends on what.
- **Where did I get lost?** Initially searching for "sources" (e.g., in `sources.yml`). In this project, the "sources" are actually "seeds", which is a different pattern than many production systems.
- **Conclusion**: A tool that automatically parses `ref()` and builds a graph would save ~15 minutes of manual clicking for even this small project. For 800k LOC, it's impossible manually.
