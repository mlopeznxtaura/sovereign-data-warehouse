"""
sovereign-data-warehouse — Entry Point

Self-hosted GPU-accelerated data warehouse: ingest, query, transform, validate.

Usage:
  python main.py --mode ingest --source ./data/events.csv --table events
  python main.py --mode query --sql "SELECT COUNT(*) FROM events"
  python main.py --mode gpu --table events
  python main.py --mode validate --table events
  python main.py --mode dbt --action run
  python main.py --mode dagster  (launch Dagster UI)
  python main.py --mode dash     (launch Streamlit dashboard)
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sovereign Data Warehouse")
    parser.add_argument("--mode", required=True,
                        choices=["ingest", "query", "gpu", "validate", "dbt", "dagster", "dash"])
    parser.add_argument("--source", help="CSV/Parquet input file")
    parser.add_argument("--table", default="events", help="Table name in lake")
    parser.add_argument("--sql", help="SQL query to run (query mode)")
    parser.add_argument("--action", default="run", help="dbt action: run/test/build/docs")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--minio", default="localhost:9000")
    parser.add_argument("--bucket", default="warehouse")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Sovereign Data Warehouse")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60)

    if args.mode == "ingest":
        from storage.minio_lake import MinIOLake, LakeConfig
        from quality.great_expectations_validator import DataQualityValidator
        import polars as pl

        lake = MinIOLake(LakeConfig(endpoint=args.minio, default_bucket=args.bucket))
        if args.source and Path(args.source).exists():
            df = pl.read_csv(args.source) if args.source.endswith(".csv") else pl.read_parquet(args.source)
        else:
            import numpy as np
            rng = np.random.default_rng(42)
            n = 50_000
            df = pl.DataFrame({
                "id": list(range(n)),
                "user_id": [f"u{rng.integers(1,1000):04d}" for _ in range(n)],
                "event_type": rng.choice(["click","view","purchase"], n).tolist(),
                "amount": rng.exponential(50, n).round(2).tolist(),
                "ts": [f"2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for _ in range(n)],
            })
            print(f"Generated {len(df)} synthetic rows")

        validator = DataQualityValidator()
        result = validator.run_suite(df, {"not_null": ["id"], "min_rows": 100})
        if result["success"]:
            lake.write_parquet(df, f"{args.table}/data_001.parquet")
            print(f"Ingested {len(df):,} rows -> {args.bucket}/{args.table}")
        else:
            print("Validation failed — data not written")

    elif args.mode == "query":
        from query.duckdb_engine import DuckDBEngine
        engine = DuckDBEngine(minio_endpoint=args.minio)
        sql = args.sql or f"SELECT COUNT(*) as n FROM read_parquet('s3://{args.bucket}/{args.table}/**/*.parquet')"
        df = engine.query(sql)
        print(df)
        engine.close()

    elif args.mode == "gpu":
        from storage.minio_lake import MinIOLake, LakeConfig
        from gpu.rapids_analytics import RAPIDSAnalytics
        import numpy as np

        analytics = RAPIDSAnalytics()
        rng = np.random.default_rng(42)
        n = 100_000
        df = __import__("polars").DataFrame({
            "user_id": [f"u{rng.integers(1,500):04d}" for _ in range(n)],
            "amount": rng.exponential(50, n).round(2).tolist(),
            "score": rng.uniform(0, 1, n).tolist(),
        })
        print("
Descriptive stats (GPU):")
        print(analytics.describe_gpu(df))
        clustered = analytics.kmeans_cluster(df, ["amount", "score"], n_clusters=5)
        print(f"
K-Means clustering done: {clustered['cluster'].n_unique()} clusters")

    elif args.mode == "validate":
        import polars as pl, numpy as np
        from quality.great_expectations_validator import DataQualityValidator
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"id": list(range(1000)), "amount": rng.exponential(50, 1000).tolist()})
        validator = DataQualityValidator()
        result = validator.run_suite(df, {
            "not_null": ["id", "amount"],
            "unique": ["id"],
            "ranges": [{"col": "amount", "min": 0}],
            "min_rows": 100,
        })
        print(f"
Validation: {'PASSED' if result['success'] else 'FAILED'}")

    elif args.mode == "dbt":
        from transforms.dbt_runner import DBTRunner
        runner = DBTRunner(project_dir=f"{args.output}/dbt_project")
        runner.init_project()
        getattr(runner, args.action)()

    elif args.mode == "dagster":
        import subprocess
        subprocess.run(["dagster", "dev", "-f", "orchestration/dagster_assets.py"])

    elif args.mode == "dash":
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
