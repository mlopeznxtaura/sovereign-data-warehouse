"""
DuckDB query engine — query Parquet on MinIO directly with SQL.
No data movement needed: DuckDB scans S3/MinIO in-place via httpfs.
SDKs: DuckDB, PyArrow, Polars
"""
import os
import time
from typing import Optional, List, Dict, Any, Union

import duckdb
import pyarrow as pa
import polars as pl


class DuckDBEngine:
    """
    DuckDB-based SQL engine over the MinIO data lake.
    Queries Parquet files directly on S3 without loading into memory first.
    Supports Arrow/Polars result materialization.
    """

    def __init__(
        self,
        minio_endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        database: str = ":memory:",
        threads: int = 0,          # 0 = auto (all cores)
        memory_limit: str = "8GB",
    ):
        self.con = duckdb.connect(database=database)
        self._configure(minio_endpoint, access_key, secret_key, threads, memory_limit)
        print(f"[DuckDB] Engine ready | memory_limit={memory_limit} | threads={threads or 'auto'}")

    def _configure(self, endpoint, access_key, secret_key, threads, memory_limit):
        """Configure DuckDB httpfs extension for MinIO access."""
        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        self.con.execute(f"SET s3_endpoint='{endpoint}';")
        self.con.execute(f"SET s3_access_key_id='{access_key}';")
        self.con.execute(f"SET s3_secret_access_key='{secret_key}';")
        self.con.execute("SET s3_use_ssl=false;")
        self.con.execute("SET s3_url_style='path';")
        if threads:
            self.con.execute(f"SET threads={threads};")
        self.con.execute(f"SET memory_limit='{memory_limit}';")

    def query(self, sql: str, as_polars: bool = True) -> Union[pl.DataFrame, pa.Table]:
        """Execute SQL and return results as Polars DataFrame or Arrow Table."""
        t0 = time.perf_counter()
        result = self.con.execute(sql)
        arrow_table = result.arrow()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[DuckDB] {len(arrow_table)} rows in {elapsed:.1f}ms")
        return pl.from_arrow(arrow_table) if as_polars else arrow_table

    def query_lake(
        self,
        bucket: str,
        table_path: str,
        sql_where: str = "1=1",
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Query a Parquet table on MinIO lake using glob pattern.
        Automatically scans all partitions under table_path.
        """
        col_expr = ", ".join(columns) if columns else "*"
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT {col_expr}
            FROM read_parquet('s3://{bucket}/{table_path}/**/*.parquet', hive_partitioning=true)
            WHERE {sql_where}
            {limit_clause}
        """
        return self.query(sql)

    def create_view(self, view_name: str, bucket: str, table_path: str):
        """Register a MinIO table as a persistent DuckDB view."""
        self.con.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('s3://{bucket}/{table_path}/**/*.parquet',
                                       hive_partitioning=true)
        """)
        print(f"[DuckDB] View created: {view_name}")

    def register_arrow(self, table_name: str, table: Union[pa.Table, pl.DataFrame]):
        """Register an in-memory Arrow/Polars table for SQL queries."""
        if isinstance(table, pl.DataFrame):
            table = table.to_arrow()
        self.con.register(table_name, table)

    def explain(self, sql: str) -> str:
        """Return query plan for a SQL statement."""
        result = self.con.execute(f"EXPLAIN {sql}").fetchdf()
        return result.to_string()

    def profile(self, sql: str) -> Dict[str, Any]:
        """Run query and return timing + row count."""
        t0 = time.perf_counter()
        df = self.query(sql)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "sql": sql[:100],
            "rows": len(df),
            "elapsed_ms": round(elapsed, 2),
            "columns": df.columns,
        }

    def export_parquet(self, sql: str, output_path: str, compression: str = "snappy"):
        """Run SQL and write result directly to Parquet."""
        self.con.execute(f"COPY ({sql}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION {compression})")
        print(f"[DuckDB] Exported to {output_path}")

    def import_csv(self, csv_path: str, table_name: str, auto_detect: bool = True):
        """Import CSV into DuckDB table (in-memory or file)."""
        self.con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=true)
        """)
        count = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"[DuckDB] Imported {count:,} rows -> {table_name}")

    def close(self):
        self.con.close()
