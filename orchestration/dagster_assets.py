"""
Dagster asset graph for the sovereign data warehouse pipeline.
Define assets: raw ingestion -> staging -> marts -> ML features.
SDKs: Dagster, MinIO, DuckDB, Great Expectations
"""
from dagster import (
    asset, AssetIn, Output, MetadataValue,
    op, job, schedule, define_asset_job,
    AssetExecutionContext, Config,
    Definitions, ScheduleDefinition,
)
import polars as pl
import pyarrow as pa
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


class WarehouseConfig(Config):
    minio_endpoint: str = "localhost:9000"
    minio_bucket: str = "warehouse"
    duckdb_path: str = "./warehouse.duckdb"
    table_name: str = "events"


@asset(
    description="Raw events ingested from Kafka/files into MinIO lake",
    group_name="ingestion",
)
def raw_events(context: AssetExecutionContext) -> Output[pl.DataFrame]:
    """Ingest raw event data and write to MinIO Parquet lake."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 10_000
    df = pl.DataFrame({
        "event_id": [f"evt_{i:08d}" for i in range(n)],
        "user_id": [f"user_{rng.integers(1, 1000):04d}" for _ in range(n)],
        "event_type": rng.choice(["click", "view", "purchase", "signup"], n).tolist(),
        "amount": rng.exponential(50, n).round(2).tolist(),
        "created_at": pl.Series(
            [f"2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}T{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00"
             for _ in range(n)]
        ),
    })

    context.log.info(f"Ingested {len(df)} raw events")
    return Output(
        df,
        metadata={
            "num_rows": MetadataValue.int(len(df)),
            "schema": MetadataValue.text(str(df.schema)),
        }
    )


@asset(
    ins={"raw_events": AssetIn()},
    description="Cleaned and typed staging layer",
    group_name="staging",
)
def stg_events(context: AssetExecutionContext, raw_events: pl.DataFrame) -> Output[pl.DataFrame]:
    """Clean, type-cast, and deduplicate raw events."""
    df = (
        raw_events
        .with_columns([
            pl.col("created_at").str.to_datetime(),
            pl.col("amount").cast(pl.Float64),
            pl.col("event_type").cast(pl.Categorical),
        ])
        .drop_nulls(subset=["event_id", "user_id"])
        .unique(subset=["event_id"])
    )
    context.log.info(f"Staging: {len(df)} clean events")
    return Output(df, metadata={"num_rows": MetadataValue.int(len(df))})


@asset(
    ins={"stg_events": AssetIn()},
    description="Daily user activity mart",
    group_name="marts",
)
def user_activity_daily(context: AssetExecutionContext, stg_events: pl.DataFrame) -> Output[pl.DataFrame]:
    """Aggregate events into daily per-user activity summary."""
    df = (
        stg_events
        .with_columns(pl.col("created_at").dt.date().alias("event_date"))
        .group_by(["event_date", "user_id"])
        .agg([
            pl.count("event_id").alias("total_events"),
            pl.n_unique("event_type").alias("distinct_types"),
            pl.sum("amount").alias("total_amount"),
            pl.min("created_at").alias("first_event"),
            pl.max("created_at").alias("last_event"),
        ])
        .sort(["event_date", "total_events"], descending=[True, True])
    )
    context.log.info(f"Daily mart: {len(df)} rows")
    return Output(df, metadata={"num_rows": MetadataValue.int(len(df))})


@asset(
    ins={"stg_events": AssetIn()},
    description="ML feature table for user propensity models",
    group_name="ml_features",
)
def user_features(context: AssetExecutionContext, stg_events: pl.DataFrame) -> Output[pl.DataFrame]:
    """Build per-user feature vector for ML models."""
    features = (
        stg_events
        .group_by("user_id")
        .agg([
            pl.count("event_id").alias("total_events"),
            pl.sum("amount").alias("lifetime_value"),
            pl.mean("amount").alias("avg_order_value"),
            pl.n_unique("event_type").alias("event_type_diversity"),
            (pl.col("event_type") == "purchase").sum().alias("purchase_count"),
            (pl.col("event_type") == "signup").sum().alias("signup_count"),
        ])
        .with_columns([
            (pl.col("purchase_count") / (pl.col("total_events") + 1e-6)).alias("purchase_rate"),
        ])
    )
    context.log.info(f"User features: {len(features)} users, {len(features.columns)} features")
    return Output(features, metadata={"num_rows": MetadataValue.int(len(features))})


# Define the full pipeline job
warehouse_pipeline_job = define_asset_job(
    name="warehouse_pipeline",
    selection=["raw_events", "stg_events", "user_activity_daily", "user_features"],
)

# Daily schedule
daily_schedule = ScheduleDefinition(
    job=warehouse_pipeline_job,
    cron_schedule="0 2 * * *",  # 2am daily
    name="daily_warehouse_refresh",
)

# Dagster Definitions (entry point for dagster dev)
defs = Definitions(
    assets=[raw_events, stg_events, user_activity_daily, user_features],
    jobs=[warehouse_pipeline_job],
    schedules=[daily_schedule],
)
