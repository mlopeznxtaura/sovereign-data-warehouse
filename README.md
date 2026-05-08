# Sovereign Data Warehouse

Cluster 12 of the NextAura 500 SDKs / 25 Clusters project.

Self-hosted, GPU-accelerated analytical platform that replaces Snowflake + Databricks. Fully open-source, runs on your own hardware.

## Architecture

- DuckDB as the primary SQL query engine (ad-hoc analytics over Parquet/Arrow)
- RAPIDS cuDF + cuML for GPU-accelerated dataframe operations and ML
- Apache Arrow + Polars for in-memory columnar processing
- Kafka + Flink for streaming ingestion into the lake
- MinIO for S3-compatible object storage (Parquet/Delta Lake)
- dbt Core for SQL transformation pipelines
- Dagster + Prefect for pipeline orchestration
- Great Expectations for data quality enforcement
- Plotly Dash + Streamlit for interactive analytics dashboards
- OpenTelemetry + Prometheus + Grafana for full observability

## SDKs Used

DuckDB, Apache Arrow, Polars, Apache Spark SDK, Apache Kafka SDK, Apache Flink SDK, dbt Core, RAPIDS cuDF, RAPIDS cuML, Dagster SDK, Prefect SDK, MinIO SDK, Great Expectations, OpenTelemetry SDK, Prometheus Client, Grafana SDK, FastAPI, Plotly Dash SDK, Streamlit SDK, SQLAlchemy

## Quickstart

```bash
pip install -r requirements.txt
docker-compose up -d  # starts MinIO, Kafka, Grafana

# Write data to lake
python main.py --mode ingest --source ./data/events.csv --table events

# Query with DuckDB
python main.py --mode query --sql "SELECT COUNT(*) FROM events WHERE date > '2025-01-01'"

# Run GPU analytics
python main.py --mode gpu --table events --op describe

# Launch dashboard
python main.py --mode dash
```

## Structure

```
storage/        MinIO lake client, Parquet write/read
query/          DuckDB engine, Arrow/Polars query layer
gpu/            RAPIDS cuDF + cuML GPU analytics
streaming/      Kafka + Flink ingestion pipeline
transforms/     dbt Core SQL transformation runner
orchestration/  Dagster asset graph + Prefect flow
quality/        Great Expectations data validation
observability/  OpenTelemetry + Prometheus metrics
dashboard/      Plotly Dash + Streamlit apps
api/            FastAPI query and admin endpoints
main.py         CLI entry point
```
