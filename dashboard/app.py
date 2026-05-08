"""
Streamlit dashboard for Sovereign Data Warehouse.
Live query interface, metrics, and GPU analytics visualization.
SDKs: Streamlit, Plotly, DuckDB, Polars
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np
import time

st.set_page_config(page_title="Sovereign Data Warehouse", layout="wide", page_icon="🏛️")

st.title("🏛️ Sovereign Data Warehouse")
st.caption("Self-hosted GPU-accelerated analytics — powered by DuckDB + RAPIDS + MinIO")

with st.sidebar:
    st.header("Connection")
    minio_endpoint = st.text_input("MinIO Endpoint", value="localhost:9000")
    bucket = st.text_input("Bucket", value="warehouse")
    table = st.text_input("Table", value="events")
    st.divider()
    st.header("Query")
    custom_sql = st.text_area("Custom SQL", value=f"SELECT event_type, COUNT(*) as n FROM events GROUP BY 1 ORDER BY n DESC LIMIT 10")
    run_query = st.button("Run Query")

# Generate demo data
@st.cache_data
def get_demo_data():
    rng = np.random.default_rng(42)
    n = 10_000
    return pl.DataFrame({
        "date": pl.Series([f"2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for _ in range(n)]),
        "event_type": rng.choice(["click","view","purchase","signup"], n).tolist(),
        "user_id": [f"u{rng.integers(1,500):04d}" for _ in range(n)],
        "amount": rng.exponential(50, n).round(2).tolist(),
    })

df = get_demo_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{len(df):,}")
col2.metric("Unique Users", f"{df['user_id'].n_unique():,}")
col3.metric("Total Revenue", f"${df['amount'].sum():,.0f}")
col4.metric("Avg Order Value", f"${df.filter(pl.col('event_type')=='purchase')['amount'].mean():.2f}")

st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Events by Type")
    type_counts = df.group_by("event_type").agg(pl.count("event_type").alias("count")).sort("count", descending=True)
    fig = px.bar(type_counts.to_pandas(), x="event_type", y="count", color="event_type",
                 template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Revenue Distribution")
    purchase_df = df.filter(pl.col("event_type") == "purchase")
    fig2 = px.histogram(purchase_df.to_pandas(), x="amount", nbins=50,
                        template="plotly_dark", color_discrete_sequence=["#00cc96"])
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Query Results")
if run_query:
    try:
        import duckdb
        con = duckdb.connect()
        con.register("events", df.to_arrow())
        result = con.execute(custom_sql).pl()
        st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Query error: {e}")
else:
    st.info("Enter a SQL query and click Run Query")
