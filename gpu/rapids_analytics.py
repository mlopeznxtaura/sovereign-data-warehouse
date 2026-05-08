"""
RAPIDS cuDF + cuML GPU-accelerated analytics.
Drop-in GPU replacement for pandas/scikit-learn — 10-100x faster.
SDKs: RAPIDS cuDF, cuML, PyArrow, Polars
"""
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

try:
    import cudf
    import cuml
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.ensemble import RandomForestClassifier as cuRFC
    from cuml.linear_model import LogisticRegression as cuLR
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    print("Warning: RAPIDS not available. Install: pip install cudf-cu12 cuml-cu12")
    import pandas as cudf   # type: ignore — CPU fallback
    import sklearn as cuml  # type: ignore

import polars as pl
import pyarrow as pa


class RAPIDSAnalytics:
    """
    GPU-accelerated analytics using RAPIDS cuDF + cuML.
    Automatically falls back to CPU (pandas/sklearn) when GPU unavailable.
    """

    def __init__(self, device: int = 0):
        if RAPIDS_AVAILABLE:
            import rmm
            rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)
            print(f"[RAPIDS] GPU {device} ready | cuDF {cudf.__version__} | cuML {cuml.__version__}")
        else:
            print("[RAPIDS] Running on CPU (RAPIDS not available)")

    def from_polars(self, df: pl.DataFrame) -> "cudf.DataFrame":
        """Convert Polars DataFrame to cuDF GPU DataFrame."""
        if RAPIDS_AVAILABLE:
            return cudf.DataFrame.from_arrow(df.to_arrow())
        return df.to_pandas()

    def to_polars(self, gdf: "cudf.DataFrame") -> pl.DataFrame:
        """Convert cuDF GPU DataFrame back to Polars."""
        if RAPIDS_AVAILABLE:
            return pl.from_arrow(gdf.to_arrow())
        return pl.from_pandas(gdf)

    def describe_gpu(self, df: pl.DataFrame) -> pl.DataFrame:
        """GPU-accelerated descriptive statistics."""
        gdf = self.from_polars(df)
        desc = gdf.describe()
        return self.to_polars(desc)

    def group_agg(
        self,
        df: pl.DataFrame,
        group_cols: List[str],
        agg_col: str,
        agg_fns: List[str] = ["mean", "std", "min", "max", "count"],
    ) -> pl.DataFrame:
        """GPU-accelerated groupby aggregation."""
        gdf = self.from_polars(df)
        agg_dict = {agg_col: agg_fns}
        result = gdf.groupby(group_cols).agg(agg_dict)
        result.columns = group_cols + [f"{agg_col}_{fn}" for fn in agg_fns]
        return self.to_polars(result)

    def kmeans_cluster(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 5,
        max_iter: int = 300,
        seed: int = 42,
    ) -> pl.DataFrame:
        """GPU-accelerated K-Means clustering. Returns df with 'cluster' column."""
        gdf = self.from_polars(df.select(feature_cols))

        if RAPIDS_AVAILABLE:
            scaler = cuStandardScaler()
            X = scaler.fit_transform(gdf)
            kmeans = cuKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=seed)
            labels = kmeans.fit_predict(X)
            result = self.from_polars(df)
            result["cluster"] = labels
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            X = StandardScaler().fit_transform(gdf)
            labels = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(X)
            result = df.to_pandas()
            result["cluster"] = labels

        return self.to_polars(result)

    def pca_reduce(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        n_components: int = 2,
    ) -> pl.DataFrame:
        """GPU-accelerated PCA dimensionality reduction."""
        gdf = self.from_polars(df.select(feature_cols))

        if RAPIDS_AVAILABLE:
            pca = cuPCA(n_components=n_components)
            components = pca.fit_transform(gdf)
            result = cudf.DataFrame(
                components.values,
                columns=[f"pc_{i+1}" for i in range(n_components)]
            )
            explained = pca.explained_variance_ratio_.tolist()
        else:
            from sklearn.decomposition import PCA
            import pandas as pd
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(gdf)
            result = pd.DataFrame(components, columns=[f"pc_{i+1}" for i in range(n_components)])
            explained = pca.explained_variance_ratio_.tolist()

        print(f"[RAPIDS] PCA: explained variance = {[round(e,3) for e in explained]}")
        return self.to_polars(result)

    def random_forest_classify(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        n_estimators: int = 100,
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """GPU-accelerated Random Forest classification."""
        X_train = self.from_polars(train_df.select(feature_cols))
        y_train = self.from_polars(train_df.select([label_col]))[label_col]

        if RAPIDS_AVAILABLE:
            clf = cuRFC(n_estimators=n_estimators, max_depth=max_depth)
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        clf.fit(X_train, y_train)
        print(f"[RAPIDS] RF trained: {n_estimators} trees, depth={max_depth}")
        return {"model": clf, "feature_cols": feature_cols, "label_col": label_col}

    def window_stats(
        self,
        df: pl.DataFrame,
        value_col: str,
        window_size: int = 7,
    ) -> pl.DataFrame:
        """Rolling window statistics (mean, std, min, max) on GPU."""
        gdf = self.from_polars(df)
        if RAPIDS_AVAILABLE:
            gdf[f"{value_col}_roll_mean"] = gdf[value_col].rolling(window_size).mean()
            gdf[f"{value_col}_roll_std"] = gdf[value_col].rolling(window_size).std()
            gdf[f"{value_col}_roll_min"] = gdf[value_col].rolling(window_size).min()
            gdf[f"{value_col}_roll_max"] = gdf[value_col].rolling(window_size).max()
        else:
            gdf[f"{value_col}_roll_mean"] = gdf[value_col].rolling(window_size).mean()
            gdf[f"{value_col}_roll_std"] = gdf[value_col].rolling(window_size).std()
        return self.to_polars(gdf)
