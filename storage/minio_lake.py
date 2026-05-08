"""
MinIO data lake client. Write/read Parquet to S3-compatible object storage.
The foundation of the sovereign warehouse — everything lands here first.
SDKs: MinIO SDK, PyArrow, Polars
"""
import os
import io
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from minio import Minio
from minio.error import S3Error


@dataclass
class LakeConfig:
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False
    default_bucket: str = "warehouse"
    default_prefix: str = "raw"


class MinIOLake:
    """
    S3-compatible data lake backed by MinIO.
    Write Arrow tables as Parquet, read back with partition pruning.
    Supports Hive-style partitioning: table/year=2025/month=01/data.parquet
    """

    def __init__(self, config: Optional[LakeConfig] = None):
        self.cfg = config or LakeConfig(
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        )
        self.client = Minio(
            self.cfg.endpoint,
            access_key=self.cfg.access_key,
            secret_key=self.cfg.secret_key,
            secure=self.cfg.secure,
        )
        self._ensure_bucket(self.cfg.default_bucket)
        print(f"[MinIO] Connected to {self.cfg.endpoint}")

    def _ensure_bucket(self, bucket: str):
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
            print(f"[MinIO] Created bucket: {bucket}")

    def write_parquet(
        self,
        table: Union[pa.Table, pl.DataFrame],
        path: str,
        bucket: Optional[str] = None,
        compression: str = "snappy",
    ) -> str:
        """
        Write Arrow table or Polars DataFrame as Parquet to MinIO.
        path: object path e.g. 'events/year=2025/month=01/batch_001.parquet'
        Returns full object URI.
        """
        bucket = bucket or self.cfg.default_bucket
        self._ensure_bucket(bucket)

        if isinstance(table, pl.DataFrame):
            arrow_table = table.to_arrow()
        else:
            arrow_table = table

        buf = io.BytesIO()
        pq.write_table(arrow_table, buf, compression=compression)
        buf.seek(0)
        size = buf.getbuffer().nbytes

        self.client.put_object(
            bucket, path,
            data=buf, length=size,
            content_type="application/octet-stream",
        )
        uri = f"s3://{bucket}/{path}"
        print(f"[MinIO] Written: {uri} ({size/1024:.1f}KB, {len(arrow_table)} rows)")
        return uri

    def read_parquet(
        self,
        path: str,
        bucket: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Read a single Parquet object from MinIO into Polars."""
        bucket = bucket or self.cfg.default_bucket
        response = self.client.get_object(bucket, path)
        buf = io.BytesIO(response.read())
        response.close()
        table = pq.read_table(buf, columns=columns)
        return pl.from_arrow(table)

    def list_objects(self, prefix: str = "", bucket: Optional[str] = None, recursive: bool = True) -> List[str]:
        """List all objects in the lake under a prefix."""
        bucket = bucket or self.cfg.default_bucket
        objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
        return [obj.object_name for obj in objects]

    def write_partitioned(
        self,
        df: pl.DataFrame,
        table_name: str,
        partition_cols: List[str],
        bucket: Optional[str] = None,
    ) -> List[str]:
        """
        Write DataFrame with Hive-style partitioning.
        e.g. table_name='events', partition_cols=['year', 'month']
        -> events/year=2025/month=01/data_<ts>.parquet
        """
        bucket = bucket or self.cfg.default_bucket
        written = []
        groups = df.partition_by(partition_cols, maintain_order=True)

        for group in groups:
            part_vals = {col: str(group[col][0]) for col in partition_cols}
            part_path = "/".join(f"{k}={v}" for k, v in part_vals.items())
            obj_path = f"{table_name}/{part_path}/data_{int(time.time()*1000)}.parquet"
            uri = self.write_parquet(group, obj_path, bucket=bucket)
            written.append(uri)

        print(f"[MinIO] Wrote {len(written)} partitions for {table_name}")
        return written

    def delete_object(self, path: str, bucket: Optional[str] = None):
        bucket = bucket or self.cfg.default_bucket
        self.client.remove_object(bucket, path)

    def get_table_stats(self, table_name: str, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Return size + object count for a logical table."""
        bucket = bucket or self.cfg.default_bucket
        objects = self.client.list_objects(bucket, prefix=f"{table_name}/", recursive=True)
        total_size = 0
        count = 0
        for obj in objects:
            total_size += obj.size or 0
            count += 1
        return {"table": table_name, "objects": count, "total_bytes": total_size,
                "total_mb": round(total_size / 1e6, 2)}
