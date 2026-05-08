"""
dbt Core SQL transformation runner.
Execute dbt models, tests, and snapshots against DuckDB or other warehouses.
SDKs: dbt-core, dbt-duckdb, SQLAlchemy
"""
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


DBT_PROJECT_TEMPLATE = """
name: sovereign_warehouse
version: '1.0.0'
config-version: 2

profile: sovereign_warehouse

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  sovereign_warehouse:
    staging:
      +materialized: view
    marts:
      +materialized: table
"""

DBT_PROFILES_TEMPLATE = """
sovereign_warehouse:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: '{db_path}'
      threads: 4
    prod:
      type: duckdb
      path: '{prod_db_path}'
      threads: 8
"""

STAGING_MODEL_EXAMPLE = """
-- models/staging/stg_events.sql
-- Staging model: clean and type-cast raw events

{{ config(materialized='view') }}

SELECT
    CAST(event_id AS VARCHAR)       AS event_id,
    CAST(user_id AS VARCHAR)        AS user_id,
    CAST(event_type AS VARCHAR)     AS event_type,
    CAST(created_at AS TIMESTAMP)   AS created_at,
    CAST(properties AS VARCHAR)     AS properties,
    DATE_TRUNC('day', CAST(created_at AS TIMESTAMP)) AS event_date
FROM {{ source('raw', 'events') }}
WHERE event_id IS NOT NULL
"""

MART_MODEL_EXAMPLE = """
-- models/marts/user_activity_daily.sql
-- Daily user activity summary mart

{{ config(materialized='table') }}

WITH events AS (
    SELECT * FROM {{ ref('stg_events') }}
),
daily AS (
    SELECT
        event_date,
        user_id,
        COUNT(*)                                      AS total_events,
        COUNT(DISTINCT event_type)                    AS distinct_event_types,
        MIN(created_at)                               AS first_event_ts,
        MAX(created_at)                               AS last_event_ts
    FROM events
    GROUP BY event_date, user_id
)
SELECT
    event_date,
    user_id,
    total_events,
    distinct_event_types,
    first_event_ts,
    last_event_ts,
    DATEDIFF('second', first_event_ts, last_event_ts) AS session_duration_sec
FROM daily
ORDER BY event_date DESC, total_events DESC
"""


class DBTRunner:
    """
    Programmatic dbt Core runner. Initialize project, run models, test quality.
    """

    def __init__(self, project_dir: str = "./dbt_project", db_path: str = "./warehouse.duckdb"):
        self.project_dir = Path(project_dir)
        self.db_path = db_path
        self.profiles_dir = self.project_dir / "profiles"

    def init_project(self, project_name: str = "sovereign_warehouse"):
        """Scaffold a dbt project with DuckDB profile."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models" / "staging").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "models" / "marts").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "tests").mkdir(exist_ok=True)
        (self.project_dir / "seeds").mkdir(exist_ok=True)
        self.profiles_dir.mkdir(exist_ok=True)

        # Write project file
        (self.project_dir / "dbt_project.yml").write_text(DBT_PROJECT_TEMPLATE)

        # Write profiles
        profiles_content = DBT_PROFILES_TEMPLATE.format(
            db_path=str(Path(self.db_path).resolve()),
            prod_db_path=str(Path(self.db_path).resolve()).replace(".duckdb", "_prod.duckdb"),
        )
        (self.profiles_dir / "profiles.yml").write_text(profiles_content)

        # Write example models
        (self.project_dir / "models" / "staging" / "stg_events.sql").write_text(STAGING_MODEL_EXAMPLE)
        (self.project_dir / "models" / "marts" / "user_activity_daily.sql").write_text(MART_MODEL_EXAMPLE)

        print(f"[dbt] Project initialized at {self.project_dir}")

    def _run_dbt(self, *args) -> Dict[str, Any]:
        """Run a dbt command and return result."""
        cmd = [
            "dbt", *args,
            "--project-dir", str(self.project_dir),
            "--profiles-dir", str(self.profiles_dir),
        ]
        print(f"[dbt] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0
        if not success:
            print(f"[dbt] Error: {result.stderr[-500:]}")
        else:
            print(f"[dbt] Success")
        return {"success": success, "stdout": result.stdout, "stderr": result.stderr}

    def run(self, select: Optional[str] = None, target: str = "dev") -> Dict:
        args = ["run", "--target", target]
        if select:
            args += ["--select", select]
        return self._run_dbt(*args)

    def test(self, select: Optional[str] = None) -> Dict:
        args = ["test"]
        if select:
            args += ["--select", select]
        return self._run_dbt(*args)

    def build(self, select: Optional[str] = None) -> Dict:
        """Run models + tests in one command."""
        args = ["build"]
        if select:
            args += ["--select", select]
        return self._run_dbt(*args)

    def generate_docs(self) -> Dict:
        return self._run_dbt("docs", "generate")

    def seed(self) -> Dict:
        return self._run_dbt("seed")
