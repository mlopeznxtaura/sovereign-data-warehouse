"""
Great Expectations data quality enforcement.
Validate DataFrames against expectation suites before writing to lake.
SDKs: Great Expectations, Polars, PyArrow
"""
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

import polars as pl
import pyarrow as pa

try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    print("Warning: great-expectations not available. Install: pip install great-expectations")


@dataclass
class ExpectationResult:
    expectation_type: str
    column: Optional[str]
    success: bool
    observed_value: Any
    expected: Any
    details: Dict = field(default_factory=dict)


class DataQualityValidator:
    """
    Validate DataFrames against expectation suites.
    Run before writing to lake to prevent bad data from entering the warehouse.
    """

    def __init__(self, context_root: str = "./gx_context"):
        self.context_root = Path(context_root)
        if GX_AVAILABLE:
            self.context = gx.get_context(context_root_dir=str(self.context_root))
        print(f"[GX] Validator ready {'(GX)' if GX_AVAILABLE else '(manual fallback)'}")

    def validate_schema(
        self, df: pl.DataFrame, expected_schema: Dict[str, str]
    ) -> List[ExpectationResult]:
        """
        Validate that a DataFrame has the expected column names and types.
        expected_schema: {col_name: dtype_str} e.g. {'user_id': 'Utf8', 'amount': 'Float64'}
        """
        results = []
        actual_schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

        for col, expected_dtype in expected_schema.items():
            if col not in actual_schema:
                results.append(ExpectationResult(
                    expectation_type="expect_column_to_exist",
                    column=col, success=False,
                    observed_value=None, expected=col,
                ))
            else:
                dtype_ok = expected_dtype.lower() in actual_schema[col].lower()
                results.append(ExpectationResult(
                    expectation_type="expect_column_values_to_be_of_type",
                    column=col, success=dtype_ok,
                    observed_value=actual_schema[col], expected=expected_dtype,
                ))
        return results

    def validate_not_null(self, df: pl.DataFrame, columns: List[str]) -> List[ExpectationResult]:
        """Check that specified columns have no null values."""
        results = []
        for col in columns:
            if col not in df.columns:
                results.append(ExpectationResult(
                    "expect_column_values_to_not_be_null", col, False, "column missing", None
                ))
                continue
            null_count = df[col].null_count()
            results.append(ExpectationResult(
                expectation_type="expect_column_values_to_not_be_null",
                column=col,
                success=null_count == 0,
                observed_value=null_count,
                expected=0,
            ))
        return results

    def validate_unique(self, df: pl.DataFrame, columns: List[str]) -> List[ExpectationResult]:
        """Check that specified columns have unique values."""
        results = []
        for col in columns:
            if col not in df.columns:
                continue
            n_unique = df[col].n_unique()
            n_total = len(df)
            results.append(ExpectationResult(
                expectation_type="expect_column_values_to_be_unique",
                column=col,
                success=n_unique == n_total,
                observed_value={"unique": n_unique, "total": n_total, "dupes": n_total - n_unique},
                expected={"unique": n_total},
            ))
        return results

    def validate_range(
        self,
        df: pl.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> ExpectationResult:
        """Check that numeric column values fall within [min_val, max_val]."""
        if column not in df.columns:
            return ExpectationResult("expect_column_values_to_be_between", column, False, None, None)
        actual_min = df[column].min()
        actual_max = df[column].max()
        ok = True
        if min_val is not None and actual_min < min_val:
            ok = False
        if max_val is not None and actual_max > max_val:
            ok = False
        return ExpectationResult(
            expectation_type="expect_column_values_to_be_between",
            column=column, success=ok,
            observed_value={"min": actual_min, "max": actual_max},
            expected={"min": min_val, "max": max_val},
        )

    def validate_row_count(
        self, df: pl.DataFrame, min_rows: int = 1, max_rows: Optional[int] = None
    ) -> ExpectationResult:
        n = len(df)
        ok = n >= min_rows and (max_rows is None or n <= max_rows)
        return ExpectationResult(
            "expect_table_row_count_to_be_between", None, ok,
            observed_value=n, expected={"min": min_rows, "max": max_rows}
        )

    def run_suite(
        self,
        df: pl.DataFrame,
        suite: Dict[str, Any],
        raise_on_failure: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a full expectation suite against a DataFrame.
        suite format:
        {
          "not_null": ["col1", "col2"],
          "unique": ["id"],
          "schema": {"id": "Int64", "name": "Utf8"},
          "ranges": [{"col": "amount", "min": 0, "max": 1e6}],
          "min_rows": 100,
        }
        """
        all_results = []

        if "not_null" in suite:
            all_results.extend(self.validate_not_null(df, suite["not_null"]))
        if "unique" in suite:
            all_results.extend(self.validate_unique(df, suite["unique"]))
        if "schema" in suite:
            all_results.extend(self.validate_schema(df, suite["schema"]))
        if "ranges" in suite:
            for r in suite["ranges"]:
                all_results.append(self.validate_range(df, r["col"], r.get("min"), r.get("max")))
        if "min_rows" in suite:
            all_results.append(self.validate_row_count(df, min_rows=suite["min_rows"]))

        n_pass = sum(1 for r in all_results if r.success)
        n_fail = len(all_results) - n_pass
        failed = [r for r in all_results if not r.success]

        print(f"[GX] Validation: {n_pass} passed, {n_fail} failed")
        for f in failed:
            print(f"  FAIL: {f.expectation_type} on {f.column}: got {f.observed_value}, expected {f.expected}")

        summary = {"passed": n_pass, "failed": n_fail, "results": all_results, "success": n_fail == 0}

        if raise_on_failure and n_fail > 0:
            raise ValueError(f"Data quality validation failed: {n_fail} expectations failed")

        return summary
