from __future__ import annotations

from typing import Optional, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

ALLOWED_ACTIVITY_UNITS = ("pM", "nM", "uM", "mM", "M")
TOP_ACTIVITY_TYPES = ("IC50", "Ki", "Kd", "EC50")


def cast_chembl_tables(
    activities: DataFrame,
    assays: DataFrame,
    targets: DataFrame,
    structures: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Cast key ChEMBL columns to friendlier Spark dtypes.

    This mirrors the cleanup you already started in EDA.
    """
    activities_c = (
        activities.select(
            F.col("activity_id").cast("long").alias("activity_id"),
            F.col("assay_id").cast("long").alias("assay_id"),
            F.col("molregno").cast("long").alias("molregno"),
            F.col("standard_type"),
            F.col("standard_relation"),
            F.col("standard_units"),
            F.col("standard_value").cast("double").alias("standard_value"),
            F.col("pchembl_value").cast("double").alias("pchembl_value"),
        )
    )

    assays_c = (
        assays.select(
            F.col("assay_id").cast("long").alias("assay_id"),
            F.col("target_id").cast("long").alias("target_id"),
            F.col("assay_type"),
            F.col("confidence_score").cast("double").alias("confidence_score"),
        )
    )

    targets_c = (
        targets.select(
            F.col("target_id").cast("long").alias("target_id"),
            F.col("target_chembl_id"),
            F.col("pref_name"),
            F.col("target_type"),
            F.col("organism"),
        )
    )

    structures_c = (
        structures.select(
            F.col("molregno").cast("long").alias("molregno"),
            F.col("canonical_smiles"),
            F.col("standard_inchi_key"),
        )
    )

    return activities_c, assays_c, targets_c, structures_c


def build_base_table(
    activities_c: DataFrame,
    assays_c: DataFrame,
    targets_c: DataFrame,
    structures_c: DataFrame,
) -> DataFrame:
    """Join the cleaned ChEMBL tables into one working dataframe."""
    return (
        activities_c.join(assays_c, on="assay_id", how="left")
        .join(targets_c, on="target_id", how="left")
        .join(structures_c, on="molregno", how="left")
    )


def filter_activity_rows(
    base: DataFrame,
    *,
    standard_type: str = "IC50",
    relation: str = "=",
    allowed_units: Sequence[str] = ALLOWED_ACTIVITY_UNITS,
    min_confidence_score: Optional[float] = 8.0,
    organism: Optional[str] = None,
    target_id: Optional[int] = None,
    require_smiles: bool = True,
) -> DataFrame:
    """
    Keep only rows appropriate for single-task regression.

    Recommended default path for your assignment:
    - one activity type, e.g. IC50
    - exact relations only
    - common units only
    - confidence_score >= 8
    - optional single organism / single target
    """
    df = base

    if require_smiles:
        df = df.filter(F.col("canonical_smiles").isNotNull())

    df = (
        df.filter(F.col("standard_type") == standard_type)
        .filter(F.col("standard_relation") == relation)
        .filter(F.col("standard_value").isNotNull())
        .filter(F.col("standard_units").isin(*allowed_units))
    )

    if min_confidence_score is not None:
        df = (
            df.filter(F.col("confidence_score").isNotNull())
            .filter(F.col("confidence_score") >= float(min_confidence_score))
        )

    if organism is not None:
        df = df.filter(F.col("organism") == organism)

    if target_id is not None:
        df = df.filter(F.col("target_id") == int(target_id))

    return df


def summarize_target_candidates(
    df: DataFrame,
    *,
    top_n: int = 30,
) -> DataFrame:
    """Return the largest targets after filtering so you can choose one."""
    return (
        df.groupBy(
            "target_id",
            "target_chembl_id",
            "pref_name",
            "target_type",
            "organism",
        )
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
    )


def add_ic50_nm_and_pic50(
    df: DataFrame,
    *,
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
    ic50_nm_col: str = "ic50_nM",
    target_col: str = "y",
) -> DataFrame:
    """
    Convert IC50 values to nM and compute pIC50.

    pIC50 = 9 - log10(IC50 in nM)
    """
    df2 = (
        df.withColumn(
            ic50_nm_col,
            F.when(F.col(unit_col) == "pM", F.col(value_col) / F.lit(1e3))
            .when(F.col(unit_col) == "nM", F.col(value_col))
            .when(F.col(unit_col) == "uM", F.col(value_col) * F.lit(1e3))
            .when(F.col(unit_col) == "mM", F.col(value_col) * F.lit(1e6))
            .when(F.col(unit_col) == "M", F.col(value_col) * F.lit(1e9)),
        )
        .filter(F.col(ic50_nm_col).isNotNull())
        .filter(F.col(ic50_nm_col) > 0)
    )

    return df2.withColumn(target_col, F.lit(9.0) - F.log10(F.col(ic50_nm_col)))


def aggregate_measurements_to_molecules(
    df: DataFrame,
    *,
    target_col: str = "y",
    use_median: bool = True,
    keep_measurement_stats: bool = True,
) -> DataFrame:
    """
    Collapse repeated measurements to one label per molecule.

    Grouping is done at the molecule level to reduce leakage between splits.
    """
    label_expr = (
        F.expr(f"percentile_approx({target_col}, 0.5)")
        if use_median
        else F.avg(F.col(target_col))
    )

    aggs = [label_expr.alias(target_col)]
    if "ic50_nM" in df.columns and keep_measurement_stats:
        aggs.append(F.expr("percentile_approx(ic50_nM, 0.5)").alias("ic50_nM_median"))
    if keep_measurement_stats:
        aggs.append(F.count("*").alias("n_measurements"))

    return df.groupBy(
        "molregno",
        "canonical_smiles",
        "standard_inchi_key",
        "target_id",
        "target_chembl_id",
        "pref_name",
    ).agg(*aggs)


def build_single_target_regression_dataset(
    activities_c: DataFrame,
    assays_c: DataFrame,
    targets_c: DataFrame,
    structures_c: DataFrame,
    *,
    target_id: int,
    standard_type: str = "IC50",
    relation: str = "=",
    min_confidence_score: Optional[float] = 8.0,
    organism: Optional[str] = None,
    aggregate_with_median: bool = True,
) -> DataFrame:
    """
    End-to-end helper for the most common path in your assignment.

    Output columns include:
    - canonical_smiles
    - y  (default: pIC50)
    - n_measurements
    and metadata columns useful for debugging/reporting.
    """
    base = build_base_table(activities_c, assays_c, targets_c, structures_c)
    filtered = filter_activity_rows(
        base,
        standard_type=standard_type,
        relation=relation,
        min_confidence_score=min_confidence_score,
        organism=organism,
        target_id=target_id,
    )

    if standard_type == "IC50":
        with_target = add_ic50_nm_and_pic50(filtered, target_col="y")
    else:
        with_target = filtered.withColumn("y", F.col("standard_value"))

    return aggregate_measurements_to_molecules(
        with_target,
        target_col="y",
        use_median=aggregate_with_median,
        keep_measurement_stats=True,
    )


def quick_modeling_report(df: DataFrame, *, target_col: str = "y") -> dict:
    """Small text-friendly summary for the final modeling table."""
    row_count = df.count()
    uniq_mols = df.select("molregno").distinct().count()
    uniq_smiles = df.select("canonical_smiles").distinct().count()

    stats_row = df.select(target_col).describe().toPandas().set_index("summary")

    return {
        "rows": row_count,
        "unique_molecules": uniq_mols,
        "unique_smiles": uniq_smiles,
        "target_summary": stats_row[target_col].to_dict() if target_col in stats_row.columns else {},
    }
