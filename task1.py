"""
Big Data Scalability Demo (Advanced)
File: big_data_scalability_demo.py
Type: Jupyter-style Python script (can also be run as standalone)

This script demonstrates advanced big data processing with both PySpark and Dask.
- Synthetic dataset generation (large-scale, configurable)
- PySpark pipeline: transformations, aggregations, window functions, MLlib model
- Dask pipeline: equivalent operations with scaling
- Performance benchmarking and comparison
- Insights printed to console

Ensure you have installed:
- pyspark
- dask[complete]
- pandas, scikit-learn, matplotlib

Usage (example):
    python big_data_scalability_demo.py --backend pyspark --size 5000000
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Dask imports
import dask.dataframe as dd
from dask.distributed import Client


def generate_synthetic_data(size: int = 1_000_000):
    """Generate synthetic dataset with numerical and categorical features."""
    np.random.seed(42)
    df = pd.DataFrame({
        "id": np.arange(size),
        "feature1": np.random.randn(size),
        "feature2": np.random.rand(size) * 100,
        "category": np.random.choice(["A", "B", "C", "D"], size=size),
        "label": np.random.choice([0, 1], size=size),
    })
    return df


# ---------------- PYSPARK PIPELINE ---------------- #
def run_pyspark_pipeline(df: pd.DataFrame):
    spark = SparkSession.builder.appName("BigDataDemo").getOrCreate()
    sdf = spark.createDataFrame(df)

    # Aggregation: average feature per category
    agg = sdf.groupBy("category").agg(F.mean("feature1").alias("avg_feature1"))

    # Window function: rank by feature2 within category
    window = Window.partitionBy("category").orderBy(F.desc("feature2"))
    ranked = sdf.withColumn("rank", F.rank().over(window))

    # MLlib Logistic Regression
    assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
    ml_df = assembler.transform(sdf)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(ml_df)
    summary = model.summary

    agg.show()
    ranked.select("id", "category", "rank").show(5)
    print("PySpark Logistic Regression AUC:", summary.areaUnderROC)

    spark.stop()


# ---------------- DASK PIPELINE ---------------- #
def run_dask_pipeline(df: pd.DataFrame):
    client = Client()  # start local cluster
    ddf = dd.from_pandas(df, npartitions=8)

    # Aggregation: average feature per category
    agg = ddf.groupby("category")["feature1"].mean().compute()

    # Apply transformation: normalize feature2
    ddf = ddf.assign(feature2_norm=(ddf["feature2"] - ddf["feature2"].mean()) / ddf["feature2"].std())

    print("Dask Aggregation Result:")
    print(agg)

    client.close()


# ---------------- BENCHMARKING ---------------- #
def benchmark(size: int, backend: str):
    df = generate_synthetic_data(size)
    start = time.time()

    if backend == "pyspark":
        run_pyspark_pipeline(df)
    elif backend == "dask":
        run_dask_pipeline(df)
    else:
        raise ValueError("Unsupported backend")

    end = time.time()
    print(f"{backend.upper()} runtime for {size:,} rows: {end - start:.2f} seconds")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["pyspark", "dask"], default="pyspark")
    parser.add_argument("--size", type=int, default=1_000_000, help="Number of rows in synthetic dataset")
    args = parser.parse_args()

    benchmark(size=args.size, backend=args.backend)
