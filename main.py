import pyspark
from pyspark.sql.functions import col, when, count
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

DATASET_PATH = "US_Accidents_Dec21_updated.csv"
OUTPUT_PATH = "outputs"


def generate_jointplot(accidents_df):
    sns.jointplot(
        x=list(accidents_df.select("Start_Lat").toPandas()["Start_Lat"]),
        y=list(accidents_df.select("Start_Lng").toPandas()["Start_Lng"]),
        height=10,
    )
    plt.ylabel("Starting latitude", fontsize=12)
    plt.xlabel("Starting longitude", fontsize=12)
    plt.savefig(os.path.join("outputs", "jointplot.png"))
    plt.show()


def generate_missing_values(accidents_df):
    missing_df = accidents_df.select(
        [
            count(
                when(
                    col(c).contains("None")
                    | col(c).contains("NULL")
                    | (col(c) == " ")
                    | col(c).isNull(),
                    c,
                )
            ).alias(c)
            for c in accidents_df.columns
        ]
    )
    pyspark_df = missing_df.toPandas()

    fig, ax = plt.subplots(figsize=(12, 18))
    ind = np.arange(pyspark_df.shape[1])
    rects = ax.barh(ind, pyspark_df.to_numpy()[0], color="blue")
    ax.set_yticks(ind)
    ax.set_yticklabels(pyspark_df.columns, rotation="horizontal")
    ax.set_xlabel("Number of missing values")
    ax.set_title("Number of missing values per column")
    fig.savefig(os.path.join("outputs", "missing.png"))
    plt.show()


def generate_histogram(accidents_df):
    # Plot the count of accidents by state.
    states = accidents_df.select("State").distinct().collect()
    counts = []

    for row in states:
        counts.append(
            accidents_df.filter(accidents_df.State == row[0]).select("ID").count()
        )

    fig, ax = plt.subplots(figsize=(40, 32))
    sns.barplot(x=[row[0] for row in states], y=counts)
    fig.savefig(os.path.join("outputs", "histogram.png"))
    plt.show()


def generate_correlation_coefficients(accidents_df):
    x_cols = [
        col
        for col, type in accidents_df.dtypes
        if col not in ["Severity"] and type in ("int", "double")
    ]

    labels = []
    values = []
    severity = list(accidents_df.select("Severity").toPandas()["Severity"])
    for col in x_cols:
        labels.append(col)
        values.append(
            np.corrcoef(
                list(accidents_df.select(col).toPandas()[col]),
                severity,
            )[0, 1],
        )
    corr_df = pd.DataFrame({"col_labels": labels, "corr_values": values})
    corr_df = corr_df.sort_values(by="corr_values")

    ind = np.arange(len(labels))
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color="y")
    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation="horizontal")
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    fig.savefig(os.path.join("outputs", "correlation_coefficients.png"))
    plt.show()


def main():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    accidents_df = spark.read.csv(DATASET_PATH, inferSchema=True, header=True)

    # Show the dataframe schema.
    accidents_df.printSchema()

    # generate_histogram(accidents_df)
    # generate_missing_values(accidents_df)
    generate_jointplot(accidents_df)
    # generate_correlation_coefficients(accidents_df)


if __name__ == "__main__":
    main()
