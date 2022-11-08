import pyspark
from pyspark.sql.functions import col, isnan, when, count
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATASET_PATH = "US_Accidents_Dec21_updated.csv"
OUTPUT_PATH = "outputs"


def generate_missing_values(accidents_df):
    missing_df = [
        count(
            when(
                col(c).contains("None") | col(c).contains("NULL") | col(c)
                == " " | col(c).isNull() | isnan(c),
                c,
            )
        ).alias(c)
        for c in accidents_df.columns
    ]
    missing_df.show()


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
    plt.show()
    plt.savefig(os.path.join("outputs", "histogram.png"))


def main():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    accidents_df = spark.read.csv(DATASET_PATH, inferSchema=True, header=True)

    # Show the dataframe schema.
    accidents_df.printSchema()

    # generate_histogram(accidents_df)
    generate_missing_values(accidents_df)


if __name__ == "__main__":
    main()
