from pyspark.sql import SparkSession
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = "US_Accidents_Dec21_updated.csv"
OUTPUT_PATH = "outputs"


def main():
    spark = SparkSession.builder.getOrCreate()
    accidents_df = spark.read.csv(DATASET_PATH, inferSchema=True, header=True)

    # Show the dataframe schema.
    accidents_df.printSchema()

    # Plot the count of accidents by state.
    states = accidents_df.State.unique()
    counts = []
    for i in accidents_df.State.unique():
        counts.append(accidents_df[accidents_df["State"] == i].count()["ID"])

    fig, ax = plt.subplots(figsize=(40, 32))
    sns.barplot(states, counts)


if __name__ == "__main__":
    main()
