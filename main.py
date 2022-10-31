import pyspark
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = "US_Accidents_Dec21_updated.csv"
OUTPUT_PATH = "outputs"


def main():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    accidents_df = spark.read.csv(DATASET_PATH, inferSchema=True, header=True)

    # Show the dataframe schema.
    accidents_df.printSchema()

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


if __name__ == "__main__":
    main()
