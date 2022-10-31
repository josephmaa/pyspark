from pyspark.sql import SparkSession

DATASET_PATH = "US_Accidents_Dec21_updated.csv"


def main():
    spark = SparkSession.builder.getOrCreate()
    accidents_df = spark.read.csv(DATASET_PATH, inferSchema=True, header=True)

    # Show the dataframe schema
    accidents_df.printSchema()


if __name__ == "__main__":
    main()
