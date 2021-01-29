import sys
import sys
import tempfile
import requests
from pyspark import StorageLevel
import pyspark.sql.functions as functions
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer

from pyspark.sql import SparkSession


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Nice way to write a tmp file onto the system
    temp_csv_file = tempfile.mktemp()
    with open(temp_csv_file, mode="wb") as f:
        data_https = requests.get(
            "https://teaching.mrsharky.com/data/iris.data"
        )
        f.write(data_https.content)

    iris_df = spark.read.csv(temp_csv_file, inferSchema="true", header="true")
    iris_df = iris_df.toDF(
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class")

    iris_df.createOrReplaceTempView("iris")
    iris_df.persist(StorageLevel.DISK_ONLY)

    # Simple SQL
    results = spark.sql("SELECT * FROM iris")
    results.show()
    # Average for each of the 4
    average_overall = spark.sql(
        """
        SELECT
                AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM iris
        """
    )
    average_overall.show()

    # Average for each of the 4 by class
    average_by_class = spark.sql(
        """
        SELECT
                class
                , AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM iris
            GROUP BY class
        """
    )
    average_by_class.show()

    # Add a new column

    iris_df = iris_df.withColumn("rand", functions.rand(seed=42))
    iris_df.createOrReplaceTempView("iris")
    results = spark.sql("SELECT * FROM iris ORDER BY rand")
    results.show()

    vector_assembler = VectorAssembler(
        inputCols=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"],
        outputCol="vector",
    )

    iris_df = vector_assembler.transform(iris_df)
    iris_df.show()


    # Numberize the class column of iris

    string_indexer = StringIndexer(inputCol="class", outputCol="indexed")
    indexer_fitted = string_indexer.fit(iris_df)
    iris_df = indexer_fitted.transform(iris_df)
    iris_df.createOrReplaceTempView("iris")
    results = spark.sql("SELECT * FROM iris ORDER BY rand")
    results.show()
    return

    # Random Forest
    random_forest_classifier = RandomForestClassifier(
        featuresCol="vector",
        labelCol="indexed"
    )
    random_forest_classifier_fitted = random_forest_classifier.fit(iris_df)
    iris_df = random_forest_classifier_fitted.transform(iris_df)
    iris_df.createOrReplaceTempView("iris")
    results = spark.sql("SELECT * FROM iris ORDER BY rand")
    results.show()

    # Calculate the model's Accuracy
    print_heading("Accuracy")
    iris_df_accuracy = spark.sql(
        """
        SELECT
                SUM(correct)/COUNT(*) AS accuracy
            FROM
                (SELECT
                        CASE WHEN prediction == class_idx THEN 1
                        ELSE 0 END AS correct
                    FROM predicted) AS TMP
              """
    )
    iris_df_accuracy.show()

if __name__ == "__main__":
    sys.exit(main())