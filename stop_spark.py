from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
spark.stop()  # Properly shuts down Spark session