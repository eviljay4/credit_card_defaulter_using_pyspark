from pymongo import MongoClient
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ----------------------------
# 1. Initialize Spark session
# ----------------------------
spark = SparkSession.builder.appName("ReducedFeatureFraudDetection").getOrCreate()

# ----------------------------
# 2. Load data from MongoDB
# ----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["credit_db"]
collection = db["transactions"]

data = list(collection.find())
for record in data:
    record["_id"] = str(record["_id"])

df = pd.DataFrame(data)
df.drop(columns=["_id"], inplace=True)

# ----------------------------
# 3. Define full feature columns
# ----------------------------
feature_cols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)
spark_df = spark_df.withColumnRenamed("default payment next month", "label")

# Assemble all features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
spark_df = assembler.transform(spark_df).select("features", "label")

# ----------------------------
# 4. Train model on full features
# ----------------------------
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
model = gbt.fit(train_df)

# ----------------------------
# 5. Get top N important features
# ----------------------------
importances = model.featureImportances.toArray()
feature_importance = list(zip(feature_cols, importances))
sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
top_n = 6
reduced_features = [feat[0] for feat in sorted_features[:top_n]]

print("\nTop Selected Features:")
for name, score in sorted_features[:top_n]:
    print(f"{name}: {score:.6f}")

# ----------------------------
# 6. Retrain model using top features only
# ----------------------------
assembler = VectorAssembler(inputCols=reduced_features, outputCol="features")
spark_df = spark.createDataFrame(df)
spark_df = spark_df.withColumnRenamed("default payment next month", "label")
spark_df = assembler.transform(spark_df).select("features", "label")
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
model = gbt.fit(train_df)

# ----------------------------
# 7. Evaluate model
# ----------------------------
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(model.transform(test_df))
print(f"\nReduced-feature Model ROC-AUC Score: {roc_auc:.4f}")

# ----------------------------
# 8. User input for prediction
# ----------------------------
print("\n--- Enter Transaction Details ---")
print(f"Enter values for: {', '.join(reduced_features)}")
input_values = list(map(float, input("Separate values by space: ").split()))
if len(input_values) != len(reduced_features):
    raise ValueError(f"You must enter exactly {len(reduced_features)} values.")

user_input = dict(zip(reduced_features, input_values))
user_df = spark.createDataFrame([user_input])
user_df = assembler.transform(user_df).select("features")

# ----------------------------
# 9. Make prediction
# ----------------------------
prediction = model.transform(user_df).select("prediction").collect()[0]["prediction"]
result = "Defaulter" if prediction == 1 else "Not a Defaulter"
print(f"\nPrediction: The transaction is predicted to be → {result}")

# ----------------------------
# 10. Stop Spark session
# ----------------------------
spark.stop()
