import streamlit as st
import pandas as pd
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --- DB Setup and Loader Functions ---

def db_setup_and_loader():
    # Example: Load CSV and insert into MongoDB
    # Replace with your actual setup/loader logic
    client = MongoClient("mongodb://localhost:27017/")
    db = client["credit_db"]
    collection = db["transactions"]
    # For demo: load from CSV (replace with your path)
    df = pd.read_csv("credit_card_transactions.csv")
    collection.delete_many({})  # Clear old data
    collection.insert_many(df.to_dict("records"))
    return "Database setup and loaded with data!"

@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("ReducedFeatureFraudDetection").getOrCreate()

@st.cache_data
def load_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["credit_db"]
    collection = db["transactions"]
    data = list(collection.find())
    for record in data:
        record["_id"] = str(record["_id"])
    df = pd.DataFrame(data)
    df.drop(columns=["_id"], inplace=True)
    return df

def train_full_model(spark, df, feature_cols):
    spark_df = spark.createDataFrame(df)
    spark_df = spark_df.withColumnRenamed("default payment next month", "label")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    spark_df = assembler.transform(spark_df).select("features", "label")
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
    model = gbt.fit(train_df)
    return model, assembler, train_df, test_df

def get_top_features(model, feature_cols, top_n=6):
    importances = model.featureImportances.toArray()
    feature_importance = list(zip(feature_cols, importances))
    sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    reduced_features = [feat[0] for feat in sorted_features[:top_n]]
    return reduced_features, sorted_features[:top_n]

def train_reduced_model(spark, df, reduced_features):
    spark_df = spark.createDataFrame(df)
    spark_df = spark_df.withColumnRenamed("default payment next month", "label")
    assembler = VectorAssembler(inputCols=reduced_features, outputCol="features")
    spark_df = assembler.transform(spark_df).select("features", "label")
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
    model = gbt.fit(train_df)
    return model, assembler, train_df, test_df

def evaluate_model(model, test_df):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(model.transform(test_df))
    return roc_auc

def main():
    st.title("Fraud Detection Dashboard (Spark + Streamlit)")

    # --- Setup Button and Session State ---
    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False

    st.write("### Step 1: Database Setup")
    if not st.session_state.db_ready:
        if st.button("Run DB Setup and Loader"):
            msg = db_setup_and_loader()
            st.session_state.db_ready = True
            st.success(msg)
            st.experimental_rerun()
        else:
            st.info("Click the button above to set up and load the database before proceeding.")
            st.stop()
    else:
        st.success("Database is ready! You can now proceed.")

    # --- Main Dashboard (only after DB setup) ---
    feature_cols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    df = load_data()
    st.write("### Data Sample")
    st.dataframe(df.head())

    spark = get_spark()

    st.write("### Training Model on Full Features...")
    full_model, full_assembler, train_df, test_df = train_full_model(spark, df, feature_cols)
    reduced_features, top_features = get_top_features(full_model, feature_cols, top_n=6)

    st.write("### Top Selected Features")
    st.table(pd.DataFrame(top_features, columns=["Feature", "Importance"]))

    st.write("### Retraining Model on Top Features...")
    reduced_model, reduced_assembler, train_df, test_df = train_reduced_model(spark, df, reduced_features)
    roc_auc = evaluate_model(reduced_model, test_df)
    st.write(f"### Reduced-feature Model ROC-AUC Score: {roc_auc:.4f}")

    st.write("## Make a Prediction")
    st.write(f"Enter values for: {', '.join(reduced_features)}")

    user_input = {}
    for feat in reduced_features:
        val = st.number_input(f"{feat}", value=float(df[feat].mean()))
        user_input[feat] = val

    if st.button("Predict"):
        user_df = spark.createDataFrame([user_input])
        user_df = reduced_assembler.transform(user_df).select("features")
        prediction = reduced_model.transform(user_df).select("prediction").collect()[0]["prediction"]
        result = "Defaulter" if prediction == 1 else "Not a Defaulter"
        st.success(f"Prediction: The transaction is predicted to be → **{result}**")

if __name__ == "__main__":
    main()