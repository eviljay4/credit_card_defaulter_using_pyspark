from flask import Flask, request, jsonify
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Initialize Flask
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("MongoDB_PySpark").getOrCreate()

# Load the trained model
model = GBTClassificationModel.load("gbt_model")

# Define feature columns (same as in training)
feature_cols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", 
                "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", 
                "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON input from the user
    df = spark.createDataFrame([data])  # Convert input to PySpark DataFrame
    
    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    
    # Make predictions
    prediction = model.transform(df).select("prediction").collect()[0]["prediction"]
    
    # Return prediction result
    return jsonify({"default_risk": "High" if prediction == 1 else "Low"})

# Run the Flask API
if __name__ == "__main__":
    app.run(debug=True)