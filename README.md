📊 Credit Card Defaulter Prediction System
A machine learning project that predicts whether a credit card user is likely to default, using a Gradient Boosted Trees model built with PySpark. The system leverages real transaction data stored in MongoDB, extracts the most important features, and enables real-time prediction via command-line user input.

🚀 Features
🔍 Credit Default Prediction using PySpark's Gradient Boosted Tree Classifier

📊 Automatic Feature Selection – selects top 6 most important features to improve model efficiency

🧠 Model Training & Evaluation – trains on MongoDB data and evaluates using ROC-AUC score

👤 Real-Time User Input – accepts transaction data from the user to predict defaulter status

🗄️ MongoDB Integration – reads transaction data directly from a NoSQL database

🛠️ Technologies Used
Programming Language: Python

Big Data Framework: Apache Spark (PySpark)

Database: MongoDB

Libraries:

pyspark.ml for model building

pymongo for MongoDB connection

pandas for initial data manipulation

BinaryClassificationEvaluator for model evaluation

📁 Project Workflow
Connects to MongoDB and loads credit card transaction data.

Trains a GBTClassifier model on all features.

Extracts top 6 most important features based on feature importances.

Retrains the model using only the top 6 features.

Evaluates the model with ROC-AUC metric.

Prompts the user to enter real-time input for prediction.

Displays the prediction result: "Defaulter" or "Not a Defaulter".

📌 Example Input (for user prompt)
When prompted, enter values for the top selected features separated by space, for example:

bash
Copy
Edit
--- Enter Transaction Details ---
Enter values for: LIMIT_BAL, PAY_0, PAY_2, BILL_AMT1, PAY_AMT1, PAY_AMT2
Input: 50000 0 0 20000 1500 2000
🧪 Evaluation
Model is evaluated using ROC-AUC score

Achieves robust performance using reduced features for faster inference

📦 How to Run
Make sure MongoDB is running and contains a collection transactions in the credit_db database.

Install dependencies:


Copy
Edit
pip install pyspark pandas pymongo



Run the script:


Copy
Edit
python model_training.py
