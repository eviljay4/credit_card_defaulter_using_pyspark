import pandas as pd
from db_setup import collection  # Import MongoDB connection

# Load dataset (Ensure 'your_dataset.csv' is inside 'bda_project')
df = pd.read_csv("default of credit card clients.csv")

# Convert DataFrame to dictionary format
data_dict = df.to_dict(orient="records")

# Insert data into MongoDB
collection.insert_many(data_dict)

print("Data successfully inserted into MongoDB!")