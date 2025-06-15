import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["credit_db"]
collection = db["transactions"]

print("Connected to MongoDB successfully!")