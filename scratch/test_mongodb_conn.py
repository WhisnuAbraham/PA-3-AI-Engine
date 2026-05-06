import pymongo
import os

# From .env in webkonselor
uri = "mongodb+srv://admin:monitoring2026@cluster0.jc0f5ag.mongodb.net/monitoring?retryWrites=true&w=majority&appName=Cluster0&authSource=admin"

try:
    client = pymongo.MongoClient(uri)
    # The ping command is cheap and does not require auth.
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    
    # List databases
    db_list = client.list_database_names()
    print("Databases:", db_list)
    
    # Check for monitoring2026
    if "monitoring2026" in db_list:
        print("Database 'monitoring2026' exists.")
        db = client["monitoring2026"]
        print("Collections in 'monitoring2026':", db.list_collection_names())
    else:
        print("Database 'monitoring2026' NOT found in list. Checking 'monitoring' instead.")
        db = client["monitoring"]
        print("Collections in 'monitoring':", db.list_collection_names())

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
