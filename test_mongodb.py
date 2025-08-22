import os
from pymongo import MongoClient
from dotenv import load_dotenv

print("=== Testing MongoDB Connection ===")

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URL from environment variables
mongo_url = os.getenv('MONGO_URL')
print(f"MongoDB URL: {mongo_url}")

if not mongo_url:
    print("ERROR: MONGO_URL not found in environment variables")
    exit(1)

try:
    # Try to connect to MongoDB
    print("\nAttempting to connect to MongoDB...")
    client = MongoClient(
        mongo_url,
        ssl=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000  # 5 second timeout
    )
    
    # Test the connection
    print("Connection successful!")
    print("Server info:")
    print(client.server_info())
    
    # Check if the database exists
    db_name = os.getenv('DB_NAME', 'clinical_ai')
    print(f"\nChecking database '{db_name}'...")
    if db_name in client.list_database_names():
        print(f"Database '{db_name}' exists!")
        db = client[db_name]
        print(f"Collections in {db_name}: {db.list_collection_names()}")
    else:
        print(f"Database '{db_name}' does not exist.")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Make sure your MongoDB Atlas cluster is running")
    print("2. Check if your IP is whitelisted in MongoDB Atlas")
    print("3. Verify your MongoDB connection string in the .env file")
    print("4. Check your internet connection")
    
    # Try to get more specific error information
    if "SSL" in str(e) or "certificate" in str(e).lower():
        print("\nSSL Certificate Error Detected!")
        print("Trying with SSL disabled...")
        try:
            client = MongoClient(
                mongo_url.replace("mongodb+srv", "mongodb"),
                ssl=False,
                serverSelectionTimeoutMS=5000
            )
            print("Successfully connected with SSL disabled!")
            print("Server info:", client.server_info())
        except Exception as e2:
            print(f"Still failed with SSL disabled: {str(e2)}")

print("\nTest completed.")
