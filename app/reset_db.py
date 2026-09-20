from pymongo import MongoClient
import toml
import certifi

# Load the URI directly from your secrets file
secrets = toml.load(".streamlit/secrets.toml")
client = MongoClient(secrets["MONGO_URI"], tlsCAFile=certifi.where())

# Connect to the database and drop the collection
db = client.healthcare_app
db.predictions.drop()

print("Patient history has been successfully reset!")