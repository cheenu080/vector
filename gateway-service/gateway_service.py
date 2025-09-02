# gateway_service.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel
import pymongo
from datetime import datetime
import os

ATLAS_URI = os.getenv("ATLAS_CONNECTION_STRING")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read the connection string from the environment variable
ATLAS_CONNECTION_STRING = os.environ.get("ATLAS_CONNECTION_STRING")

# Initialize MongoDB client and database
client = pymongo.MongoClient(ATLAS_CONNECTION_STRING)
db = client.realtime_ai_db
logs_collection = db.logs

class QueryData(BaseModel):
    """
    Data model for the incoming user query.
    """
    query: str

@app.post("/ask")
async def ask_question(data: QueryData):
    """
    This endpoint acts as the gateway for the Q&A system.
    It receives a user question, forwards it to the RAG service,
    and logs the interaction to MongoDB.
    """
    start_time = datetime.utcnow()
    # Updated URL to use the Docker Compose service name and new endpoint
    rag_service_url = "http://rag-service:8002/ask"
    
    try:
        # Forward the request to the RAG service with the new data model
        response = requests.post(rag_service_url, json={"query": data.query})
        
        if response.status_code == 200:
            result = response.json()
            
            # Log the successful request to MongoDB
            log_entry = {
                "input_query": data.query,
                "answer": result.get("answer"),
                "context": result.get("context"),
                "request_timestamp": start_time,
                "response_timestamp": datetime.utcnow()
            }
            logs_collection.insert_one(log_entry)
            
            return result
        
        else:
            # Handle cases where the RAG service returns an error
            return {"error": "RAG service returned an error.", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        # Log the error if the RAG service is not reachable
        log_entry = {
            "input_query": data.query,
            "error": str(e),
            "request_timestamp": start_time,
            "response_timestamp": datetime.utcnow()
        }
        logs_collection.insert_one(log_entry)

        return {"error": "Could not connect to RAG service."}
