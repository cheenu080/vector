# gateway_service.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pymongo
from datetime import datetime
import os
import uuid

# Env variables
ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# FastAPI app
app = FastAPI(title="Gateway Service")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client & collection
client = pymongo.MongoClient(ATLAS_CONNECTION_STRING)
db = client.realtime_ai_db
logs_collection = db.logs


class QueryData(BaseModel):
    """Incoming user query."""
    query: str


@app.post("/ask")
async def ask_question(data: QueryData):
    """
    Gateway endpoint:
    - Receives user query
    - Forwards to rag-service
    - Logs Q/A in MongoDB
    """
    start_time = datetime.utcnow()
    rag_service_url = "http://rag-service:8002/ask"

    try:
        response = requests.post(rag_service_url, json={"query": data.query})
        if response.status_code == 200:
            result = response.json()

            log_entry = {
                "request_id": str(uuid.uuid4()),
                "input_query": data.query,
                "answer": result.get("answer"),
                "source": result.get("source"),
                "score": result.get("score"),
                "request_timestamp": start_time,
                "response_timestamp": datetime.utcnow(),
            }
            logs_collection.insert_one(log_entry)

            return result
        else:
            return {
                "error": "RAG service returned an error.",
                "details": response.text,
            }

    except requests.exceptions.RequestException as e:
        log_entry = {
            "request_id": str(uuid.uuid4()),
            "input_query": data.query,
            "error": str(e),
            "request_timestamp": start_time,
            "response_timestamp": datetime.utcnow(),
        }
        logs_collection.insert_one(log_entry)

        return {"error": "Could not connect to RAG service."}


@app.get("/logs")
async def get_logs(limit: int = 10):
    """
    Return recent logs from MongoDB (default: 10 latest).
    """
    logs = list(
        logs_collection.find().sort("request_timestamp", -1).limit(limit)
    )
    # Convert ObjectId + datetime to strings
    for log in logs:
        log["_id"] = str(log["_id"])
        if "request_timestamp" in log:
            log["request_timestamp"] = log["request_timestamp"].isoformat()
        if "response_timestamp" in log:
            log["response_timestamp"] = log["response_timestamp"].isoformat()
    return {"logs": logs}
