from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from datetime import datetime

import httpx
from motor.motor_asyncio import AsyncIOMotorClient

# ---------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------
ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

if not ATLAS_CONNECTION_STRING:
    raise RuntimeError("ATLAS_CONNECTION_STRING must be set.")

if not RAG_SERVICE_URL:
    raise RuntimeError("RAG_SERVICE_URL must be set.")

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="Gateway Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# MongoDB Client
# ---------------------------------------------------------
mongo_client = AsyncIOMotorClient(ATLAS_CONNECTION_STRING)
db = mongo_client.realtime_ai_db
logs_collection = db.logs

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class QueryData(BaseModel):
    query: str

# ---------------------------------------------------------
# Auth
# ---------------------------------------------------------
async def verify_admin(x_api_key: str = Header(None)):
    if not ADMIN_API_KEY:
        raise HTTPException(500, "ADMIN_API_KEY not configured.")
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(401, "Unauthorized")
    return True

# ---------------------------------------------------------
# Main Ask Endpoint
# ---------------------------------------------------------
@app.post("/ask")
async def ask_question(data: QueryData):
    start_time = datetime.utcnow()

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                RAG_SERVICE_URL,
                json={"query": data.query}
            )

        if response.status_code == 200:
            result = response.json()

            await logs_collection.insert_one({
                "request_id": str(uuid.uuid4()),
                "input_query": data.query,
                "answer": result.get("answer"),
                "source": result.get("source"),
                "score": result.get("score"),
                "request_timestamp": start_time,
                "response_timestamp": datetime.utcnow(),
            })

            return result

        # RAG error
        error_msg = f"RAG service error: {response.text}"
        await logs_collection.insert_one({
            "request_id": str(uuid.uuid4()),
            "input_query": data.query,
            "error": error_msg,
            "request_timestamp": start_time,
            "response_timestamp": datetime.utcnow(),
        })
        return {"error": "RAG service returned an error.", "details": response.text}

    except httpx.TimeoutException:
        await logs_collection.insert_one({
            "request_id": str(uuid.uuid4()),
            "input_query": data.query,
            "error": "Timeout communicating with RAG service.",
            "request_timestamp": start_time,
            "response_timestamp": datetime.utcnow(),
        })
        return {"error": "RAG service timed out."}

    except Exception as e:
        await logs_collection.insert_one({
            "request_id": str(uuid.uuid4()),
            "input_query": data.query,
            "error": str(e),
            "request_timestamp": start_time,
            "response_timestamp": datetime.utcnow(),
        })
        return {"error": "Could not connect to RAG service."}

# ---------------------------------------------------------
# Admin Logs Endpoint
# ---------------------------------------------------------
@app.get("/logs")
async def get_logs(limit: int = 10, authorized: bool = Depends(verify_admin)):
    cursor = logs_collection.find().sort("request_timestamp", -1).limit(limit)
    logs = await cursor.to_list(length=limit)

    for log in logs:
        log["_id"] = str(log["_id"])
        if "request_timestamp" in log:
            log["request_timestamp"] = log["request_timestamp"].isoformat()
        if "response_timestamp" in log:
            log["response_timestamp"] = log["response_timestamp"].isoformat()

    return {"count": len(logs), "logs": logs}
