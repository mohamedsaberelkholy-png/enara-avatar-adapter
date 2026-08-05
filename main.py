import os
import uuid
import httpx
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Enara Avatar Adapter", version="1.0.0")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables & Defaults
ADAPTER_BEARER_TOKEN = os.getenv("ADAPTER_TOKEN", "EnaraAvatar2026!")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY", "9813e2f240354329ae6d72f8d15170f9")
TAVUS_PAL_ID = os.getenv("TAVUS_PAL_ID") or os.getenv("TAVUS_PERSONA_ID")
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID") or os.getenv("TAVUS_FACE_ID")

# In-memory storage for generated visual artifacts during live sessions
ARTIFACT_STORE = {}

# Security Helper
async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    token_parts = authorization.split()
    if len(token_parts) != 2 or token_parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid token format")
    if token_parts[1] != ADAPTER_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Data Models
class ChatMessage(BaseModel):
    role: str
    content: str

class TavusChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
async def root():
    return {"status": "ok", "service": "Enara Avatar Adapter Production"}

# ---------------------------------------------------------------------------
# 1. Create Tavus Conversation Session Endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(authenticated: bool = Depends(verify_token)):
    """
    Spawns a new Tavus Conversational AI video session.
    Cleanly passes persona_id / replica_id without schema alias duplication errors.
    """
    conversation_id = f"enara_sess_{uuid.uuid4().hex[:12]}"
    tavus_url = "https://tavusapi.com/v2/conversations"
    
    headers = {
        "x-api-key": TAVUS_API_KEY,
        "Content-Type": "application/json"
    }

    # Base payload
    payload = {
        "conversational_context": f"Session: {conversation_id}\nRole: Enara AI Tutor",
        "custom_greeting": "Hello! I am your Enara AI tutor. What would you like to focus on today?"
    }

    # Pass persona_id exclusively (no pal_id duplicate)
    if TAVUS_PAL_ID:
        payload["persona_id"] = TAVUS_PAL_ID
    
    # Pass replica_id exclusively (no face_id duplicate)
    if TAVUS_REPLICA_ID:
        payload["replica_id"] = TAVUS_REPLICA_ID

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(tavus_url, headers=headers, json=payload)
            
            if resp.status_code not in (200, 201):
                print(f"[TAVUS ERROR] Status: {resp.status_code}, Response: {resp.text}")
                raise HTTPException(
                    status_code=resp.status_code, 
                    detail=f"Tavus API Error: {resp.text}"
                )

            data = resp.json()
            
            return {
                "conversation_id": data.get("conversation_id", conversation_id),
                "conversation_url": data.get("conversation_url"),
                "status": "active"
            }

        except httpx.RequestError as exc:
            print(f"[HTTPX ERROR] Request failed: {exc}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to communicate with Tavus API: {str(exc)}"
            )

# ---------------------------------------------------------------------------
# 2. End / Delete Tavus Conversation Session
# ---------------------------------------------------------------------------
@app.delete("/v1/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(conversation_id: str, authenticated: bool = Depends(verify_token)):
    tavus_url = f"https://tavusapi.com/v2/conversations/{conversation_id}"
    headers = {"x-api-key": TAVUS_API_KEY}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            await client.delete(tavus_url, headers=headers)
            ARTIFACT_STORE.pop(conversation_id, None)
            return {"status": "ended", "conversation_id": conversation_id}
        except Exception as e:
            return {"status": "ended", "note": str(e)}

# ---------------------------------------------------------------------------
# 3. Artifact Storage / Polling Endpoints (For Visual Aids)
# ---------------------------------------------------------------------------
@app.get("/v1/artifact/{session_key}")
async def get_artifact(session_key: str):
    """
    Polled by the LiveAvatarModal frontend to pull visual aids dynamically.
    """
    html_content = ARTIFACT_STORE.get(session_key)
    if html_content:
        return {"html": html_content}
    return {"html": None}

@app.post("/v1/artifact/{session_key}")
async def set_artifact(session_key: str, request: Request, authenticated: bool = Depends(verify_token)):
    """
    Called when Claude generates a new HTML visual artifact for the student.
    """
    body = await request.json()
    html_content = body.get("html")
    if html_content:
        ARTIFACT_STORE[session_key] = html_content
        return {"status": "stored", "session_key": session_key}
    raise HTTPException(status_code=400, detail="Missing HTML payload")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
