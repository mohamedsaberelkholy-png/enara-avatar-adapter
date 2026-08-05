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

# Environment Variables & Secrets
ADAPTER_BEARER_TOKEN = os.getenv("ADAPTER_TOKEN", "EnaraAvatar2026!")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY", "9813e2f240354329ae6d72f8d15170f9")
TAVUS_PAL_ID = os.getenv("TAVUS_PAL_ID") or os.getenv("TAVUS_PERSONA_ID")
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID") or os.getenv("TAVUS_FACE_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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
# Helper: Generate HTML Visual Card with Claude
# ---------------------------------------------------------------------------
async def generate_claude_visual_artifact(user_prompt: str) -> Optional[str]:
    """
    Calls Anthropic Claude API to generate self-contained, beautifully styled HTML/CSS
    visual cards (RTL Arabic supported) for the student's learning session.
    """
    if not ANTHROPIC_API_KEY:
        print("[WARNING] ANTHROPIC_API_KEY is not set. Skipping Claude artifact generation.")
        return None

    system_prompt = """
    You are an expert AI Tutor visual designer for an educational platform called Enara.
    Your job is to generate self-contained, modern, beautiful HTML visual aids to help students learn.
    
    RULES:
    1. Output strictly ONLY the raw HTML content inside <div> tags (no ```html code fences, no extra conversational text).
    2. Support Arabic RTL (Right-to-Left) direction if the prompt is in Arabic (direction: rtl; text-align: right;).
    3. Use clean inline CSS with modern UI cards, clear headers, icons/emojis, and bullet points.
    4. Make the design modern, responsive, readable, and visually appealing for high school & college students.
    """

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1500,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Create an interactive/visual educational card for this topic: {user_prompt}"}
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post("[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                html_code = data["content"][0]["text"].strip()
                # Clean up fences if returned
                if html_code.startswith("```html"):
                    html_code = html_code.replace("```html", "").replace("
