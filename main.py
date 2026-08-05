import os
import uuid
import httpx
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Enara Avatar Adapter", version="1.0.0")

# ---------------------------------------------------------------------------
# 1. Global CORS Configuration (Allows all origins & preflight requests)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(status_code=200, content={"status": "ok"})

# Environment Variables
ADAPTER_BEARER_TOKEN = os.getenv("ADAPTER_TOKEN", "EnaraAvatar2026!")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY", "9813e2f240354329ae6d72f8d15170f9")
TAVUS_PAL_ID = os.getenv("TAVUS_PAL_ID") or os.getenv("TAVUS_PERSONA_ID")
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID") or os.getenv("TAVUS_FACE_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# In-memory artifact storage
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
# 2. Tavus Session Management Endpoints (POST & DELETE)
# ---------------------------------------------------------------------------
@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(authenticated: bool = Depends(verify_token)):
    """
    Spawns a new Tavus Conversational AI video session configured for Arabic.
    """
    conversation_id = f"enara_sess_{uuid.uuid4().hex[:12]}"
    tavus_url = "https://tavusapi.com/v2/conversations"
    
    headers = {
        "x-api-key": TAVUS_API_KEY,
        "Content-Type": "application/json"
    }

    # Enhanced payload for Arabic speech & tutoring context
    payload = {
        "conversational_context": (
            f"Session ID: {conversation_id}\n"
            "Role: Enara AI Tutor (معلم عنارة الذكي)\n"
            "Language Instructions: You must speak and respond exclusively in natural fluent Arabic (اللغة العربية).\n"
            "Tone: Warm, encouraging, concise, and educational."
        ),
        "custom_greeting": "مرحباً بك! أنا معلمك الذكي من منصة عنارة. كيف يمكنني مساعدتك في دراستك اليوم؟"
    }

    if TAVUS_PAL_ID:
        payload["persona_id"] = TAVUS_PAL_ID
    if TAVUS_REPLICA_ID:
        payload["replica_id"] = TAVUS_REPLICA_ID

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(tavus_url, headers=headers, json=payload)
            if resp.status_code not in (200, 201):
                print(f"[TAVUS ERROR] Status {resp.status_code}: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Tavus API Error: {resp.text}")
            
            data = resp.json()
            return {
                "conversation_id": data.get("conversation_id", conversation_id),
                "conversation_url": data.get("conversation_url"),
                "status": "active"
            }
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to communicate with Tavus API: {str(exc)}")

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
# 3. Helper: Call Anthropic Claude safely for HTML Visual Cards
# ---------------------------------------------------------------------------
async def generate_claude_visual_artifact(user_prompt: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        print("[WARNING] ANTHROPIC_API_KEY is missing from Railway environment variables.")
        return None

    system_prompt = """
    You are an expert AI Tutor visual designer for an educational platform called Enara.
    Your job is to generate self-contained, modern, beautiful HTML visual aids to help students learn.
    
    RULES:
    1. Output strictly ONLY raw HTML content inside <div> tags (no ```html code fences, no extra text).
    2. Support Arabic RTL (Right-to-Left) direction if prompt is in Arabic (direction: rtl; text-align: right;).
    3. Use clean inline CSS with modern UI cards, clear headers, icons/emojis, and bullet points.
    4. Make the design modern, responsive, readable, and visually appealing.
    """

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Verified Anthropic API model identifiers with fallbacks
    models_to_try = [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307"
    ]

    endpoint_url = "[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)"

    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in models_to_try:
            payload = {
                "model": model,
                "max_tokens": 1500,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": f"Create an educational visual card for this topic: {user_prompt}"}
                ]
            }
            try:
                resp = await client.post(endpoint_url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    html_code = data["content"][0]["text"].strip()
                    if html_code.startswith("```html"):
                        html_code = html_code.replace("```html", "").replace("```", "").strip()
                    elif html_code.startswith("```"):
                        html_code = html_code.replace("```", "").strip()
                    return html_code
                else:
                    print(f"[CLAUDE ERROR] Model '{model}' failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                print(f"[CLAUDE EXCEPTION] Model '{model}' request failed: {e}")

    return None

# ---------------------------------------------------------------------------
# 4. Artifact & Text Chat Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/artifact/{session_key}")
async def get_artifact(session_key: str):
    html_content = ARTIFACT_STORE.get(session_key)
    return {"html": html_content}

@app.post("/v1/artifact/{session_key}")
async def set_artifact(session_key: str, request: Request, authenticated: bool = Depends(verify_token)):
    body = await request.json()
    html_content = body.get("html")
    if html_content:
        ARTIFACT_STORE[session_key] = html_content
        return {"status": "stored", "session_key": session_key}
    raise HTTPException(status_code=400, detail="Missing HTML payload")

@app.post("/v1/chat/completions")
async def text_chat_completion(
    request: TavusChatRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        user_message = request.messages[-1].content if request.messages else ""
        session_key = f"text_sess_{uuid.uuid4().hex[:8]}"

        html_artifact = await generate_claude_visual_artifact(user_message)

        if html_artifact:
            ARTIFACT_STORE[session_key] = html_artifact
            reply_text = f"لقد قمت بإنشاء لوحة توضيحية لك بناءً على سؤالك: '{user_message}'"
        else:
            reply_text = f"تم استلام طلبك: '{user_message}' (لم يتم توليد لوحة بصرية)"

        return JSONResponse(
            status_code=200,
            content={
                "id": session_key,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": reply_text
                        }
                    }
                ],
                "artifact_key": session_key if session_key in ARTIFACT_STORE else None
            }
        )
    except Exception as err:
        print(f"[CHAT ERROR] Unhandled exception: {err}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(err)}
        )

# ---------------------------------------------------------------------------
# Dynamic Port Binding for Railway
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
