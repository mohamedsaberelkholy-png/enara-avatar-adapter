"""
Enara AI <-> Tavus Adapter
"""

import json
import time
import uuid
import httpx
import os
import asyncio
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Enara Avatar Adapter", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://enara-platform-production-12bb.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

security = HTTPBearer()

ENARA_BASE_URL    = os.environ["ENARA_BASE_URL"]
ENARA_API_KEY     = os.environ["ENARA_API_KEY"]
ADAPTER_TOKEN     = os.environ["ADAPTER_TOKEN"]
TAVUS_API_KEY     = os.environ["TAVUS_API_KEY"]
TAVUS_REPLICA_ID  = os.environ["TAVUS_REPLICA_ID"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TAVUS_PAL_ID = os.environ.get("TAVUS_PAL_ID", "p679b746586b")

# In-memory store for visual artifacts
artifact_store: dict = {}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != ADAPTER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "enara-tutor"
    messages: list[ChatMessage]
    stream: Optional[bool] = True
    course_id: Optional[str] = None
    section_ids: Optional[list[str]] = None
    teaching_method: Optional[str] = "socratic"


def extract_enara_context(messages: list[ChatMessage]) -> dict:
    ctx = {}
    for msg in messages:
        if msg.role == "system":
            try:
                first_line = msg.content.strip().split("\n")[0]
                ctx = json.loads(first_line)
            except (json.JSONDecodeError, IndexError):
                pass

            session_val = None
            for line in reversed(msg.content.splitlines()):
                if line.strip().startswith("Session:"):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        session_val = val
                        break
            if session_val:
                ctx["conversation_id"] = session_val
    return ctx


def build_chat_history(messages: list[ChatMessage]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})
    return history


async def generate_visual(question: str, answer: str, session_key: str):
    """Ask Claude Haiku if a visual is needed and generate it if so."""
    if not ANTHROPIC_API_KEY:
        print(f"[VISUAL] ANTHROPIC_API_KEY not set, skipping", flush=True)
        return

    prompt = f"""You are a visual aid generator for an AI tutor.

Student question: {question}
Tutor answer: {answer}

Decide if a visual aid would genuinely help understanding (grammar tables, verb conjugations, 
comparisons, timelines, vocabulary lists, step-by-step processes). 
Simple conversational exchanges do NOT need visuals.

If a visual would help: respond with ONLY a clean, self-contained HTML snippet using inline styles.
No markdown fences, no explanation, just raw HTML starting with <div or <table.
Use a white background, clean fonts, teal (#0A5F6D) as accent color, max-width 100%.
If no visual is needed: respond with exactly: NO_VISUAL"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5",  # FIXED: Updated model string
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["content"][0]["text"].strip()

            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else ""
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0].strip()

            if result != "NO_VISUAL" and "<" in result:
                artifact_store[session_key] = {
                    "html": result,
                    "expires_at": time.time() + 120
                }
                print(f"[VISUAL] ✅ Stored visual for session_key={session_key}", flush=True)
    except Exception as e:
        print(f"[VISUAL] ❌ Error for session_key={session_key}: {e}", flush=True)


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars > len(text) * 0.15:
        return "arabic"
    
    romanized_arabic = [
        "marhaba", "assalamu", "salaam", "wa alaikum", "walaikum",
        "sabah", "masaa", "allo", "kayf", "ahlak", "ahlan",
        "ayakur", "mamkint", "mafi", "yalla", "habibi", "habibti",
        "inshallah", "wallah", "khalas", "tayeb", "tamam", "kwayyes",
        "enta", "enti", "entom", "mesh", "leish", "shu", "shno",
        "meen", "wein", "wayn", "kifak", "tammam", "zain",
    ]
    
    lower = text.lower()
    if any(word in lower for word in romanized_arabic):
        return "arabic"
    
    return "english"


def extract_user_text(content: str) -> str:
    if "<user_audio_analysis>" in content:
        parts = content.split("</user_audio_analysis>")
        if len(parts) > 1:
            return parts[1].strip()
        return re.sub(r"<[^>]+>.*?(?:</[^>]+>|$)", "", content, flags=re.DOTALL).strip()
    return content.strip()


def sse_chunk(content: str, model: str, finish: bool = False) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if not finish else {},
            "finish_reason": "stop" if finish else None,
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def get_or_create_pal(client: httpx.AsyncClient) -> str:
    global TAVUS_PAL_ID

    if TAVUS_PAL_ID:
        return TAVUS_PAL_ID

    resp = await client.post(
        "[https://tavusapi.com/v2/pals](https://tavusapi.com/v2/pals)",
        headers={
            "x-api-key": TAVUS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "pal_name": "Enara AI Tutor",
            "system_prompt": (
                "You are Enara, an AI tutor helping students master their course material. "
                "Guide students using the Socratic method — ask questions rather than just giving answers. "
                "Keep responses short since they will be spoken aloud. "
                "Respond in the same language the student uses, Arabic or English.\n"
                "Session: {conversation_id}"
            ),
            "pipeline_mode": "full",
            "default_face_id": TAVUS_REPLICA_ID,
            "layers": {
                "llm": {
                    "model": "enara-tutor",
                    "base_url": "[https://enara-avatar-adapter-production.up.railway.app/v1](https://enara-avatar-adapter-production.up.railway.app/v1)",
                    "api_key": ADAPTER_TOKEN,
                }
            }
        }
    )
    resp.raise_for_status()
    data = resp.json()
    TAVUS_PAL_ID = data["pal_id"]
    return TAVUS_PAL_ID


@app.get("/v1/artifact/{session_key:path}")
async def get_artifact(session_key: str):
    now = time.time()
    expired = [k for k, v in artifact_store.items() if v["expires_at"] < now]
    for k in expired:
        del artifact_store[k]

    artifact = artifact_store.get(session_key)
    if artifact:
        del artifact_store[session_key]
        return {"html": artifact["html"], "session_key": session_key}
    return {"html": None, "session_key": session_key}


@app.get("/health")
async def health():
    return {"status": "healthy", "pal_id": TAVUS_PAL_ID or "not yet created"}


@app.get("/v1/tavus/credits")
async def get_credits(_token: str = Depends(verify_token)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "[https://tavusapi.com/v2/credits](https://tavusapi.com/v2/credits)",
                headers={"x-api-key": TAVUS_API_KEY}
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.text}")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _token: str = Depends(verify_token)
):
    messages = [
        m for m in request.messages
        if not (m.role == "system" and m.content.strip().lower() == "respond in english")
    ]
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    raw_user_query = next((m.content for m in reversed(messages) if m.role == "user"), None)
    if not raw_user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    user_query = extract_user_text(raw_user_query) or raw_user_query

    ctx = extract_enara_context(messages)
    course_id       = request.course_id       or ctx.get("course_id", "336627af-732e-4349-bda8-b73c702dcf42")
    section_ids     = request.section_ids     or ctx.get("section_ids", [])
    teaching_method = request.teaching_method or ctx.get("teaching_method", "socratic")

    language = detect_language(user_query)
    conversation_id = ctx.get("conversation_id")
    session_key = conversation_id if conversation_id else str(abs(hash(tuple(m.content for m in messages))))[-8:]

    enara_payload = {
        "course_id":       course_id,
        "query":           user_query,
        "section_ids":     section_ids,
        "teaching_method": teaching_method,
        "chat_history":    build_chat_history(messages),
        "language":        language,
    }

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{ENARA_BASE_URL}/chat/query",
                    headers={
                        "X-API-Key":    ENARA_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json=enara_payload,
                )
                resp.raise_for_status()
                data = resp.json()

            except httpx.TimeoutException:
                yield sse_chunk("Sorry, the tutoring service timed out. Please try again.", request.model)
                yield sse_chunk("", request.model, finish=True)
                yield "data: [DONE]\n\n"
                return
            except httpx.HTTPStatusError as e:
                yield sse_chunk(f"Backend error ({e.response.status_code}). Please try again.", request.model)
                yield sse_chunk("", request.model, finish=True)
                yield "data: [DONE]\n\n"
                return

            answer = data.get("answer", "")
            
            # Fire-and-forget background task for visual generation
            asyncio.create_task(generate_visual(user_query, answer, session_key))

            # Stream words with minimal inter-token latency
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield sse_chunk(chunk, request.model)
                await asyncio.sleep(0.01)  # Cooperatively yield execution

            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.delete("/v1/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(
    conversation_id: str,
    _token: str = Depends(verify_token)
):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.delete(
                f"[https://tavusapi.com/v2/conversations/](https://tavusapi.com/v2/conversations/){conversation_id}",
                headers={"x-api-key": TAVUS_API_KEY}
            )
            resp.raise_for_status()
            return {"ended": True, "conversation_id": conversation_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.text}")


@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(
    _token: str = Depends(verify_token)
):
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            pal_id = await get_or_create_pal(client)

            async def prewarm():
                try:
                    await client.get(
                        f"{ENARA_BASE_URL}/health",
                        headers={"X-API-Key": ENARA_API_KEY},
                        timeout=5.0
                    )
                except Exception:
                    pass
            asyncio.create_task(prewarm())

            actual_id = uuid.uuid4().hex
            payload = {
                "persona_id": pal_id,
                "replica_id": TAVUS_REPLICA_ID,
                "conversation_name": f"Enara Tutor - {actual_id[:8]}",
                "conversational_context": f"Session: {actual_id}"
            }

            resp = await client.post(
                "[https://tavusapi.com/v2/conversations](https://tavusapi.com/v2/conversations)",
                headers={
                    "x-api-key": TAVUS_API_KEY,
                    "Content-Type": "application/json"
                },
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "conversation_url": data["conversation_url"],
                "conversation_id": actual_id
            }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.text}")
