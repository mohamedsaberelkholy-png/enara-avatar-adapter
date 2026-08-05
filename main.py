"""
Enara AI <-> Tavus Adapter
"""

import json
import time
import uuid
import httpx
import os
import asyncio
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

ENARA_BASE_URL   = os.environ["ENARA_BASE_URL"]
ENARA_API_KEY    = os.environ["ENARA_API_KEY"]
ADAPTER_TOKEN    = os.environ["ADAPTER_TOKEN"]
TAVUS_API_KEY    = os.environ["TAVUS_API_KEY"]
TAVUS_REPLICA_ID = os.environ["TAVUS_REPLICA_ID"]
TAVUS_PAL_ID     = os.environ.get("TAVUS_PAL_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REDIS_URL = os.environ.get("REDIS_URL", "")

redis_client: aioredis.Redis | None = None


async def warmup_modal():
    """Ping Enara Modal backend every 90 seconds to prevent cold starts."""
    await asyncio.sleep(10)
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{ENARA_BASE_URL}/chat/query",
                    headers={"X-API-Key": ENARA_API_KEY, "Content-Type": "application/json"},
                    json={"course_id": "336627af-732e-4349-bda8-b73c702dcf42", "query": ".", "section_ids": [], "teaching_method": "socratic", "chat_history": [], "language": "english"},
                )
        except Exception:
            pass
        await asyncio.sleep(90)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    if REDIS_URL:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    asyncio.create_task(warmup_modal())
    yield
    if redis_client:
        await redis_client.aclose()


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
    """Extract course context and session key from system messages."""
    for msg in messages:
        if msg.role == "system":
            try:
                first_line = msg.content.strip().split("\n")[0]
                return json.loads(first_line)
            except (json.JSONDecodeError, IndexError):
                pass
    return {}


def extract_session_key(messages: list[ChatMessage]) -> str:
    """Extract session key from Tavus system message.
    Tavus injects: 'Session: <conversation_id>'
    We use the last 8 chars to match what the frontend polls with.
    """
    for msg in messages:
        if msg.role == "system":
            for line in msg.content.split("\n"):
                line = line.strip()
                if line.startswith("Session:"):
                    val = line.replace("Session:", "").strip()
                    if val:
                        return val[-8:]
    # Fallback: hash of messages
    return str(abs(hash(tuple(m.content for m in messages))))[-8:]


def build_chat_history(messages: list[ChatMessage]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})
    return history


def detect_language(text: str) -> str:
    """Detect if the message is Arabic or English."""
    arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "arabic" if arabic_count > len(text) * 0.15 else "english"


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


async def generate_visual(question: str, answer: str, session_key: str):
    """Ask Claude Haiku if a visual is needed and generate it if so."""
    if not ANTHROPIC_API_KEY:
        print("generate_visual: no ANTHROPIC_API_KEY set", flush=True)
        return

    prompt = f"""You are a visual aid generator for an AI tutor.

Student question: {question}
Tutor answer: {answer}

Decide if a visual aid would genuinely help understanding (grammar tables, verb conjugations, 
comparisons, timelines, vocabulary lists, step-by-step processes, email structure diagrams).
Simple conversational exchanges do NOT need visuals.

If a visual would help: respond with ONLY a clean, self-contained HTML snippet using inline styles.
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
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["content"][0]["text"].strip()
            print(f"generate_visual key={session_key}: {result[:80]}", flush=True)

            if result != "NO_VISUAL" and "<" in result:
                if redis_client:
                    await redis_client.setex(f"artifact:{session_key}", 120, result)
                    print(f"Visual stored in Redis key=artifact:{session_key}", flush=True)
    except Exception as e:
        print(f"Visual generation error: {e}", flush=True)


async def get_or_create_pal(client: httpx.AsyncClient) -> str:
    global TAVUS_PAL_ID
    if TAVUS_PAL_ID:
        return TAVUS_PAL_ID

    resp = await client.post(
        "https://tavusapi.com/v2/personas",
        headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
        json={
            "persona_name": "Enara AI Tutor",
            "system_prompt": (
                "You are Enara, an AI tutor helping students master their course material. "
                "Guide students using the Socratic method. "
                "Keep responses short since they will be spoken aloud. "
                "Respond in the same language the student uses, Arabic or English."
            ),
            "pipeline_mode": "full",
            "default_replica_id": TAVUS_REPLICA_ID,
            "layers": {
                "llm": {
                    "model": "enara-tutor",
                    "base_url": "https://enara-avatar-adapter-production.up.railway.app/v1",
                    "api_key": ADAPTER_TOKEN,
                    "speculative_inference": False,
                }
            }
        }
    )
    resp.raise_for_status()
    data = resp.json()
    TAVUS_PAL_ID = data["persona_id"]
    return TAVUS_PAL_ID


@app.get("/v1/artifact/{session_key}")
async def get_artifact(session_key: str):
    """Poll for a visual artifact. Returns html if available, null if not."""
    if redis_client:
        html = await redis_client.get(f"artifact:{session_key}")
        if html:
            await redis_client.delete(f"artifact:{session_key}")
            return {"html": html, "session_key": session_key}
    return {"html": None, "session_key": session_key}


@app.get("/health")
async def health():
    return {"status": "healthy", "pal_id": TAVUS_PAL_ID or "not yet created"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _token: str = Depends(verify_token)
):
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_query = next(
        (m.content for m in reversed(messages) if m.role == "user"),
        None
    )
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    ctx = extract_enara_context(messages)
    course_id       = request.course_id       or ctx.get("course_id", "336627af-732e-4349-bda8-b73c702dcf42")
    section_ids     = request.section_ids     or ctx.get("section_ids", [])
    teaching_method = request.teaching_method or ctx.get("teaching_method", "socratic")
    language        = detect_language(user_query)
    session_key     = extract_session_key(messages)

    print(f"chat_completions: lang={language} session={session_key} query={user_query[:40]}", flush=True)

    enara_payload = {
        "course_id":       course_id,
        "query":           user_query,
        "section_ids":     section_ids,
        "teaching_method": teaching_method,
        "chat_history":    build_chat_history(messages),
        "language":        language,
    }

    async def generate():
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{ENARA_BASE_URL}/chat/query",
                    headers={"X-API-Key": ENARA_API_KEY, "Content-Type": "application/json"},
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
        asyncio.create_task(generate_visual(user_query, answer, session_key))

        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield sse_chunk(chunk, request.model)

        yield sse_chunk("", request.model, finish=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.delete("/v1/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(
    conversation_id: str,
    _token: str = Depends(verify_token)
):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.delete(
                f"https://tavusapi.com/v2/conversations/{conversation_id}",
                headers={"x-api-key": TAVUS_API_KEY}
            )
            resp.raise_for_status()
            return {"ended": True, "conversation_id": conversation_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.status_code} - {e.response.text}")


@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(_token: str = Depends(verify_token)):
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            pal_id = await get_or_create_pal(client)

            async def prewarm():
                try:
                    await client.get(f"{ENARA_BASE_URL}/health", headers={"X-API-Key": ENARA_API_KEY}, timeout=5.0)
                except Exception:
                    pass
            asyncio.create_task(prewarm())

            payload = {
                "persona_id": pal_id,
                "replica_id": TAVUS_REPLICA_ID,
                "conversation_name": f"Enara Tutor - {uuid.uuid4().hex[:8]}",
                "properties": {"language": "multilingual"}
            }
            print(f"DEBUG sending to Tavus: persona_id={pal_id} replica_id={TAVUS_REPLICA_ID} api_key={TAVUS_API_KEY[:8]}", flush=True)

            resp = await client.post(
                "https://tavusapi.com/v2/conversations",
                headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
                json=payload
            )
            print(f"DEBUG Tavus response {resp.status_code}: {resp.text}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            return {
                "conversation_url": data["conversation_url"],
                "conversation_id": data["conversation_id"]
            }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.status_code} - {e.response.text}")
