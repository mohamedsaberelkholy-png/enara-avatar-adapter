"""
Enara AI <-> Tavus Adapter
"""

import json
import re
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

ENARA_BASE_URL    = os.environ["ENARA_BASE_URL"]
ENARA_API_KEY     = os.environ["ENARA_API_KEY"]
ADAPTER_TOKEN     = os.environ["ADAPTER_TOKEN"]
TAVUS_API_KEY     = os.environ["TAVUS_API_KEY"]
TAVUS_REPLICA_ID  = os.environ["TAVUS_REPLICA_ID"]
TAVUS_PAL_ID      = os.environ.get("TAVUS_PAL_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REDIS_URL         = os.environ.get("REDIS_URL", "")

LANG_TTL = 7200  # 2 hours

redis_client: aioredis.Redis | None = None


async def warmup_modal():
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


async def prewarm_modal():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{ENARA_BASE_URL}/health", headers={"X-API-Key": ENARA_API_KEY})
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    if REDIS_URL:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        print(f"Redis connected: {REDIS_URL[:30]}...", flush=True)
    else:
        print("WARNING: No REDIS_URL set — visuals disabled", flush=True)
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


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

def extract_enara_context(messages: list[ChatMessage]) -> dict:
    for msg in messages:
        if msg.role == "system":
            try:
                first_line = msg.content.strip().split("\n")[0]
                return json.loads(first_line)
            except (json.JSONDecodeError, IndexError):
                pass
    return {}


def extract_session_key(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.role == "system":
            match = re.search(r'\b(c[0-9a-f]{15,})\b', msg.content)
            if match:
                return match.group(1)[-8:]
            for line in msg.content.split("\n"):
                line = line.strip()
                if line.startswith("Session:"):
                    val = line.replace("Session:", "").strip()
                    return val[-8:] if len(val) >= 8 else val
    return str(abs(hash(tuple(m.content for m in messages))))[-8:]


def build_chat_history(messages: list[ChatMessage]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})
    return history


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

FRANCO_STRONG = {
    "ya3ni", "ya3ny", "mesh", "msh", "ezay", "leih", "leh",
    "mafish", "mafesh", "yalla", "khalas", "delwa2ty", "delwaqti",
    "ba3dein", "b3deen", "3alshan", "3shan", "3andi", "3ndi",
    "3ayiz", "aayiz", "mumkin", "momken", "lazim", "laazim",
}

FRANCO_BROAD = FRANCO_STRONG | {
    "ana", "enta", "enti", "fe", "fi", "keda", "kida",
    "zay", "law", "meen", "fein", "fyn", "wala", "walla",
    "aho", "taman", "tamam", "tayeb", "tayyeb", "howa", "hiya",
    "ihna", "shoof", "shof", "7aga", "kol", "kull", "aslan",
    "awy", "gedan", "sa3at", "el", "al", "bas",
}

ARABIC_PHRASES = [
    "in arabic", "explain in arabic", "respond in arabic", "answer in arabic",
    "باللغة العربية", "بالعربي", "بالعربية", "translate to arabic",
    "say it in arabic", "tell me in arabic", "switch to arabic",
]
ENGLISH_PHRASES = [
    "in english", "explain in english", "respond in english", "answer in english",
    "بالانجليزي", "بالإنجليزية", "بالانجليزية", "translate to english",
    "say it in english", "tell me in english", "switch to english",
]


def detect_language_from_text(text: str) -> tuple[str, str]:
    text_lower = text.lower().strip()
    for phrase in ARABIC_PHRASES:
        if phrase in text_lower:
            return "arabic", "explicit_phrase"
    for phrase in ENGLISH_PHRASES:
        if phrase in text_lower:
            return "english", "explicit_phrase"
    total = len(text.replace(" ", ""))
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if total > 0:
        if total <= 10 and arabic_chars > 0:
            return "arabic", "arabic_script_short"
        if arabic_chars / total > 0.10:
            return "arabic", "arabic_script"
    words = set(text_lower.split())
    strong = words & FRANCO_STRONG
    if strong:
        return "arabic", f"franco_strong:{next(iter(strong))}"
    broad = words & FRANCO_BROAD
    if len(broad) >= 2:
        return "arabic", f"franco_multi:{','.join(list(broad)[:3])}"
    if len(broad) == 1 and len(words) <= 5:
        return "arabic", f"franco_single_short:{next(iter(broad))}"
    return "english", "default"


async def resolve_language(session_key: str, user_text: str) -> tuple[str, str]:
    text_lower = user_text.lower().strip()
    switch_to: str | None = None
    for phrase in ARABIC_PHRASES:
        if phrase in text_lower:
            switch_to = "arabic"
            break
    if not switch_to:
        for phrase in ENGLISH_PHRASES:
            if phrase in text_lower:
                switch_to = "english"
                break
    if switch_to:
        if redis_client:
            await redis_client.setex(f"lang:{session_key}", LANG_TTL, switch_to)
        return switch_to, "override"
    if redis_client:
        pinned = await redis_client.get(f"lang:{session_key}")
        if pinned:
            return pinned, "pinned"
    lang, signal = detect_language_from_text(user_text)
    return lang, f"detected:{signal}"


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def normalize_query(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = re.sub(r"\.{3,}", ".", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_tavus_internal(text: str) -> bool:
    return "<user_audio_analysis>" in text or "The speaker sounds" in text


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse_chunk(content: str, model: str, finish: bool = False) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if not finish else {}, "finish_reason": "stop" if finish else None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def silent_stream(model: str):
    yield sse_chunk("", model, finish=True)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Visual aid generation — runs as standalone coroutine, not inside generator
# ---------------------------------------------------------------------------

async def generate_visual(question: str, answer: str, session_key: str):
    print(f"generate_visual START key={session_key}", flush=True)

    if not ANTHROPIC_API_KEY:
        print("generate_visual: no ANTHROPIC_API_KEY", flush=True)
        return
    if not redis_client:
        print("generate_visual: no redis_client", flush=True)
        return

    prompt = (
        "You are a visual aid generator for an AI tutor.\n\n"
        f"Student question: {question}\n"
        f"Tutor answer: {answer}\n\n"
        "Decide if a visual aid would genuinely help (grammar tables, verb conjugations, "
        "comparisons, timelines, vocabulary lists, step-by-step processes, email structure). "
        "Simple conversational exchanges do NOT need visuals.\n\n"
        "If yes: respond with ONLY a clean self-contained HTML snippet using inline styles. "
        "White background, teal (#0A5F6D) accent, max-width 100%. "
        "Do NOT wrap in markdown code fences. Return raw HTML only.\n"
        "If no: respond with exactly: NO_VISUAL"
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
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
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            result = resp.json()["content"][0]["text"].strip()
            print(f"generate_visual key={session_key}: {result[:80]}", flush=True)

            if result == "NO_VISUAL":
                print(f"generate_visual: NO_VISUAL for key={session_key}", flush=True)
                return

            # Strip markdown fences if present
            result = re.sub(r'^```[a-z]*\n?', '', result, flags=re.MULTILINE)
            result = re.sub(r'\n?```$', '', result, flags=re.MULTILINE)
            result = result.strip()

            if "<" in result:
                await redis_client.setex(f"artifact:{session_key}", 120, result)
                print(f"Visual stored → artifact:{session_key} ({len(result)} chars)", flush=True)
            else:
                print(f"generate_visual: result has no HTML tags, skipping", flush=True)

    except Exception as e:
        print(f"Visual generation error: {e}", flush=True)


# ---------------------------------------------------------------------------
# Tavus persona management
# ---------------------------------------------------------------------------

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
                },
                "tts": {"tts_engine": "tavus-auto", "tts_emotion_control": False},
                "stt": {"stt_engine": "tavus-whisper"},
            },
        },
    )
    resp.raise_for_status()
    TAVUS_PAL_ID = resp.json()["persona_id"]
    return TAVUS_PAL_ID


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass
    return {"status": "healthy", "pal_id": TAVUS_PAL_ID or "not yet created", "redis": redis_ok}


@app.get("/v1/artifact/{session_key}")
async def get_artifact(session_key: str):
    if redis_client:
        html = await redis_client.get(f"artifact:{session_key}")
        if html:
            await redis_client.delete(f"artifact:{session_key}")
            return {"html": html, "session_key": session_key}
    return {"html": None, "session_key": session_key}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _token: str = Depends(verify_token),
):
    messages = request.messages

    user_query = next((m.content for m in reversed(messages) if m.role == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    if is_tavus_internal(user_query):
        print(f"Dropping internal Tavus message: {user_query[:60]!r}", flush=True)
        return StreamingResponse(silent_stream(request.model), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    ctx             = extract_enara_context(messages)
    course_id       = request.course_id       or ctx.get("course_id", "336627af-732e-4349-bda8-b73c702dcf42")
    section_ids     = request.section_ids     or ctx.get("section_ids", [])
    teaching_method = request.teaching_method or ctx.get("teaching_method", "socratic")
    session_key     = extract_session_key(messages)
    language, lang_source = await resolve_language(session_key, user_query)
    normalized_query = normalize_query(user_query)

    if not normalized_query:
        return StreamingResponse(silent_stream(request.model), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    print(f"chat_completions: lang={language} ({lang_source}) session={session_key} query={normalized_query[:50]!r}", flush=True)

    enara_payload = {
        "course_id":       course_id,
        "query":           normalized_query,
        "section_ids":     section_ids,
        "teaching_method": teaching_method,
        "chat_history":    build_chat_history(messages),
        "language":        language,
    }

    # Call Enara backend first (not inside generator — avoids task cancellation)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ENARA_BASE_URL}/chat/query",
                headers={"X-API-Key": ENARA_API_KEY, "Content-Type": "application/json"},
                json=enara_payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        async def timeout_stream():
            yield sse_chunk("Sorry, the tutoring service timed out. Please try again.", request.model)
            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"
        return StreamingResponse(timeout_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except httpx.HTTPStatusError as e:
        async def error_stream():
            yield sse_chunk(f"Backend error ({e.response.status_code}). Please try again.", request.model)
            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    answer = data.get("answer", "")
    print(f"Enara answer: {answer[:60]!r}", flush=True)

    # Fire visual generation as top-level task BEFORE returning StreamingResponse
    asyncio.create_task(generate_visual(normalized_query, answer, session_key))
    print(f"Visual task created for session={session_key}", flush=True)

    async def stream_answer():
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield sse_chunk(word + (" " if i < len(words) - 1 else ""), request.model)
        yield sse_chunk("", request.model, finish=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_answer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(_token: str = Depends(verify_token)):
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            pal_id = await get_or_create_pal(client)
            asyncio.create_task(prewarm_modal())
            payload = {
                "persona_id": pal_id,
                "replica_id": TAVUS_REPLICA_ID,
                "conversation_name": f"Enara Tutor - {uuid.uuid4().hex[:8]}",
                "properties": {"language": "Arabic"},
            }
            print(f"Tavus create: persona_id={pal_id} api_key={TAVUS_API_KEY[:8]}...", flush=True)
            resp = await client.post(
                "https://tavusapi.com/v2/conversations",
                headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
                json=payload,
            )
            print(f"Tavus response {resp.status_code}: {resp.text}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            return {"conversation_url": data["conversation_url"], "conversation_id": data["conversation_id"]}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.status_code} - {e.response.text}")


@app.delete("/v1/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(conversation_id: str, _token: str = Depends(verify_token)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.delete(
                f"https://tavusapi.com/v2/conversations/{conversation_id}",
                headers={"x-api-key": TAVUS_API_KEY},
            )
            resp.raise_for_status()
            session_key = conversation_id[-8:]
            if redis_client:
                await redis_client.delete(f"lang:{session_key}", f"artifact:{session_key}")
            return {"ended": True, "conversation_id": conversation_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.status_code} - {e.response.text}")
