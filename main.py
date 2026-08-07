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

FALLBACK_MESSAGES = {
    "arabic": "عذراً، لم أفهم جيداً. هل يمكنك إعادة الصياغة أو توضيح سؤالك؟",
    "english": "Sorry, I didn't quite catch that. Could you rephrase or clarify your question?",
}

UNCLEAR_MESSAGES = {
    "arabic": "لم أسمعك بوضوح. هل يمكنك التحدث مرة أخرى؟",
    "english": "I didn't hear that clearly. Could you say that again?",
}

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
        try:
            redis_url = REDIS_URL
            if redis_url.startswith("redis://"):
                redis_url = "rediss://" + redis_url[8:]
            print(f"[REDIS] Connecting to: {redis_url[:50]}...", flush=True)
            redis_client = aioredis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await redis_client.ping()
            print("[REDIS] Connected successfully", flush=True)
        except Exception as e:
            print(f"[REDIS] Connection failed: {type(e).__name__}: {e}", flush=True)
            redis_client = None
    else:
        print("[REDIS] WARNING: No REDIS_URL set - visuals will NOT work", flush=True)

    asyncio.create_task(warmup_modal())
    yield

    if redis_client:
        try:
            await redis_client.aclose()
            print("[REDIS] Connection closed", flush=True)
        except Exception as e:
            print(f"[REDIS] Error closing: {e}", flush=True)


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
            for line in reversed(msg.content.split("\n")):
                line = line.strip()
                if line.startswith("Session:"):
                    val = line.replace("Session:", "").strip()
                    if val:
                        return val
            match = re.search(r"\b(c[0-9a-f]{15,})\b", msg.content)
            if match:
                return match.group(1)
    return uuid.uuid4().hex


def build_chat_history(messages: list[ChatMessage]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})
    return history


# ---------------------------------------------------------------------------
# Language detection — multi-dialect, score-based
# ---------------------------------------------------------------------------

ARABIC_SWITCH_PHRASES = [
    "in arabic", "respond in arabic", "answer in arabic", "explain in arabic",
    "speak arabic", "say it in arabic", "tell me in arabic", "switch to arabic",
    "use arabic", "باللغة العربية", "بالعربي", "بالعربية", "بالعربية من فضلك",
    "كلمني عربي", "رد بالعربي", "تكلم عربي",
]
ENGLISH_SWITCH_PHRASES = [
    "in english", "respond in english", "answer in english", "explain in english",
    "speak english", "say it in english", "tell me in english", "switch to english",
    "use english", "بالانجليزي", "بالإنجليزية", "بالانجليزية",
    "كلمني انجليزي", "رد بالانجليزي",
]

# Franco-Arabic number-as-letter context — unambiguous signal
FRANCO_NUMBER_CONTEXT = re.compile(
    r"\b(ya3ni|ya3ny|3ayiz|3arif|3andi|3alshan|ba3d|b3d|2ana|2enta|7aga|7abibi|sa7|"
    r"msh|knt|bnt|w2t|f2|m3|t3|b3|l2|3l|3n|f3l)\b",
    re.IGNORECASE
)

# Egyptian Arabic
EGYPTIAN = {
    "ya3ni", "ya3ny", "ezay", "leih", "leh", "mafish", "mafesh",
    "yalla", "khalas", "delwa2ty", "ba3dein", "b3deen", "3alshan", "3shan",
    "3andi", "3ndi", "3ayiz", "aayiz", "mumkin", "momken", "lazim",
    "mesh", "msh", "keda", "kida", "zay", "meen", "fein", "fyn",
    "aho", "tamam", "tayeb", "tayyeb", "howa", "hiya", "ihna",
    "shoof", "shof", "7aga", "aslan", "awy", "gedan", "walla",
    "ana", "enta", "enti", "bas", "law", "wala", "fe", "fi",
    "beta3", "beta3ti", "el", "hategy", "gayy", "raye7",
    "ma3lesh", "maashi", "afandem",
}

# Levantine (Syrian, Lebanese, Palestinian, Jordanian)
LEVANTINE = {
    "shu", "shno", "kifak", "kifik", "kif", "wein", "wayn", "wen",
    "haida", "hayde", "hek", "la2", "mnih", "mnee7", "3m", "3am",
    "baddak", "baddi", "bade", "hayda", "shou",
    "halla2", "hala2", "hon", "hone", "mno", "elo",
    "shi", "ktir", "leish", "laish", "yimkin",
    "akh", "wallah", "mashallah", "habibi", "habibti",
}

# Gulf (Saudi, UAE, Kuwait, Qatar, Bahrain)
GULF = {
    "shlon", "shlonk", "shfih", "laish", "liwain", "wain",
    "abee", "abi", "agdar", "zain", "zayn",
    "esh", "cham", "shda3wa", "mafi", "khosh",
    "inzain", "tara", "wayed", "kafi",
    "mashallah", "wallah", "yalla", "habibi",
}

# Moroccan Darija
MOROCCAN = {
    "wach", "bghit", "ma3andish", "kifash", "fin", "mnin",
    "wakha", "ewa", "mzyan", "bzzaf", "shwiya", "daba",
    "hna", "huma", "nta", "nti",
    "mashi", "makaynsh", "kayn", "fach",
    "3andek", "3andi",
}

# Phonetic Arabic patterns (common in Whisper STT output)
PHONETIC_PATTERNS = [
    re.compile(r"\binsh[ae]ll?[ae]h\b", re.IGNORECASE),
    re.compile(r"\bwall?[ae]h\b", re.IGNORECASE),
    re.compile(r"\bhab[ie]b[it]?[ia]?\b", re.IGNORECASE),
    re.compile(r"\bmash?all?[ae]h\b", re.IGNORECASE),
    re.compile(r"\b(marh?aba|ahlan|salam)\b", re.IGNORECASE),
    re.compile(r"\byall?a\b", re.IGNORECASE),
    re.compile(r"\b(shukran|shoukran)\b", re.IGNORECASE),
    re.compile(r"\bmabrook\b", re.IGNORECASE),
    re.compile(r"\btayyeb\b", re.IGNORECASE),
    re.compile(r"\b(aiwa|aywa)\b", re.IGNORECASE),
    re.compile(r"\bmumk[iy]n\b", re.IGNORECASE),
    re.compile(r"\b(tab3an|tab3n)\b", re.IGNORECASE),
]

ALL_ARABIC_WORDS = EGYPTIAN | LEVANTINE | GULF | MOROCCAN


def _score_arabic(text: str) -> tuple[int, list[str]]:
    """Return (confidence_score, signals). Score >= 2 means Arabic."""
    signals = []
    text_lower = text.lower()
    words = set(text_lower.split())

    # Arabic script — definitive
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    total_chars = len(text.replace(" ", ""))
    if total_chars > 0 and arabic_chars > 0:
        ratio = arabic_chars / total_chars
        if ratio > 0.3:
            signals.append(f"arabic_script:{ratio:.0%}")
            return 10, signals
        elif ratio > 0.1:
            signals.append(f"arabic_script_mixed:{ratio:.0%}")
            return 8, signals

    # Franco number patterns — very high confidence
    if FRANCO_NUMBER_CONTEXT.search(text_lower):
        match = FRANCO_NUMBER_CONTEXT.search(text_lower)
        signals.append(f"franco_number:{match.group()}")
        return 9, signals

    # Dialect word matching
    matched_words = words & ALL_ARABIC_WORDS
    if matched_words:
        score = min(len(matched_words) * 3, 9)
        signals.append(f"dialect_words:{','.join(list(matched_words)[:4])}")
        return score, signals

    # Phonetic patterns from STT
    for pattern in PHONETIC_PATTERNS:
        m = pattern.search(text_lower)
        if m:
            signals.append(f"phonetic:{m.group()}")
            return 6, signals

    # Arabic phoneme clusters in short text
    if len(words) <= 3:
        arabic_phonemes = re.compile(
            r"\b(kh|gh|3|7|2|ain|ghayn|qaf)\w*\b",
            re.IGNORECASE
        )
        if arabic_phonemes.search(text_lower):
            signals.append("arabic_phonemes")
            return 4, signals

    return 0, []


def detect_language_from_text(text: str) -> tuple[str, str]:
    """Returns (language, signal_description)."""
    if not text.strip():
        return "english", "empty"

    text_lower = text.lower().strip()

    # 1. Explicit switch phrases
    for phrase in ARABIC_SWITCH_PHRASES:
        if phrase in text_lower:
            return "arabic", f"explicit:{phrase}"
    for phrase in ENGLISH_SWITCH_PHRASES:
        if phrase in text_lower:
            return "english", f"explicit:{phrase}"

    # 2. Score-based detection
    score, signals = _score_arabic(text)
    if score >= 2:
        return "arabic", "|".join(signals)

    # 3. Code-switching: strong Arabic word in short sentence
    words = set(text_lower.split())
    strong_match = words & (EGYPTIAN | LEVANTINE | GULF)
    if strong_match and len(words) <= 8:
        return "arabic", f"codemix:{next(iter(strong_match))}"

    return "english", "default"


async def resolve_language(session_key: str, user_text: str) -> tuple[str, str]:
    """Priority: explicit override > Redis pin > auto-detect."""
    text_lower = user_text.lower().strip()

    switch_to: str | None = None
    for phrase in ARABIC_SWITCH_PHRASES:
        if phrase in text_lower:
            switch_to = "arabic"
            break
    if not switch_to:
        for phrase in ENGLISH_SWITCH_PHRASES:
            if phrase in text_lower:
                switch_to = "english"
                break

    if switch_to:
        if redis_client:
            try:
                await redis_client.setex(f"lang:{session_key}", LANG_TTL, switch_to)
                print(f"[LANG] Pinned {switch_to} for session={session_key[:8]}", flush=True)
            except Exception as e:
                print(f"[LANG] Redis pin error: {e}", flush=True)
        return switch_to, "override"

    if redis_client:
        try:
            pinned = await redis_client.get(f"lang:{session_key}")
            if pinned:
                return pinned, "pinned"
        except Exception as e:
            print(f"[LANG] Redis get error: {e}", flush=True)

    lang, signal = detect_language_from_text(user_text)
    print(f"[LANG] detected={lang} signal={signal} text={user_text[:40]!r}", flush=True)
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


def extract_user_text(message: str) -> str | None:
    """Strip Tavus internal XML tags. Returns clean user text, or None if nothing left."""
    import re
    cleaned = re.sub(r'<user_audio_analysis>.*?</user_audio_analysis>', '', message, flags=re.DOTALL).strip()
    return cleaned if cleaned else None

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
# Visual aid generation
# ---------------------------------------------------------------------------

async def generate_visual(question: str, answer: str, session_key: str):
    print(f"[VISUAL] Starting for session_key={session_key}", flush=True)

    if not ANTHROPIC_API_KEY:
        print("[VISUAL] No ANTHROPIC_API_KEY set, skipping", flush=True)
        return
    if not redis_client:
        print("[VISUAL] No Redis connection, skipping", flush=True)
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
        print(f"[VISUAL] Calling Claude API for session_key={session_key}", flush=True)
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
            print(f"[VISUAL] Claude response status: {resp.status_code}", flush=True)
            resp.raise_for_status()
            result = resp.json()["content"][0]["text"].strip()
            print(f"[VISUAL] Claude response: {result[:80]}", flush=True)

            if result == "NO_VISUAL":
                print(f"[VISUAL] No visual needed for session_key={session_key}", flush=True)
                return

            # Strip markdown fences if present
            result = re.sub(r'^```[a-z]*\n?', '', result, flags=re.MULTILINE)
            result = re.sub(r'\n?```$', '', result, flags=re.MULTILINE)
            result = result.strip()

            if "<" in result:
                try:
                    await redis_client.setex(f"artifact:{session_key}", 120, result)
                    print(f"[VISUAL] ✅ Stored artifact for session_key={session_key} ({len(result)} bytes)", flush=True)
                except Exception as e:
                    print(f"[VISUAL] ❌ Redis setex failed: {e}", flush=True)
            else:
                print("[VISUAL] No HTML tags in response, skipping", flush=True)

    except Exception as e:
        print(f"[VISUAL] ❌ Error: {type(e).__name__}: {e}", flush=True)


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
            print("[HEALTH] Redis: ✅", flush=True)
        except Exception as e:
            print(f"[HEALTH] Redis: ❌ ({e})", flush=True)
    else:
        print("[HEALTH] Redis: ⚠️ NOT CONNECTED", flush=True)
    return {
        "status": "healthy",
        "pal_id": TAVUS_PAL_ID or "not yet created",
        "redis": "connected" if redis_ok else "disconnected",
    }


@app.get("/v1/artifact/{session_key}")
async def get_artifact(session_key: str):
    if not redis_client:
        print(f"[ARTIFACT] Redis not available for session_key={session_key}", flush=True)
        return {"html": None, "session_key": session_key}
    try:
        html = await redis_client.get(f"artifact:{session_key}")
        if html:
            await redis_client.delete(f"artifact:{session_key}")
            print(f"[ARTIFACT] ✅ Retrieved and deleted artifact for session_key={session_key}", flush=True)
            return {"html": html, "session_key": session_key}
    except Exception as e:
        print(f"[ARTIFACT] ❌ Redis error: {e}", flush=True)
    print(f"[ARTIFACT] No artifact found for session_key={session_key}", flush=True)
    return {"html": None, "session_key": session_key}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _token: str = Depends(verify_token),
):
    messages = request.messages

    user_query = next((m.content for m in reversed(messages) if m.role == "user"), None)
    print(f"[STT] raw input: {user_query!r}", flush=True)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found")

    user_query = extract_user_text(user_query)
if not user_query:
    print(f"[CHAT] Dropping pure internal Tavus message (no user text)")
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
        print("[CHAT] Empty query after normalization", flush=True)
        return StreamingResponse(silent_stream(request.model), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if len(normalized_query.strip()) <= 2:
        unclear_msg = UNCLEAR_MESSAGES.get(language, UNCLEAR_MESSAGES["english"])
        print(f"[CHAT] Query too short ({len(normalized_query)} chars) — asking to repeat", flush=True)
        async def unclear_stream():
            for word in unclear_msg.split(" "):
                yield sse_chunk(word + " ", request.model)
            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"
        return StreamingResponse(unclear_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    print(f"[CHAT] lang={language} ({lang_source}) | session={session_key[:8]} | query={normalized_query[:50]!r}", flush=True)

    enara_payload = {
        "course_id":       course_id,
        "query":           normalized_query,
        "section_ids":     section_ids,
        "teaching_method": teaching_method,
        "chat_history":    build_chat_history(messages),
        "language":        language,
    }

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
        timeout_msg = "عذراً، انتهت المهلة. حاول مرة أخرى." if language == "arabic" else "Sorry, the service timed out. Please try again."
        async def timeout_stream():
            yield sse_chunk(timeout_msg, request.model)
            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"
        return StreamingResponse(timeout_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except httpx.HTTPStatusError as e:
        error_msg = "عذراً، حدث خطأ. حاول مرة أخرى." if language == "arabic" else "Something went wrong. Please try again."
        async def error_stream():
            yield sse_chunk(error_msg, request.model)
            yield sse_chunk("", request.model, finish=True)
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    answer = data.get("answer", "").strip()
    print(f"[CHAT] Enara answered: {answer[:60]!r}", flush=True)

    if not answer:
        answer = FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["english"])
        print(f"[CHAT] Empty answer — using fallback in {language}", flush=True)

    asyncio.create_task(generate_visual(normalized_query, answer, session_key))
    print(f"[CHAT] Visual task created for session_key={session_key[:8]}", flush=True)

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

            session_id = uuid.uuid4().hex

            payload = {
                "persona_id": pal_id,
                "replica_id": TAVUS_REPLICA_ID,
                "conversation_name": f"Enara Tutor - {session_id[:8]}",
                "conversational_context": f"Session: {session_id}",
                "properties": {"language": "Arabic"},  # STT hint for bilingual transcription
            }
            print(f"[TAVUS] Creating conversation: persona_id={pal_id} session={session_id[:8]}", flush=True)
            resp = await client.post(
                "https://tavusapi.com/v2/conversations",
                headers={"x-api-key": TAVUS_API_KEY, "Content-Type": "application/json"},
                json=payload,
            )
            print(f"[TAVUS] Response {resp.status_code}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            tavus_conv_id = data["conversation_id"]

            if redis_client:
                try:
                    await redis_client.setex(f"tavus_id:{session_id}", 86400, tavus_conv_id)
                except Exception as e:
                    print(f"[TAVUS] Redis store error: {e}", flush=True)

            print(f"[TAVUS] session_id={session_id} tavus_conv_id={tavus_conv_id}", flush=True)
            return {"conversation_url": data["conversation_url"], "conversation_id": session_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Tavus API error: {e.response.status_code} - {e.response.text}")


@app.delete("/v1/tavus/conversation/{conversation_id}")
async def end_tavus_conversation(conversation_id: str, _token: str = Depends(verify_token)):
    tavus_conv_id = conversation_id
    if redis_client:
        try:
            stored = await redis_client.get(f"tavus_id:{conversation_id}")
            if stored:
                tavus_conv_id = stored
                print(f"[END] Resolved tavus_conv_id={tavus_conv_id} for session={conversation_id[:8]}", flush=True)
            else:
                print(f"[END] No Redis mapping for session={conversation_id[:8]}, using as-is", flush=True)
        except Exception as e:
            print(f"[END] Redis lookup error: {e}", flush=True)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.delete(
                f"https://tavusapi.com/v2/conversations/{tavus_conv_id}",
                headers={"x-api-key": TAVUS_API_KEY},
            )
            resp.raise_for_status()
            print(f"[END] Tavus conversation {tavus_conv_id} ended", flush=True)
        except httpx.HTTPStatusError as e:
            print(f"[END] Tavus delete error: {e.response.status_code} - {e.response.text}", flush=True)

    if redis_client:
        try:
            await redis_client.delete(
                f"tavus_id:{conversation_id}",
                f"lang:{conversation_id}",
                f"artifact:{conversation_id}",
            )
            print(f"[END] Redis keys cleaned for session={conversation_id[:8]}", flush=True)
        except Exception as e:
            print(f"[END] Redis cleanup error: {e}", flush=True)

    return {"ended": True, "conversation_id": conversation_id}
