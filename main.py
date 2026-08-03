"""
Enara AI <-> Tavus Adapter
"""

import json
import time
import uuid
import httpx
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Enara Avatar Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://enara-platform-production-12bb.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

ENARA_BASE_URL   = os.environ["ENARA_BASE_URL"]
ENARA_API_KEY    = os.environ["ENARA_API_KEY"]
ADAPTER_TOKEN    = os.environ["ADAPTER_TOKEN"]
TAVUS_API_KEY    = os.environ["TAVUS_API_KEY"]
TAVUS_REPLICA_ID = os.environ["TAVUS_REPLICA_ID"]

# PAL ID — created once via /v2/pals and stored here.
# If empty, the adapter will create one on first conversation request.
TAVUS_PAL_ID = os.environ.get("TAVUS_PAL_ID", "p9892496020e")


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
    for msg in messages:
        if msg.role == "system":
            try:
                first_line = msg.content.strip().split("\n")[0]
                return json.loads(first_line)
            except (json.JSONDecodeError, IndexError):
                pass
    return {}


def build_chat_history(messages: list[ChatMessage]) -> list[dict]:
    history = []
    for msg in messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.content})
    return history


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
    """Returns the PAL ID to use for conversations.
    If TAVUS_PAL_ID env var is set, use it directly.
    Otherwise create a new PAL and return its ID.
    """
    global TAVUS_PAL_ID

    if TAVUS_PAL_ID:
        return TAVUS_PAL_ID

    # Create a new PAL with Enara as the custom LLM
    resp = await client.post(
        "https://tavusapi.com/v2/pals",
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
                "Respond in the same language the student uses, Arabic or English."
            ),
            "pipeline_mode": "full",
            "default_face_id": TAVUS_REPLICA_ID,
            "layers": {
                "llm": {
                    "model": "enara-tutor",
                    "base_url": "https://enara-avatar-adapter-production.up.railway.app/v1",
                    "api_key": ADAPTER_TOKEN,
                }
            }
        }
    )
    resp.raise_for_status()
    data = resp.json()
    TAVUS_PAL_ID = data["pal_id"]
    return TAVUS_PAL_ID


@app.get("/health")
async def health():
    return {"status": "healthy", "pal_id": TAVUS_PAL_ID or "not yet created"}


@app.get("/v1/tavus/credits")
async def get_credits(_token: str = Depends(verify_token)):
    """Returns remaining Tavus conversational credits."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "https://tavusapi.com/v2/credits",
                headers={"x-api-key": TAVUS_API_KEY}
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Tavus API error: {e.response.status_code} - {e.response.text}"
            )


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

    enara_payload = {
        "course_id":       course_id,
        "query":           user_query,
        "section_ids":     section_ids,
        "teaching_method": teaching_method,
        "chat_history":    build_chat_history(messages),
    }

    async def generate():
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield sse_chunk(chunk, request.model)

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
    """Ends a Tavus conversation session to stop credit consumption."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.delete(
                f"https://tavusapi.com/v2/conversations/{conversation_id}",
                headers={"x-api-key": TAVUS_API_KEY}
            )
            resp.raise_for_status()
            return {"ended": True, "conversation_id": conversation_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Tavus API error: {e.response.status_code} - {e.response.text}"
            )


@app.post("/v1/tavus/conversation")
async def create_tavus_conversation(
    _token: str = Depends(verify_token)
):
    """Creates a Tavus conversation session using a PAL and returns the embed URL."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            pal_id = await get_or_create_pal(client)

            payload = {
                "persona_id": pal_id,
                "replica_id": TAVUS_REPLICA_ID,
                "conversation_name": f"Enara Tutor - {uuid.uuid4().hex[:8]}"
            }
            print(f"DEBUG sending to Tavus: persona_id={pal_id} replica_id={TAVUS_REPLICA_ID} api_key={TAVUS_API_KEY[:8]}", flush=True)

            resp = await client.post(
                "https://tavusapi.com/v2/conversations",
                headers={
                    "x-api-key": TAVUS_API_KEY,
                    "Content-Type": "application/json"
                },
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
            raise HTTPException(
                status_code=502,
                detail=f"Tavus API error: {e.response.status_code} - {e.response.text}"
            )
