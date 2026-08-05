# ---------------------------------------------------------------------------
# Helper: Call Anthropic Claude safely (Haiku & Sonnet fallback)
# ---------------------------------------------------------------------------
async def generate_claude_visual_artifact(user_prompt: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        print("[WARNING] ANTHROPIC_API_KEY is not configured in environment.")
        return None

    system_prompt = """
    You are an expert AI Tutor visual designer for an educational platform called Enara.
    Your job is to generate self-contained, modern, beautiful HTML visual aids to help students learn.
    
    RULES:
    1. Output strictly ONLY raw HTML content inside <div> tags (no ```html code fences, no extra conversational text).
    2. Support Arabic RTL (Right-to-Left) direction if prompt is in Arabic (direction: rtl; text-align: right;).
    3. Use clean inline CSS with modern UI cards, clear headers, icons/emojis, and bullet points.
    4. Make the design modern, responsive, readable, and visually appealing.
    """

    # Resolve Base URL safely with explicit https:// scheme
    raw_base_url = os.getenv("ANTHROPIC_BASE_URL", "[https://api.anthropic.com](https://api.anthropic.com)")
    if not raw_base_url.startswith("http://") and not raw_base_url.startswith("https://"):
        base_url = f"https://{raw_base_url}"
    else:
        base_url = raw_base_url

    endpoint_url = f"{base_url.rstrip('/')}/v1/messages"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Model identifiers list
    models_to_try = [
        os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022"
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
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
                    print(f"[CLAUDE ERROR] Model '{model}' returned status {resp.status_code}: {resp.text}")
            except httpx.RequestError as exc:
                print(f"[HTTPX REQUEST EXCEPTION] Failed for model '{model}' at '{endpoint_url}': {exc}")
            except Exception as e:
                print(f"[CLAUDE EXCEPTION] {e}")

    return None
