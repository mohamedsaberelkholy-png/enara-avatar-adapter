async def generate_claude_visual_artifact(user_prompt: str) -> Optional[str]:
    """
    Calls Anthropic Claude API (Haiku 4.5) to generate self-contained, 
    beautifully styled HTML/CSS visual cards for the student's learning session.
    """
    if not ANTHROPIC_API_KEY:
        print("[WARNING] ANTHROPIC_API_KEY is not set. Skipping Claude artifact generation.")
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

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": "claude-haiku-4.5",  # <-- Updated to Claude Haiku 4.5
        "max_tokens": 1500,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Create an educational visual card for this topic: {user_prompt}"}
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post("[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                html_code = data["content"][0]["text"].strip()
                if html_code.startswith("```html"):
                    html_code = html_code.replace("```html", "").replace("```", "").strip()
                return html_code
            else:
                print(f"[CLAUDE ERROR] Status: {resp.status_code}, Response: {resp.text}")
                return None
        except Exception as e:
            print(f"[CLAUDE EXCEPTION] {e}")
            return None
