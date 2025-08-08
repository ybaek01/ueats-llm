"""
Uber Eats Simulation - Score/description guaranteed
Environment Variable: OPENAI_API_KEY
"""

import argparse, asyncio, json, pathlib
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI

client = AsyncOpenAI()  # Load api_key from env

HEADLESS       = True
MAX_DOM_CHARS  = 4000

TASK_SYSTEM = """
You are a UX agent interacting with the Uber Eats mobile web site.
Goal: order ONE Buffalo Wings, add a Diet Pepsi, and add ranch dressing.
After each DOM snapshot, reply with ONE JSON only:
  {"action":"click|type|wait","selector":"<css>","text":"<opt>","ms":<opt>}
Stop when checkout is complete or a blocker occurs.
"""

ANALYSIS_SYSTEM = """
You are an expert usability auditor.
Return STRICT JSON ONLY with keys:
{
  "score": <float between 1.0 and 5.0 with 1 decimal>,
  "description": "<one-line persona summary (age, location, diet, goal)>",
  "markdown": "## Persona\\n... (<=200 words; sections: Persona, Critical Issues, Minor Friction, Suggested Improvements)"
}
Rules:
- Output must be only a single JSON object with exactly the three keys above.
- score must be one decimal like 4.3.
- markdown MUST start with "## Persona".
"""

async def act_with_llm(page, persona):
    history = []
    while True:
        dom = (await page.content())[:MAX_DOM_CHARS]
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": TASK_SYSTEM + f"\nPersona:\n{json.dumps(persona, ensure_ascii=False)}"},
                {"role": "user",
                 "content": f"DOM:\n{dom}\n\nHistory:{json.dumps(history, ensure_ascii=False)}"}
            ],
            temperature=0.2,
        )
        cmd = json.loads(resp.choices[0].message.content)
        act, sel, txt = cmd.get("action"), cmd.get("selector"), cmd.get("text", "")
        try:
            if act == "click":
                await page.click(sel, timeout=8000)
            elif act == "type":
                await page.fill(sel, txt, timeout=8000)
            elif act == "wait":
                await page.wait_for_timeout(int(cmd.get("ms", 1000)))
            else:
                history.append({"error": f"unknown action {act}"}); break
            history.append(cmd)
        except PWTimeout:
            history.append({"error": f"timeout @ {sel}"}); break
        if "Thank you" in await page.content() or len(history) > 60:
            break
    return {"history": history}

async def run_one(play, persona):
    browser = await play.webkit.launch(headless=HEADLESS)
    device = play.devices.get("iPhone 15") or {"viewport": {"width": 393, "height": 852}, "user_agent": "Mozilla/5.0"}
    context = await browser.new_context(**device)
    page    = await context.new_page()
    await page.goto("https://www.ubereats.com", timeout=30000)
    result  = await act_with_llm(page, persona)
    await browser.close()
    return result

def _fallback_description(p):
    return f"{p.get('age','?')}yo in {p.get('location','?')} ({p.get('diet','none')}); goal: {p.get('goal','n/a')}"

async def main(args):
    personas = json.load(open(args.personas))
    root     = pathlib.Path(args.output); root.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        for idx, persona in enumerate(personas, 1):
            pid  = persona.get("id") or f"P-{idx:02}"
            sess = root / pid
            if (sess / "issues.json").exists():
                print(f"{pid} ✔︎ Skip"); continue
            sess.mkdir(parents=True, exist_ok=True); print(f"▶ {pid}")

            result = await run_one(p, persona)

            analysis_resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {"role": "user",   "content": json.dumps({"persona": persona, **result}, ensure_ascii=False)}
                ],
                temperature=0
            )
            try:
                aobj = json.loads(analysis_resp.choices[0].message.content)
                score = float(aobj.get("score", 3.0))
                score = max(1.0, min(5.0, float(f"{score:.1f}")))
                description = aobj.get("description") or _fallback_description(persona)
                markdown = aobj.get("markdown") or f"## Persona\n{description}\n"
            except Exception:
                score = 3.0
                description = _fallback_description(persona)
                markdown = f"## Persona\n{description}\n\n## Critical Issues\n- \n## Minor Friction\n- \n## Suggested Improvements\n1. \n"

            (sess / "issues.md").write_text(
                f"{score:.1f} / 5.0\n"
                f"**Description:** {description}\n\n"
                f"{markdown}", encoding="utf-8"
            )
            with open(sess / "issues.json", "w", encoding="utf-8") as f:
                json.dump({
                    "persona": persona,
                    "run": result,
                    "analysis": markdown,
                    "score": float(f"{score:.1f}"),
                    "description": description
                }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True)
    ap.add_argument("--output",   required=True)
    asyncio.run(main(ap.parse_args()))
