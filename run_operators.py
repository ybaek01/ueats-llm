"""
Uber Eats Simulation - LLM v1 API
pip install "openai>=1.0.0" playwright rapidfuzz
playwright install
Environment Variable:  OPENAI_API_KEY
"""

import argparse, asyncio, json, os, pathlib
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI

# ────────── OpenAI 비동기 클라이언트 ────────── #
client = AsyncOpenAI()                     # api_key는 env에서 로드

HEADLESS       = True
MAX_DOM_CHARS  = 4000

TASK_SYSTEM = """
You are a UX agent interacting with the Uber Eats **mobile web** site.
Goal: order ONE Buffalo Wings, add a Diet Pepsi, and add ranch dressing.
After each DOM snapshot, reply with ONE JSON only:
  {"action":"click|type|wait","selector":"<css>","text":"<opt>","ms":<opt>}
Stop when checkout is complete or a blocker occurs.
"""

ANALYSIS_SYSTEM = """
Return a markdown report:
1. Persona name
2. Critical issues
3. Minor friction
4. Suggested improvements
≤ 200 words.
"""

# ────────── LLM ↔ Playwright 루프 ────────── #
async def act_with_llm(page, persona):
    history = []
    while True:
        dom = (await page.content())[:MAX_DOM_CHARS]

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TASK_SYSTEM + f"\nPersona:\n{json.dumps(persona, ensure_ascii=False)}"},
                {"role": "user",   "content": f"DOM:\n{dom}\n\nHistory:{history}"}
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
                history.append({"error": f"unknown action {act}"})
                break
            history.append(cmd)
        except PWTimeout:
            history.append({"error": f"timeout @ {sel}"})
            break

        if "Thank you" in await page.content():
            break
        if len(history) > 60:
            break
    return {"history": history}

# ────────── 퍼소나 한 명 실행 ────────── #
async def run_one(play, persona, out_dir):
    browser = await play.webkit.launch(headless=HEADLESS)
    context = await browser.new_context(**play.devices["iPhone 16"])
    page    = await context.new_page()

    await page.goto("https://www.ubereats.com", timeout=30000)
    result = await act_with_llm(page, persona)
    await browser.close()
    return result

# ────────── 메인 ────────── #
async def main(args):
    personas = json.load(open(args.personas))
    root     = pathlib.Path(args.output); root.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        for idx, persona in enumerate(personas, 1):
            tag, sess = persona.get("id") or f"P-{idx:02}", root / (persona.get("id") or f"P-{idx:02}")
            if (sess / "issues.json").exists():
                print(f"{tag} ✔︎ 스킵")
                continue
            sess.mkdir(parents=True, exist_ok=True)
            print(f"▶ {tag}")

            result = await run_one(p, persona, sess)

            analysis_resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {"role": "user",   "content": json.dumps({"persona": persona, **result}, ensure_ascii=False)}
                ],
                temperature=0
            )
            analysis = analysis_resp.choices[0].message.content

            with open(sess / "issues.md", "w") as f:
                f.write(analysis)
            with open(sess / "issues.json", "w") as f:
                json.dump({"persona": persona, "run": result, "analysis": analysis},
                          f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True)
    ap.add_argument("--output",   required=True)
    asyncio.run(main(ap.parse_args()))
