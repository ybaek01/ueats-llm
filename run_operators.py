"""
run_operators.py
- Uber Eats mobile web UX simulation & analysis (HCI-first, pre-checkout only)
- English outputs. Avoid network/latency talk; DO NOT place an order.
- Token-efficient prompting: DOM/Persona/History digests; task placed LAST.
- Minimize timeouts via selector pre-check + soft waits.
- Human-centered notes prioritized; if none, don't fabricate — 5/5 is OK.
- Scoring: integer 1–5 using pre-checkout milestones + severe human-centered notes.
- Robust navigation: domcontentloaded, geolocation/locale/timezone, engine fallback.
- ENV: OPENAI_API_KEY
"""

import argparse, asyncio, json, pathlib, hashlib, random, re, time
from typing import Dict, Any, List
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI

# ---------------------------
# OpenAI client
# ---------------------------
client = AsyncOpenAI()  # api_key from env

# ---------------------------
# Globals
# ---------------------------
MAX_DOM_CHARS  = 12000
MAX_STEPS      = 60
NAV_TIMEOUT_MS = 90000  # 90s navigation budget

# ---------------------------
# 1) System prompt (compact, JSON-only)
# ---------------------------
COMPACT_SYSTEM = """
You are a mobile UX agent on Uber Eats (iPhone). Output STRICT JSON only:
{"action":"click|type|wait_ms|wait_for|note","selector":"<css|text>","text":"<opt>","ms":<opt>,"state":"<opt>","tag":"<opt>","detail":"<opt>"}

Rules:
- PRE-CHECKOUT ONLY. NEVER click: "Place order", "Pay", "Apple Pay", "Checkout".
- Prefer state waits: wait_for {state: visible|attached|hidden|detached} (≤1500ms).
  wait_ms only if necessary (avoid exactly 500ms; prefer 350ms or 700ms).
- Use diet/allergen filters/search; verify item details vs persona needs.
- On human-centered issues, emit ONE note(tag in: diet_mismatch, allergen_missing, filter_missing,
  label_ambiguous, fee_surprise, upsell_overwhelming, aria_missing, contrast_low, tiny_tap_target; + detail).
- Record milestones as notes: milestone_item_added, milestone_cart_open, milestone_review.
- Stop when cart/review is visible or a blocker occurs.
Think step-by-step silently. OUTPUT JSON ONLY.
""".strip()

ANALYSIS_SYSTEM = r"""
You are a meticulous UX auditor. Analyze the session 'history' AS the persona.
Focus on human-centered issues (diet/allergen labeling & filters, findability, clarity, accessibility).
Do NOT discuss network speed or millisecond waits unless they prevented reaching the pre-checkout review.
If no meaningful critical issue is evident, clearly say so ("None observed.").

Return STRICT JSON with EXACTLY three string keys: "score", "description", "markdown".
- "score": one of {"1","2","3","4","5"} (integer string).
- "description": ONE line in English: "<age>yo in <location> (<income>; <diet>/<accessibility>); goal: <goal>."
- "markdown": <=200 words in English, first-person voice, using EXACT section headers:

## Persona
Who I am, brief context (age, income, time pressure), and what I tried to do.

## Critical Issues
ONE most impactful, human-centered issue. Prefer evidence from a 'note' in history
(e.g., "step 12 note: diet_mismatch — menu marked vegan but contains meat").
If there is no meaningful critical issue, write "None observed." and move on.

## Minor Friction
1–3 smaller annoyances. Avoid repeating the same phrasing across personas.

## Suggested Improvements
1–3 concrete, persona-tailored fixes (e.g., explicit vegan badges, allergen chips, clearer labels).
"""

# ---------------------------
# 2) Token-efficient digests
# ---------------------------
def digest_dom(html: str, max_chars: int = 3500) -> str:
    # (원하면 script/style 제거 로직 추가 가능)
    return (html or "")[:max_chars]

def digest_persona(p: Dict[str, Any]) -> str:
    return (f"{p.get('age','?')}yo, {p.get('location','?')}, "
            f"{p.get('income','?')}; diet={p.get('diet','none')}, "
            f"accessibility={p.get('accessibility','none')}; goal={p.get('goal','n/a')}")

def digest_history(hist: List[Dict[str, Any]], k: int = 8) -> str:
    if not hist:
        return "None"
    slim = []
    for h in hist[-k:]:
        if 'action' in h:
            a = h['action']
            if a in ('click','type'):
                slim.append({"step": h.get("step"), "action": a, "selector": (h.get("selector") or "")[:80]})
            elif a in ('wait_ms','wait_for'):
                slim.append({"step": h.get("step"), "action": a})
            elif a == 'note':
                slim.append({"step": h.get("step"), "note": f"{h.get('tag')}::{(h.get('detail') or '')[:60]}"})
        elif 'error' in h:
            slim.append({"step": h.get("step"), "error": (h.get("error") or "")[:80]})
        elif 'warn' in h:
            slim.append({"step": h.get("step"), "warn": (h.get("warn") or "")[:80]})
        elif 'info' in h:
            slim.append({"step": h.get("step"), "info": (h.get("info") or "")[:80]})
    return json.dumps(slim, ensure_ascii=False)

# ---------------------------
# 3) Helpers to minimize timeouts
# ---------------------------
_PROHIBITED_CLICK_PAT = re.compile(r"(place\s*order|apple\s*pay|google\s*pay|\bpay\b|\bcheckout\b)", re.I)

async def _exists_quick(page, sel: str) -> bool:
    try:
        return (await page.locator(sel).count()) > 0
    except Exception:
        return False

async def _soft_wait_for(page, sel: str, state: str = "visible", ms: int = 1200) -> bool:
    """Polling loop (<=ms) that avoids raising timeouts; returns True if condition met."""
    end = time.time() + max(200, ms) / 1000.0
    interval = 0.15
    try:
        loc = page.locator(sel)
        while time.time() < end:
            cnt = await loc.count()
            if cnt > 0:
                if state == "visible":
                    try:
                        if await loc.first.is_visible():
                            return True
                    except Exception:
                        pass
                elif state in {"attached","hidden","detached"}:
                    if state == "attached" and cnt > 0:
                        return True
                    if state in {"hidden","detached"} and cnt == 0:
                        return True
                else:
                    return True
            await page.wait_for_timeout(int(interval * 1000))
    except Exception:
        pass
    return False

async def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 220):
    """Force JSON if supported; otherwise fall back."""
    try:
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens
        )
    except Exception:
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens
        )

def _parse_json_or_fallback(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        raise

# ---------------------------
# 4) Agent loop (DOM/Persona/History digests; task LAST)
# ---------------------------
async def act_with_llm(page, persona: Dict[str, Any]) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    step = 0

    while True:
        dom = (await page.content())[:MAX_DOM_CHARS]
        dom_digest = digest_dom(dom)
        persona_line = digest_persona(persona)
        hist_digest = digest_history(history)

        messages = [
            {"role": "system", "content": COMPACT_SYSTEM},
            {"role": "user", "content":
                "DOM (digest):\n" + dom_digest +
                "\n\nPersona:\n" + persona_line +
                "\n\nRecent History (digest):\n" + hist_digest +
                "\n\nTASK (choose exactly ONE):\n"
                "- Decide the next minimal action towards pre-checkout, following rules.\n"
                "- Output ONLY the JSON object with required fields.\n"
            }
        ]

        try:
            resp = await _call_llm(messages)
            raw = resp.choices[0].message.content
            cmd = _parse_json_or_fallback(raw)
        except Exception as e:
            history.append({"error": f"parse-fail: {repr(e)}"})
            break

        act = cmd.get("action")
        sel = (cmd.get("selector") or "").strip()
        txt = cmd.get("text", "")
        ms  = int(cmd.get("ms", 800))
        state = (cmd.get("state") or "visible").lower()
        step += 1

        try:
            if act == "click":
                # Pre-check + guard against finalization
                if _PROHIBITED_CLICK_PAT.search(sel) or _PROHIBITED_CLICK_PAT.search(txt or ""):
                    history.append({"error": f"blocked_click @ {sel or txt}", "step": step})
                    break
                if not await _exists_quick(page, sel):
                    history.append({"warn": f"selector_missing {sel}", "step": step})
                else:
                    await page.click(sel, timeout=4000)

            elif act == "type":
                if not await _exists_quick(page, sel):
                    history.append({"warn": f"selector_missing {sel}", "step": step})
                else:
                    await page.fill(sel, txt, timeout=4000)

            elif act in ("wait", "wait_ms"):
                if ms == 500:
                    ms = random.choice([350, 700])
                await page.wait_for_timeout(ms)

            elif act == "wait_for":
                ok = await _soft_wait_for(page, sel, state=state, ms=min(ms, 1500))
                history.append({"info": f"soft_wait_{'ok' if ok else 'miss'}", "selector": sel, "state": state, "ms": ms, "step": step})

            elif act == "note":
                # Human-centered problem or milestone
                pass

            else:
                history.append({"error": f"unknown action {act}", "step": step})
                break

            cmd["step"] = step
            history.append(cmd)

        except PWTimeout:
            # Soften timeouts: record & continue to avoid making every CI a timeout
            history.append({"warn": f"timeout @ {sel}", "step": step})
        except Exception as e:
            history.append({"error": f"action-fail @ {sel}: {repr(e)}", "step": step})

        # Stop when pre-checkout is reached (no actual order)
        content_lc = (await page.content()).lower()
        if any(k in content_lc for k in ["your cart", "review order", "review your order", "cart subtotal"]):
            history.append({"info": "stop_precheckout", "step": step})
            break

        if len(history) >= MAX_STEPS:
            history.append({"info": "max-steps-reached", "step": step})
            break

    return {"history": history}

# ---------------------------
# 5) Robust navigation + engine fallback
# ---------------------------
async def run_one(play, persona: Dict[str, Any], engine: str = "webkit", headful: bool = False) -> Dict[str, Any]:
    """
    Try chosen engine first, then fall back to Chromium.
    - Wait until domcontentloaded (avoid load-event stalls)
    - Locale/timezone/geolocation + permission for stability
    """
    async def _run_with(browser_type):
        browser = await browser_type.launch(headless=not headful)
        try:
            # iPhone 15 preset if available
            try:
                device = play.devices["iPhone 15"]
            except KeyError:
                device = {
                    "viewport": {"width": 393, "height": 852},
                    "user_agent": (
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                        "Mobile/15E148 Safari/604.1"
                    ),
                    "isMobile": True, "hasTouch": True,
                }

            context = await browser.new_context(
                **device,
                locale="en-US",
                timezone_id="America/New_York",
                geolocation={"latitude": 40.7128, "longitude": -74.0060},  # NYC
                permissions=["geolocation"],
            )
            page = await context.new_page()

            # Navigate: domcontentloaded, then short stabilization
            await page.goto("https://www.ubereats.com", wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass

            return page, context, browser
        except Exception:
            await browser.close()
            raise

    # First try user-chosen engine
    browser_type = getattr(play, engine)
    try:
        page, context, browser = await _run_with(browser_type)
    except Exception:
        # Fallback to Chromium
        page, context, browser = await _run_with(play.chromium)

    try:
        result = await act_with_llm(page, persona)
    finally:
        await browser.close()
    return result

# ---------------------------
# 6) Scoring (1–5; no-fabrication; milestones)
# ---------------------------
def _rng_for_persona(persona: Dict[str, Any]) -> random.Random:
    pid = persona.get("id") or json.dumps(persona, sort_keys=True)
    seed = int(hashlib.sha256(pid.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)

def _signals(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = [h for h in history if "error" in h]
    warns  = [h for h in history if "warn" in h]  # soft signals incl. timeouts
    timeouts = [h for h in warns if "timeout" in h.get("warn", "")]
    steps = sum(1 for h in history if "action" in h)
    waits = [h for h in history if h.get("action") in ("wait","wait_ms")]
    long_waits = sum(1 for w in waits if int(w.get("ms", 0)) >= 1500)

    notes = [h for h in history if h.get("action") == "note"]
    severe_tags = {"diet_mismatch","allergen_missing","filter_missing","aria_missing","contrast_low","tiny_tap_target"}
    severe_notes = [n for n in notes if (n.get("tag") in severe_tags)]

    milestones = {n.get("tag") for n in notes if str(n.get("tag","")).startswith("milestone_")}
    m_item   = "milestone_item_added" in milestones
    m_cart   = "milestone_cart_open" in milestones
    m_review = "milestone_review"    in milestones
    prechk_stop = any(h.get("info") == "stop_precheckout" for h in history)

    return {
        "errors": len(errors),
        "warns": len(warns),
        "timeouts": len(timeouts),
        "steps": steps,
        "long_waits": long_waits,
        "notes_severe": len(severe_notes),
        "m_item": m_item, "m_cart": m_cart, "m_review": m_review,
        "prechk_stop": prechk_stop,
    }

def _score_from_signals(sig: Dict[str, Any], rng: random.Random) -> int:
    # Perfect happy path → 5
    if sig["m_review"] and sig["notes_severe"] == 0 and sig["errors"] == 0 and sig["timeouts"] == 0 and sig["long_waits"] == 0:
        return 5

    # Base by milestone progress
    if sig["m_review"]:
        base = 4
    elif sig["m_cart"] and sig["m_item"]:
        base = 4
    elif sig["m_item"]:
        base = 3
    else:
        base = 2 if (sig["errors"] or sig["steps"] > 25) else 3

    # Penalize severe human-centered notes
    if sig["notes_severe"] >= 2:
        base -= 2
    elif sig["notes_severe"] == 1:
        base -= 1

    # Soft penalties
    if sig["timeouts"] >= 2:
        base -= 1
    if sig["long_waits"] >= 2:
        base -= 1

    # Deterministic jitter (±1) to avoid clustering
    jitter = rng.choice([-1, 0, 0, 0, +1])
    return max(1, min(5, base + jitter))

# ---------------------------
# 7) Analysis/report (English; allow "None observed")
# ---------------------------
def _fallback_description(p: Dict[str, Any]) -> str:
    return (f"{p.get('age','?')}yo in {p.get('location','?')} "
            f"({p.get('income','?')}; {p.get('diet','none')}/{p.get('accessibility','none')}); "
            f"goal: {p.get('goal','n/a')}")

async def analyze_and_save(root: pathlib.Path, persona: Dict[str, Any], run: Dict[str, Any], pid: str):
    try:
        analysis_resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user",   "content": json.dumps({"persona": persona, **run}, ensure_ascii=False)}
            ],
            temperature=0.55
        )
        aobj = json.loads(analysis_resp.choices[0].message.content)
    except Exception:
        aobj = {}

    desc = aobj.get("description")
    if not isinstance(desc, str) or not desc.strip():
        desc = _fallback_description(persona)

    md = aobj.get("markdown")
    if not isinstance(md, str) or not md.strip():
        notes = [h for h in run.get("history", []) if h.get("action") == "note" and not str(h.get("tag","")).startswith("milestone_")]
        if notes:
            ci = notes[0]
            ci_txt = f"step {ci.get('step')}: {ci.get('tag')} — {ci.get('detail')}"
        else:
            ci_txt = "None observed."
        md = (
            "## Persona\n"
            f"{desc}\n\n"
            "## Critical Issues\n"
            f"- {ci_txt}\n\n"
            "## Minor Friction\n"
            "- A few extra taps or unclear labels while narrowing choices.\n\n"
            "## Suggested Improvements\n"
            "- Clear vegan/allergen badges and dedicated filters on list & item pages.\n"
        )

    sig = _signals(run.get("history", []))
    rng = _rng_for_persona(persona)
    score_int = _score_from_signals(sig, rng)

    sess = root / pid
    sess.mkdir(parents=True, exist_ok=True)

    (sess / "issues.md").write_text(
        f"{score_int:.1f} / 5.0\n"
        f"**Description:** {desc}\n\n"
        f"{md}",
        encoding="utf-8"
    )

    with open(sess / "issues.json", "w", encoding="utf-8") as f:
        json.dump({
            "persona": persona,
            "run": run,
            "analysis": md,
            "score": float(f"{score_int:.1f}"),
            "description": desc,
            "signals": sig
        }, f, ensure_ascii=False, indent=2)

# ---------------------------
# 8) Main
# ---------------------------
async def main(args):
    personas = json.load(open(args.personas, encoding="utf-8"))
    root     = pathlib.Path(args.output); root.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        for idx, persona in enumerate(personas, 1):
            pid  = persona.get("id") or f"P-{idx:02}"
            sess = root / pid
            if (sess / "issues.json").exists():
                print(f"{pid} ✔︎ Skip"); continue
            print(f"▶ {pid}")

            result = await run_one(p, persona, engine=args.engine, headful=args.headful)
            await analyze_and_save(root, persona, result, pid)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True)
    ap.add_argument("--output",   required=True)
    ap.add_argument("--engine", choices=["webkit","chromium","firefox"], default="webkit")
    ap.add_argument("--headful", action="store_true", help="Run with a visible browser window")
    asyncio.run(main(ap.parse_args()))
