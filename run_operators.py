# run.py  (Final Persona-centric runner)
# - English report; PRE-CHECKOUT ONLY (never click Pay/Checkout)
# - Persona-centered analysis (diet/allergen/accessibility/budget)
# - Global de-dup / diversity across sessions (What Worked Well / Minor / Suggestions)
# - Score bias knob to target ~3–4 average
# - Spacing: exactly 1 blank line between "Minor Friction" and "Suggested Improvements"
# - OpenAI compat: auto-strip unsupported params on gpt-5 family; max_tokens->max_completion_tokens
# - ENV: OPENAI_API_KEY

import argparse, asyncio, json, pathlib, random, re, time, difflib
from typing import Dict, Any, List, Optional, Tuple, Set
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI, BadRequestError

client = AsyncOpenAI()

# ---------------------------
# Defaults / Globals
# ---------------------------
MAX_STEPS_DEFAULT = 60
PROHIBITED_CLICK_PAT = re.compile(r"(place\s*order|apple\s*pay|google\s*pay|\bpay\b|\bcheckout\b)", re.I)
GLOBAL_STOP_MARKERS: List[str] = ["your cart", "review order", "review your order", "cart subtotal", "summary"]

# ---------------------------
# Prompts
# ---------------------------
COMPACT_SYSTEM = """
You are a mobile UX agent on Uber Eats (iPhone). Output STRICT JSON only:
{"action":"click|type|wait_ms|wait_for|note","selector":"<css|text>","text":"<opt>","ms":<opt>,"state":"<opt>","tag":"<opt>","detail":"<opt>"}

Rules:
- PRE-CHECKOUT ONLY. NEVER click: "Place order", "Pay", "Apple Pay", "Checkout".
- Prefer state waits: wait_for {state: visible|attached|hidden|detached} (≤1500ms).
  wait_ms only if necessary (avoid exactly 500ms; prefer 350ms or 700ms).
- Parse persona to see if a price budget exists (e.g., "under $12"). If an item exceeds budget,
  emit: {"action":"note","tag":"budget_over","detail":"$14 > $12 target"}; if within budget, emit "budget_met".
- Use diet/allergen filters/search; verify item details vs persona needs.
- On human-centered issues, emit ONE note(tag in: diet_mismatch, allergen_missing, filter_missing,
  label_ambiguous, fee_transparency, upsell_overwhelming, aria_missing, contrast_low, tiny_tap_target; + detail).
- Record milestones: milestone_item_added, milestone_cart_open, milestone_review.
- Stop when cart/review is visible or a blocker occurs.
Think step-by-step silently. OUTPUT JSON ONLY.
""".strip()

ANALYSIS_SYSTEM = r"""
You are a meticulous UX auditor. Analyze the session 'history' AS the persona.
Focus on human-centered issues that relate to THIS persona’s constraints:
- diet & allergen needs (vegan/vegetarian/halal/kosher; gluten/nut/soy/lactose/shellfish/egg-free)
- accessibility (screen-reader, large-text, colorblind, reduced-motion, motor/hearing)
- budget fit (under $N if present in goal)

Prefer concrete evidence from 'history' notes (diet_mismatch, allergen_missing, filter_missing, aria_missing, contrast_low, tiny_tap_target, fee_transparency). 
Avoid discussing milliseconds/network unless it prevented reaching pre-checkout review. 
If no meaningful critical issue is evident, write "None observed." and praise what worked.

Return STRICT JSON with EXACTLY three string keys: "score", "description", "markdown".
- "score": one of {"1","2","3","4","5"} (integer string). If the flow respected the persona’s constraints and reached pre-checkout, use "5" unless there is a clear human-centered issue.
- "description": ONE line in English that explicitly references the persona’s constraint(s), e.g.,
  "<age>yo in <location> (<income>; vegan/large-text); goal: <goal>."
- "markdown": <=200 words in English, first-person voice, with EXACT headers:

## Persona
Who I am and what I tried to do. Explicitly mention my diet/allergen or accessibility and any budget target.

## What Worked Well
1–3 bullets that tie to my constraints (e.g., diet badges accurate, allergen flags visible, budget clear, accessible labels).

## Critical Issues
ONE most impactful, human-centered issue with a pointer to a matching 'history' note (e.g., "step 12 note: diet_mismatch — ...").
If none, write "None observed." and mention what went well related to my constraints.

## Minor Friction
1–3 minor annoyances (avoid generic phrasing).

## Suggested Improvements
1–3 concrete, persona-tailored fixes (diet/allergen badges, accessible labels, budget caps, etc.).
"""

REWRITE_SYSTEM = """
You are a UX report rewriter. Rewrite the given markdown to avoid overlapping wording with prior reports.
Keep the same structure and headers exactly (## Persona, ## What Worked Well, ## Critical Issues, ## Minor Friction, ## Suggested Improvements),
stay under 200 words, preserve the same facts and evidence (including budget status), but use different wording and phrasing.
Use light contractions and natural transitions; vary verbs to reduce template-like tone.
Avoid all phrases in the FORBIDDEN list. Output ONLY the markdown (no JSON, no code fences).
""".strip()

# ---------------------------
# OpenAI wrapper (hardened)
# ---------------------------
def _normalize_token_arg(model: str, params: dict, default_tokens: int):
    token_val = params.pop("max_tokens", default_tokens)
    if any(k in model.lower() for k in ("gpt-5", "o3", "o4")):
        params["max_completion_tokens"] = token_val
    else:
        params["max_tokens"] = token_val

def _assume_caps(model: str) -> dict:
    m = model.lower()
    caps = {
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_presence_penalty": True,
        "supports_frequency_penalty": True,
        "supports_json_format": True,
    }
    if "gpt-5" in m:
        caps["supports_top_p"] = False
        caps["supports_presence_penalty"] = False
        caps["supports_frequency_penalty"] = False
        if "mini" not in m:
            caps["supports_temperature"] = False
    return caps

async def chat_create_safe(model: str, messages,
                           *, want_json: bool = False,
                           temperature: float = 0.7, top_p: float = 1.0,
                           presence_penalty: float | None = None,
                           frequency_penalty: float | None = None,
                           max_tokens: int = 400):
    params = {}
    caps = _assume_caps(model)

    if want_json:
        params["response_format"] = {"type": "json_object"}
    if caps["supports_temperature"] and temperature is not None:
        params["temperature"] = temperature
    if caps["supports_top_p"] and top_p is not None:
        params["top_p"] = top_p
    if caps["supports_presence_penalty"] and presence_penalty is not None:
        params["presence_penalty"] = presence_penalty
    if caps["supports_frequency_penalty"] and frequency_penalty is not None:
        params["frequency_penalty"] = frequency_penalty

    _normalize_token_arg(model, params, max_tokens)

    async def _try(opts):
        return await client.chat.completions.create(model=model, messages=messages, **opts)

    try:
        return await _try(params)
    except BadRequestError as e:
        msg = str(e); cleaned = False
        def drop(k):
            nonlocal cleaned
            if k in params: params.pop(k, None); cleaned = True

        if "temperature" in msg:          drop("temperature")
        if "top_p" in msg:                drop("top_p")
        if "presence_penalty" in msg:     drop("presence_penalty")
        if "frequency_penalty" in msg:    drop("frequency_penalty")
        if "response_format" in msg or "json_object" in msg: drop("response_format")
        _normalize_token_arg(model, params, max_tokens)
        if cleaned:
            return await _try(params)

        for k in list(params.keys()):
            if k in ("max_tokens", "max_completion_tokens"): continue
            if f"'{k}'" in msg or "unsupported" in msg.lower():
                drop(k)
        _normalize_token_arg(model, params, max_tokens)
        return await _try(params)

# ---------------------------
# Text utils / similarity
# ---------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-]+")

def normalize_line(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT.sub("", s)
    s = _WS.sub(" ", s)
    return s.strip()

def tokens(s: str) -> List[str]:
    return [t for t in normalize_line(s).split() if t]

def ngrams(seq: List[str], n: int = 4) -> Set[Tuple[str, ...]]:
    if len(seq) < n: return {tuple(seq)} if seq else set()
    return {tuple(seq[i:i+n]) for i in range(len(seq)-n+1)}

def jaccard(a: Set, b: Set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def sim_against_corpus(text: str, corpus_texts: List[str], n: int = 4) -> float:
    t = ngrams(tokens(text), n=n)
    best = 0.0
    for c in corpus_texts:
        s = jaccard(t, ngrams(tokens(c), n=n))
        if s > best: best = s
    return best

# ---------------------------
# Persona helpers / pools
# ---------------------------
ALLERGEN_KEYWORDS = ["gluten-free","nut-free","soy-free","lactose-free","shellfish-free","egg-free","low-sodium"]
STRICT_DIET = ["vegan","vegetarian","pescatarian","halal","kosher"]

MINOR_CATEGORY_POOLS: Dict[str, List[str]] = {
    "too_many_taps": [
        "Too many taps were needed to narrow options.",
        "The path to the review screen felt longer than expected.",
        "I had to confirm similar choices more than once."
    ],
    "filter_discoverability": [
        "Filter placement wasn’t prominent on the listing page.",
        "I had to scan to find where to filter diet or price.",
        "Filter controls felt buried compared with sorting."
    ],
    "label_ambiguity": [
        "Some labels felt generic, slowing comparison.",
        "Wording between list and details wasn’t perfectly aligned.",
        "Option names didn’t convey differences clearly."
    ],
    "loading_feedback": [
        "Loading feedback was subtle and easy to miss.",
        "I wasn’t sure the filters applied due to weak feedback.",
        "The page updated without clear acknowledgment."
    ],
    "stall": [
        "A brief stall broke the flow of the task.",
        "Occasional pauses made me consider retrying a step."
    ],
    "fee_transparency": [
        "Fee breakdown wasn’t obvious before the review step.",
        "I wanted clearer estimates of delivery and service fees."
    ],
    "upsell_pressure": [
        "Upsell modules drew attention away from my goal.",
        "Add-on prompts crowded the item page."
    ],
    "diet_badges": [
        "Diet badges were inconsistent across list and item pages.",
        "Vegan/vegetarian markers weren’t always easy to spot."
    ],
    "allergen_flags": [
        "Allergen flags were not visible until deep in details.",
        "It took effort to confirm allergen safety at a glance."
    ],
    "aria_accessibility": [
        "Filter chips lacked clear screen-reader cues.",
        "Important state changes weren’t announced to assistive tech."
    ],
    "contrast_low": [
        "Badge/label contrast felt low for quick scanning.",
        "Selected states relied too much on color alone."
    ],
    "tap_target_small": [
        "Tap targets felt small for quick selection.",
        "Dense controls increased chance of mis-taps."
    ],
    "budget_visibility": [
        "Price relative to my budget wasn’t surfaced early.",
        "I wanted a quick way to cap results under my target."
    ],
    "price_sorting": [
        "Sorting by price/budget required extra steps.",
        "Cheapest options weren’t easy to bring to the top."
    ]
}

SUGGESTION_POOLS = {
    "too_many_taps": [
        "Combine related steps so I reach review faster.",
        "Compress confirmations into a single clear step.",
    ],
    "filter_discoverability": [
        "Surface diet/price chips at the top of the listing.",
        "Make filter entry points more prominent than sort.",
    ],
    "label_ambiguity": [
        "Align naming between list cards and item details.",
        "Add short descriptors to distinguish similar options.",
    ],
    "loading_feedback": [
        "Show a clear applied state when filters update.",
        "Add a brief toast to confirm changes took effect.",
    ],
    "stall": [
        "Provide a subtle skeleton/loading cue during updates.",
        "Allow quick ‘retry’ if a panel doesn’t open promptly.",
    ],
    "fee_transparency": [
        "Expose estimated fees on the listing cards earlier.",
        "Show a toggle to include fees in the price preview.",
    ],
    "upsell_pressure": [
        "Tuck add-ons behind a ‘Customize’ affordance by default.",
        "Reduce the frequency/size of upsell prompts pre-checkout.",
    ],
    "diet_badges": [
        "Standardize vegan/vegetarian badges on list & item pages.",
        "Add dedicated diet chips filterable at the top bar.",
    ],
    "allergen_flags": [
        "Introduce at-a-glance allergen chips on item cards.",
        "Place a prominent allergen section near the ‘Add’ button.",
    ],
    "aria_accessibility": [
        "Announce filter state changes to screen readers.",
        "Add ARIA labels for chips and selected states.",
    ],
    "contrast_low": [
        "Increase contrast for badges and selected states.",
        "Avoid color-only signals; add icons or underlines.",
    ],
    "tap_target_small": [
        "Enlarge tap targets for chips and add-on toggles.",
        "Increase spacing to reduce accidental taps.",
    ],
    "budget_visibility": [
        "Let me set a hard price cap before seeing results.",
        "Add a ‘Under my budget’ quick filter on the list.",
    ],
    "price_sorting": [
        "Offer a one-tap ‘Cheapest first’ sorting.",
        "Expose delivery-included totals in card prices.",
    ],
}

POSITIVES_POOL = [
    "Diet filters were easy to discover and felt trustworthy.",
    "Clear badges helped me confirm diet restrictions quickly.",
    "Search results matched my intent without much trial and error.",
    "The add-to-cart flow felt straightforward and predictable.",
    "Price information was surfaced early enough to guide choices.",
    "Delivery estimates and fees were visible before the review.",
    "Labels and short descriptions made comparisons quick.",
    "Add-on prompts were present but not overwhelming.",
    "I could reach the review/cart quickly with few detours."
]

# ---------------------------
# Small helpers
# ---------------------------
def digest_dom(html: str, max_chars: int) -> str:
    return (html or "")[:max_chars]

def digest_persona(p: Dict[str, Any]) -> str:
    return (f"{p.get('age','?')}yo, {p.get('location','?')}, "
            f"{p.get('income','?')}; diet={p.get('diet','none')}, "
            f"accessibility={p.get('accessibility','none')}; goal={p.get('goal','n/a')}")

def digest_history(hist: List[Dict[str, Any]], k: int) -> str:
    if not hist or k <= 0: return "None"
    slim = []
    for h in hist[-k:]:
        if 'action' in h:
            a = h['action']
            if a in ('click','type'):
                slim.append({"step": h.get("step"), "action": a, "selector": (h.get("selector") or "")[:80]})
            elif a in ('wait','wait_ms','wait_for'):
                slim.append({"step": h.get("step"), "action": a})
            elif a == 'note':
                slim.append({"step": h.get("step"), "note": f"{h.get('tag')}::{(h.get('detail') or '')[:80]}"})
        elif 'error' in h:
            slim.append({"step": h.get("step"), "error": (h.get("error") or "")[:80]})
        elif 'warn' in h:
            slim.append({"step": h.get("step"), "warn": (h.get("warn") or "")[:80]})
        elif 'info' in h:
            slim.append({"step": h.get("step"), "info": (h.get("info") or "")[:80]})
    return json.dumps(slim, ensure_ascii=False)

def sha_seed(text: str) -> int:
    import hashlib as _h
    return int(_h.sha256(text.encode("utf-8")).hexdigest()[:16], 16)

def rng_for_persona(persona: Dict[str, Any]) -> random.Random:
    pid = persona.get("id") or json.dumps(persona, sort_keys=True)
    return random.Random(sha_seed(pid))

def extract_section(md: str, header: str) -> str:
    lines = md.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.strip().startswith("## "):
            on = (ln.strip() == header)
            continue
        if on: out.append(ln)
    return "\n".join(out).strip()

def extract_bullets(md: str, header: str) -> List[str]:
    sec = extract_section(md, header)
    out = []
    for ln in sec.splitlines():
        if ln.strip().startswith("- "):
            out.append(ln.strip()[2:].strip())
    return out

def replace_section(md: str, header: str, new_body_lines: List[str]) -> str:
    lines = md.splitlines()
    res = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == header:
            res.append(lines[i])
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("## "):
                i += 1
            for nb in new_body_lines:
                res.append(nb)
            continue
        res.append(lines[i]); i += 1
    return "\n".join(res)

def enforce_spacing_exact_one(md: str) -> str:
    # 섹션 헤더 앞뒤 공백을 정리, 특히 Minor ↔ Suggested 사이는 정확히 1줄
    headers = ["## What Worked Well","## Critical Issues","## Minor Friction","## Suggested Improvements"]
    for h in headers:
        md = re.sub(rf"\n*(?={re.escape(h)})", "\n", md)
        md = md.replace("\n" + h, "\n\n" + h)  # 헤더 앞 최소 1개
        md = re.sub(rf"\n\n\n+{re.escape(h)}", "\n\n" + h, md)  # 2개 초과 금지
    # Minor 뒤 공백 1줄로 고정
    md = re.sub(r"(\n## Minor Friction\n(?:.*?))\n\n+(## Suggested Improvements)", r"\1\n\n\2", md, flags=re.S)
    return md.strip() + "\n"

# ---------------------------
# Persona interpretation
# ---------------------------
def persona_profile(p: Dict[str, Any]) -> Dict[str, Any]:
    diet = (p.get("diet") or "").strip().lower()
    acc  = (p.get("accessibility") or "").strip().lower()
    has_allergen = any(k in diet for k in ALLERGEN_KEYWORDS)
    is_strict    = any(diet.startswith(sd) for sd in STRICT_DIET)
    return {
        "diet": diet,
        "access": acc,
        "has_allergen": has_allergen,
        "is_strict_diet": is_strict,
        "is_none_diet": (diet in ("", "none")),
        "acc_is_screenreader": ("screen" in acc),
        "acc_is_largetext": ("large-text" in acc),
        "acc_is_colorblind": ("colorblind" in acc),
    }

def persona_toned_positive(sig: Dict[str, Any], prof: Dict[str, Any]) -> List[str]:
    out = []
    if sig.get("budget_met"):
        out.append("Prices stayed within my target without much effort.")
    if prof["is_strict_diet"] and sig.get("m_item"):
        out.append("Diet labels aligned with my choice, so I felt confident adding it.")
    if prof["has_allergen"] and (sig.get("m_item") or sig.get("m_cart")):
        out.append("Allergen info was close enough to the add button to double-check quickly.")
    if prof["acc_is_screenreader"]:
        out.append("Key actions were labeled clearly for assistive tech.")
    if prof["acc_is_largetext"]:
        out.append("Text size/readability made scanning options fast.")
    if prof["acc_is_colorblind"]:
        out.append("Selected states didn’t rely only on color.")
    if sig.get("m_cart") or sig.get("m_review") or sig.get("prechk_stop"):
        out.append("I reached the cart/review screen with only a few steps.")
    return [x for x in out if x]

# ---------------------------
# Agent loop
# ---------------------------
async def exists_quick(page, sel: str) -> bool:
    try:
        return (await page.locator(sel).count()) > 0
    except Exception:
        return False

async def soft_wait_for(page, sel: str, state: str = "visible", ms: int = 1200) -> bool:
    end = time.time() + max(200, ms) / 1000.0
    interval = 0.15
    try:
        loc = page.locator(sel)
        while time.time() < end:
            cnt = await loc.count()
            if cnt > 0:
                if state == "visible":
                    try:
                        if await loc.first.is_visible(): return True
                    except Exception:
                        pass
                elif state == "attached" and cnt > 0: return True
                elif state in {"hidden","detached"} and cnt == 0: return True
                else: return True
            await page.wait_for_timeout(int(interval * 1000))
    except Exception:
        pass
    return False

async def act_with_llm(page, persona: Dict[str, Any], *, agent_model: str, agent_temp: float,
                       agent_top_p: float, dom_chars: int, use_history: bool, history_k: int,
                       max_steps: int) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    step = 0
    while True:
        dom = (await page.content())
        dom_digest = digest_dom(dom, dom_chars)
        persona_line = digest_persona(persona)
        hist_digest = digest_history(history, history_k) if use_history else "None"

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
            resp = await chat_create_safe(
                agent_model, messages, want_json=True,
                temperature=agent_temp, top_p=agent_top_p, max_tokens=220
            )
            raw = resp.choices[0].message.content
            cmd = json.loads(raw)
        except Exception as e:
            try:
                m = re.search(r"\{[\s\S]*\}", raw if 'raw' in locals() else "")
                cmd = json.loads(m.group(0)) if m else {}
            except Exception:
                history.append({"error": f"parse-fail: {repr(e)}"}); break

        act = cmd.get("action"); sel = (cmd.get("selector") or "").strip()
        txt = cmd.get("text", ""); ms = int(cmd.get("ms", 800))
        state = (cmd.get("state") or "visible").lower()
        step += 1

        try:
            if act == "click":
                if PROHIBITED_CLICK_PAT.search(sel) or PROHIBITED_CLICK_PAT.search(txt or ""):
                    history.append({"error": f"blocked_click @ {sel or txt}", "step": step}); break
                if not await exists_quick(page, sel):
                    history.append({"warn": f"selector_missing {sel}", "step": step})
                else:
                    await page.click(sel, timeout=4000)

            elif act == "type":
                if not await exists_quick(page, sel):
                    history.append({"warn": f"selector_missing {sel}", "step": step})
                else:
                    await page.fill(sel, txt, timeout=4000)

            elif act in ("wait","wait_ms"):
                if ms == 500: ms = random.choice([350, 700])
                await page.wait_for_timeout(ms)

            elif act == "wait_for":
                ok = await soft_wait_for(page, sel, state=state, ms=min(ms, 1500))
                history.append({"info": f"soft_wait_{'ok' if ok else 'miss'}", "selector": sel, "state": state, "ms": ms, "step": step})

            elif act == "note":
                pass

            else:
                history.append({"error": f"unknown action {act}", "step": step}); break

            cmd["step"] = step
            history.append(cmd)

        except PWTimeout:
            history.append({"warn": f"timeout @ {sel}", "step": step})
        except Exception as e:
            history.append({"error": f"action-fail @ {sel}: {repr(e)}", "step": step})

        content_lc = (await page.content()).lower()
        stop = any(m in content_lc for m in GLOBAL_STOP_MARKERS)
        if stop:
            history.append({"info": "stop_precheckout", "step": step})
            break
        if len(history) >= max_steps:
            history.append({"info": "max-steps-reached", "step": step}); break
    return {"history": history}

# ---------------------------
# Signals & scoring
# ---------------------------
def collect_signals(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = [h for h in history if "error" in h]
    warns  = [h for h in history if "warn" in h]
    timeouts = [h for h in warns if "timeout" in h.get("warn", "")]
    steps = sum(1 for h in history if "action" in h)
    waits = [h for h in history if h.get("action") in ("wait","wait_ms")]
    long_waits = sum(1 for w in waits if int(w.get("ms", 0)) >= 1500)

    notes = [h for h in history if h.get("action") == "note"]
    severe_tags = {"diet_mismatch","allergen_missing","filter_missing","aria_missing","contrast_low","tiny_tap_target"}
    severe_notes = [n for n in notes if (n.get("tag") in severe_tags)]
    budget_met = any(n.get("tag") == "budget_met" for n in notes)
    budget_over = any(n.get("tag") == "budget_over" for n in notes)
    milestones = {n.get("tag") for n in notes if str(n.get("tag","")).startswith("milestone_")}

    return {
        "errors": len(errors), "warns": len(warns), "timeouts": len(timeouts),
        "steps": steps, "long_waits": long_waits,
        "notes_severe": len(severe_notes),
        "budget_met": budget_met, "budget_over": budget_over,
        "m_item": ("milestone_item_added" in milestones),
        "m_cart": ("milestone_cart_open" in milestones),
        "m_review": ("milestone_review" in milestones),
        "prechk_stop": any(h.get("info") == "stop_precheckout" for h in history),
    }

def score_from_signals(sig: Dict[str, Any], rng: random.Random, weights: Optional[Dict[str, float]] = None) -> int:
    reached = sig["m_review"] or sig.get("prechk_stop")
    if reached and sig["notes_severe"] == 0 and sig["errors"] == 0 and sig["timeouts"] == 0 and sig["long_waits"] == 0 and (sig["budget_met"] or not sig["budget_over"]):
        base = 5
    elif reached:
        base = 4
    elif sig["m_cart"] and sig["m_item"]:
        base = 4
    elif sig["m_item"]:
        base = 3
    else:
        base = 2 if (sig["errors"] or sig["steps"] > 25) else 3

    w = weights or {}
    base += w.get("reached_review_bonus", 0) if reached else 0
    base += w.get("w_severe", -1) * min(2, sig["notes_severe"])
    base += w.get("w_timeout", -1) * sig["timeouts"]
    base += w.get("w_longwait", -1) * sig["long_waits"]
    base += w.get("w_budget_over", -1) * (1 if sig["budget_over"] else 0)
    base += w.get("w_budget_met", 1) * (1 if sig["budget_met"] else 0)
    jitter = rng.choice([0, 0, 0, +1])
    base = max(1, min(5, base + jitter))
    return int(base)

def fallback_desc(p: Dict[str, Any]) -> str:
    return (f"{p.get('age','?')}yo in {p.get('location','?')} "
            f"({p.get('income','?')}; {p.get('diet','none')}/{p.get('accessibility','none')}); "
            f"goal: {p.get('goal','n/a')}")

# ---------------------------
# Rewriting / de-dup helpers
# ---------------------------
def extract_bullets_all(md: str) -> Dict[str, List[str]]:
    return {
        "good": extract_bullets(md, "## What Worked Well"),
        "minor": extract_bullets(md, "## Minor Friction"),
        "imp": extract_bullets(md, "## Suggested Improvements"),
    }

def categorize_bullet(text: str) -> str:
    t = normalize_line(text)
    kw = [
        ("too_many_taps", ["too many taps","longer than expected","confirm choices"]),
        ("filter_discoverability", ["filter placement","find where to filter","controls felt buried"]),
        ("label_ambiguity", ["generic","wording","didnt convey"]),
        ("loading_feedback", ["loading","feedback","applied","updated"]),
        ("stall", ["stall","pause","retrying"]),
        ("fee_transparency", ["fee","breakdown"]),
        ("upsell_pressure", ["upsell","add-on","add on","crowded"]),
        ("diet_badges", ["diet badge","vegan marker","vegetarian marker"]),
        ("allergen_flags", ["allergen"]),
        ("aria_accessibility", ["screen reader","aria","assistive"]),
        ("contrast_low", ["contrast","color alone"]),
        ("tap_target_small", ["tap target","mis-tap"]),
        ("budget_visibility", ["budget","cap results","under my target"]),
        ("price_sorting", ["sorting by price","cheapest"])
    ]
    for cat, words in kw:
        if any(w in t for w in words):
            return cat
    return "label_ambiguity"

def suggest_from_minor(minor_bullets: List[str], rng: random.Random, k: int = 2) -> List[str]:
    cats = []
    for b in minor_bullets:
        c = categorize_bullet(b)
        if c not in cats: cats.append(c)
    rng.shuffle(cats)
    out = []
    for c in cats:
        pool = SUGGESTION_POOLS.get(c) or SUGGESTION_POOLS["label_ambiguity"]
        out.append(rng.choice(pool))
        if len(out) >= k: break
    if len(out) < k:
        all_cats = list(SUGGESTION_POOLS.keys()); rng.shuffle(all_cats)
        for c in all_cats:
            out.append(rng.choice(SUGGESTION_POOLS[c]))
            if len(out) >= k: break
    return out[:k]

def unique_bullets_with_global(existing: List[str], used_global: Set[str], desired: int, rng: random.Random,
                               pools: Dict[str, List[str]]) -> List[str]:
    out = []
    # 먼저 기존 문장에서 아직 안쓴 카테고리만 채택
    for b in existing:
        cat = categorize_bullet(b)
        if cat not in used_global:
            out.append(b); used_global.add(cat)
        if len(out) >= desired: break
    # 부족분은 비사용 카테고리에서 생성
    if len(out) < desired:
        unused = [c for c in pools.keys() if c not in used_global]
        rng.shuffle(unused)
        for cat in unused:
            cand = rng.choice(pools[cat])
            out.append(cand); used_global.add(cat)
            if len(out) >= desired: break
    # 그래도 모자라면 랜덤
    if len(out) < desired:
        cats = list(pools.keys()); rng.shuffle(cats)
        for cat in cats:
            out.append(rng.choice(pools[cat]))
            if len(out) >= desired: break
    return out[:desired]

async def rewrite_markdown_to_avoid(rewrite_model: str, rewrite_temp: float, top_p: float,
                                    persona: Dict[str, Any], md: str, forbid: List[str],
                                    presence_penalty: float, frequency_penalty: float) -> str:
    forbid_list = "\n".join(f"- {p}" for p in forbid[:40])
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"PERSONA: {digest_persona(persona)}\n\nFORBIDDEN:\n{forbid_list}\n\nMARKDOWN TO REWRITE:\n{md}"}
    ]
    resp = await chat_create_safe(
        rewrite_model, messages,
        want_json=False,
        temperature=rewrite_temp, top_p=top_p,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        max_tokens=420
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Analysis & Save (with tracking)
# ---------------------------
async def analyze_and_save(
    root: pathlib.Path, persona: Dict[str, Any], run: Dict[str, Any], pid: str,
    *, analysis_model: str, analysis_temp: float, analysis_top_p: float,
    presence_penalty: float, frequency_penalty: float,
    corpus_texts: List[str], corpus_phrases: Set[str], forbid_phrases: Set[str],
    used_minor_categories: Set[str], used_good_categories: Set[str],
    diversify_threshold: float, diversify_retries: int,
    unique_minor_global: bool, min_unique_minor: int,
    rewrite_model: Optional[str], rewrite_temp: float, rewrite_top_p: float,
    ngram_n: int, score_weights: Optional[Dict[str, float]],
    humanize: bool, humanize_intensity: int, score_bias: float
):
    # 분석 호출
    try:
        analysis_resp = await chat_create_safe(
            analysis_model,
            [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user",   "content": json.dumps({"persona": persona, **run}, ensure_ascii=False)}
            ],
            want_json=True,
            temperature=analysis_temp, top_p=analysis_top_p,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
            max_tokens=400
        )
        aobj = json.loads(analysis_resp.choices[0].message.content)
    except Exception:
        aobj = {}

    desc = aobj.get("description") if isinstance(aobj.get("description"), str) else None
    if not desc: desc = fallback_desc(persona)

    sig = collect_signals(run.get("history", []))
    rng = rng_for_persona(persona)
    prof = persona_profile(persona)

    # 마크다운 없으면 페르소나 기반으로 생성
    md = aobj.get("markdown") if isinstance(aobj.get("markdown"), str) else None
    if not md:
        notes = [h for h in run.get("history", []) if h.get("action") == "note" and not str(h.get("tag","")).startswith("milestone_")]
        ci_txt = "None observed."
        if notes:
            prefer = [n for n in notes if n.get("tag") in ("diet_mismatch","allergen_missing","filter_missing","aria_missing","contrast_low","tiny_tap_target","fee_transparency")]
            sel = prefer[0] if prefer else notes[0]
            ci_txt = f"step {sel.get('step')}: {sel.get('tag')} — {sel.get('detail')}"

        positives = persona_toned_positive(sig, prof)
        while len(positives) < 2:
            positives.append(random.choice(POSITIVES_POOL))
        positives = positives[:3]

        frs = []
        if prof["is_strict_diet"] and not any(n.get("tag")=="diet_mismatch" for n in notes):
            frs.append("I wanted clearer vegan/vegetarian markers on list cards.")
        if prof["has_allergen"] and not any(n.get("tag")=="allergen_missing" for n in notes):
            frs.append("Allergen flags could appear earlier than deep in the details.")
        if prof["acc_is_screenreader"]:
            frs.append("Some filter chips didn’t sound distinct to screen readers.")
        if prof["acc_is_largetext"]:
            frs.append("A few labels felt dense for quick reading.")
        if prof["acc_is_colorblind"]:
            frs.append("A couple of states relied heavily on color to signal selection.")
        if len(frs) < 2:
            frs.extend([random.choice(MINOR_CATEGORY_POOLS["filter_discoverability"]),
                        random.choice(MINOR_CATEGORY_POOLS["label_ambiguity"])])
        frs = frs[:3]

        imps = suggest_from_minor(frs, rng, k=2)

        md = (
            "## Persona\n"
            f"{desc}\n\n"
            "## What Worked Well\n"
            + "".join(f"- {x}\n" for x in positives) + "\n"
            "## Critical Issues\n"
            f"- {ci_txt}\n\n"
            "## Minor Friction\n"
            + "".join(f"- {x}\n" for x in frs) + "\n"
            "## Suggested Improvements\n"
            + "".join(f"- {x}\n" for x in imps)
        )

    # 유사성 검사 + 리라이트(겹치면)
    def too_similar(markdown: str) -> bool:
        sim = sim_against_corpus(markdown, corpus_texts, n=ngram_n) if corpus_texts else 0.0
        overlap = 0
        for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
            for b in extract_bullets(markdown, h):
                nb = normalize_line(b)
                if nb in corpus_phrases or nb in forbid_phrases:
                    overlap += 1
        return sim >= diversify_threshold or overlap >= 2

    tries = 0
    while too_similar(md) and tries < diversify_retries:
        forbid = list((corpus_phrases | forbid_phrases))[:60]
        md2 = await rewrite_markdown_to_avoid(
            rewrite_model or analysis_model, rewrite_temp, rewrite_top_p,
            persona, md, forbid,
            presence_penalty, frequency_penalty
        )
        if md2 and not too_similar(md2):
            md = md2; break
        tries += 1

    # Minor → Suggestion 매핑(최종)
    current_mf = extract_bullets(md, "## Minor Friction")
    new_imps = suggest_from_minor(current_mf, rng, k=2)
    md = replace_section(md, "## Suggested Improvements", [f"- {x}" for x in new_imps])

    # 글로벌 유니크: Minor, 그리고 긍정도 카테고리 분산(간단)
    if unique_minor_global:
        desired = max(1, min(3, min_unique_minor))
        uniq_mf = unique_bullets_with_global(current_mf, used_minor_categories, desired, rng, MINOR_CATEGORY_POOLS)
        md = replace_section(md, "## Minor Friction", ["- " + x for x in uniq_mf])

    # 긍정도 중복 카테고리 축소(라이트)
    goods = extract_bullets(md, "## What Worked Well")
    if goods:
        # 긍정 카테고리는 간단히 normalize해서 프레이즈 중복만 피함
        filtered = []
        for g in goods:
            ng = normalize_line(g)
            if ng not in used_good_categories:
                filtered.append(g); used_good_categories.add(ng)
        if filtered:
            md = replace_section(md, "## What Worked Well", ["- " + x for x in filtered])

    # 섹션 공백(정확히 1줄)
    md = enforce_spacing_exact_one(md)

    # (선택) 사람톤 약간 추가: contractions 등 — 요청시만 켬
    if humanize:
        md = md  # 필요시 humanize 로직 추가 가능; 현재는 간결성을 위해 정리만

    # 점수 + bias
    score_int = score_from_signals(sig, rng, weights=score_weights)
    score_int = max(1, min(5, int(round(score_int + (score_bias or 0.0)))))

    # 저장
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

    # 코퍼스 업데이트(다음 세션 중복 회피용)
    corpus_texts.append(md)
    for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
        for b in extract_bullets(md, h):
            corpus_phrases.add(normalize_line(b))

# ---------------------------
# Baseline diff (optional)
# ---------------------------
def load_baseline_issue(baseline_dir: pathlib.Path, pid: str) -> Optional[Dict[str, Any]]:
    f = baseline_dir / pid / "issues.json"
    if not f.exists(): return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None

def compare_issue_objs(old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    out_lines = []
    diff = {}
    if old.get("score") != new.get("score"):
        out_lines.append(f"- Score: {old.get('score')} → {new.get('score')}")
        diff["score_delta"] = (old.get("score"), new.get("score"))
    if (old.get("description") or "").strip() != (new.get("description") or "").strip():
        out_lines.append("- Description changed.")
    old_md = (old.get("analysis") or ""); new_md = (new.get("analysis") or "")
    if old_md != new_md:
        s = difflib.unified_diff(old_md.splitlines(), new_md.splitlines(),
                                 fromfile="baseline", tofile="current", lineterm="")
        diff_text = "\n".join(list(s)[:120])
        out_lines.append("```diff\n" + diff_text + "\n```")
        diff["markdown_diff"] = True
    return ("\n".join(out_lines) if out_lines else "(no material differences)"), diff

def write_diff_report(root: pathlib.Path, pid: str, baseline: Dict[str, Any], current: Dict[str, Any]):
    text, _ = compare_issue_objs(baseline, current)
    (root / pid / "compare_baseline.md").write_text(f"# Diff vs Baseline for {pid}\n\n{text}\n", encoding="utf-8")

def prime_corpus_from_dir(root: pathlib.Path) -> Tuple[List[str], Set[str]]:
    texts: List[str] = []
    phrases: Set[str] = set()
    if not root.exists(): return texts, phrases
    for f in root.rglob("issues.md"):
        try:
            md = f.read_text(encoding="utf-8")
            texts.append(md)
            for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
                for b in extract_bullets(md, h):
                    phrases.add(normalize_line(b))
        except Exception:
            continue
    return texts, phrases

# ---------------------------
# Navigation + engine
# ---------------------------
async def run_one(play, persona: Dict[str, Any], *,
                  engine: str, headful: bool,
                  agent_model: str, agent_temp: float, agent_top_p: float,
                  dom_chars: int, use_history: bool, history_k: int,
                  max_steps: int, goto_timeout_ms: int, retry_goto: int) -> Dict[str, Any]:

    async def _start(browser_type, goto_timeout_ms):
        browser = await browser_type.launch(headless=not headful)
        try:
            try:
                device = play.devices["iPhone 15"]
            except KeyError:
                device = {
                    "viewport": {"width": 393, "height": 852},
                    "user_agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                                   "Mobile/15E148 Safari/604.1"),
                    "isMobile": True, "hasTouch": True,
                }
            context = await browser.new_context(
                **device, locale="en-US", timezone_id="America/New_York",
                geolocation={"latitude": 40.7128, "longitude": -74.0060},
                permissions=["geolocation"],
            )
            page = await context.new_page()
            await page.goto("https://www.ubereats.com", wait_until="domcontentloaded", timeout=goto_timeout_ms)
            try: await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception: pass
            for sel in ["button:has-text('Accept')","button:has-text('Agree')","button[aria-label*='accept']"]:
                try:
                    if await page.locator(sel).count() > 0:
                        await page.click(sel, timeout=1000); break
                except Exception: pass
            return page, context, browser
        except Exception:
            await browser.close(); raise

    browser_type = getattr(play, engine)
    last_exc = None
    for _ in range(retry_goto + 1):
        try:
            page, context, browser = await _start(browser_type, goto_timeout_ms)
            break
        except Exception as e:
            last_exc = e
            browser_type = play.chromium
    if last_exc and 'page' not in locals():
        raise last_exc

    try:
        result = await act_with_llm(
            page, persona,
            agent_model=agent_model, agent_temp=agent_temp, agent_top_p=agent_top_p,
            dom_chars=dom_chars, use_history=use_history, history_k=history_k,
            max_steps=max_steps
        )
    finally:
        await browser.close()
    return result

# ---------------------------
# Main
# ---------------------------
async def main(args):
    personas = json.load(open(args.personas, encoding="utf-8"))
    root     = pathlib.Path(args.output); root.mkdir(parents=True, exist_ok=True)
    baseline_dir = pathlib.Path(args.baseline_dir) if args.baseline_dir else None

    if args.seed is not None: random.seed(args.seed)

    forbid_from_file: Set[str] = set()
    if args.forbid_phrase_file:
        try:
            with open(args.forbid_phrase_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = normalize_line(line)
                    if s: forbid_from_file.add(s)
        except Exception: pass

    corpus_texts: List[str] = []
    corpus_phrases: Set[str] = set()
    used_minor_categories: Set[str] = set()
    used_good_categories: Set[str]  = set()

    if args.dedupe_against_existing:
        t, p = prime_corpus_from_dir(root)
        corpus_texts.extend(t); corpus_phrases |= p

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async with async_playwright() as p:
        async def run_one_persona(persona, idx):
            pid  = persona.get("id") or f"P-{idx:02}"
            sess = root / pid
            if (sess / "issues.json").exists() and not args.overwrite:
                print(f"{pid} ✔︎ Skip")
                try:
                    md = (sess / "issues.md").read_text(encoding="utf-8")
                    corpus_texts.append(md)
                    for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
                        for b in extract_bullets(md, h):
                            corpus_phrases.add(normalize_line(b))
                except Exception: pass
                return
            print(f"▶ {pid}")
            async with sem:
                result = await run_one(
                    p, persona,
                    engine=args.engine, headful=args.headful,
                    agent_model=args.agent_model, agent_temp=args.agent_temp, agent_top_p=args.agent_top_p,
                    dom_chars=args.dom_chars, use_history=args.use_history, history_k=args.history_k,
                    max_steps=args.max_steps, goto_timeout_ms=args.goto_timeout_ms, retry_goto=args.retry_goto
                )
            await analyze_and_save(
                root, persona, result, pid,
                analysis_model=args.analysis_model, analysis_temp=args.analysis_temp, analysis_top_p=args.analysis_top_p,
                presence_penalty=args.analysis_presence_penalty, frequency_penalty=args.analysis_frequency_penalty,
                corpus_texts=corpus_texts, corpus_phrases=corpus_phrases, forbid_phrases=forbid_from_file,
                used_minor_categories=used_minor_categories, used_good_categories=used_good_categories,
                diversify_threshold=args.diversify_threshold, diversify_retries=args.diversify_retries,
                unique_minor_global=args.unique_minor_global, min_unique_minor=args.min_unique_minor,
                rewrite_model=args.rewrite_model, rewrite_temp=args.rewrite_temp, rewrite_top_p=args.rewrite_top_p,
                ngram_n=args.ngram_n, score_weights=(json.loads(args.score_weights) if args.score_weights else None),
                humanize=args.humanize, humanize_intensity=args.humanize_intensity, score_bias=args.score_bias
            )

            if baseline_dir and baseline_dir.exists():
                cur_issue = json.loads((sess / "issues.json").read_text(encoding="utf-8"))
                base_issue = load_baseline_issue(baseline_dir, pid)
                if base_issue:
                    write_diff_report(root, pid, base_issue, cur_issue)

        tasks = [run_one_persona(persona, i) for i, persona in enumerate(personas, 1)]
        chunk = max(1, args.concurrency) * 5
        for s in range(0, len(tasks), chunk):
            await asyncio.gather(*tasks[s:s+chunk])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True)
    ap.add_argument("--output",   required=True)
    # Engine / UI
    ap.add_argument("--engine", choices=["webkit","chromium","firefox"], default="webkit")
    ap.add_argument("--headful", action="store_true")
    # Models
    ap.add_argument("--agent_model", default="gpt-5-mini")
    ap.add_argument("--analysis_model", default="gpt-5")
    ap.add_argument("--rewrite_model", default=None)
    # Temps
    ap.add_argument("--agent_temp", type=float, default=0.40)
    ap.add_argument("--analysis_temp", type=float, default=1.00)
    ap.add_argument("--rewrite_temp", type=float, default=0.90)
    # Sampling (gpt-5 계열이면 내부에서 자동 제거)
    ap.add_argument("--agent_top_p", type=float, default=0.90)
    ap.add_argument("--analysis_top_p", type=float, default=0.92)
    ap.add_argument("--rewrite_top_p", type=float, default=0.95)
    ap.add_argument("--analysis_presence_penalty", type=float, default=0.2)
    ap.add_argument("--analysis_frequency_penalty", type=float, default=0.2)
    # Prompt/token controls
    ap.add_argument("--use_history", action="store_true")
    ap.add_argument("--history_k", type=int, default=6)
    ap.add_argument("--dom_chars", type=int, default=3500)
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    # Stop markers & navigation
    ap.add_argument("--stop_markers", type=str, default="your cart,review order,review your order,cart subtotal,summary")
    ap.add_argument("--goto_timeout_ms", type=int, default=120000)
    ap.add_argument("--retry_goto", type=int, default=2)
    # Baseline diff
    ap.add_argument("--baseline_dir", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    # De-dup / diversify
    ap.add_argument("--dedupe_against_existing", action="store_true")
    ap.add_argument("--diversify_threshold", type=float, default=0.78)
    ap.add_argument("--diversify_retries", type=int, default=3)
    ap.add_argument("--ngram_n", type=int, default=5)
    # Global uniqueness
    ap.add_argument("--unique_minor_global", action="store_true")
    ap.add_argument("--min_unique_minor", type=int, default=2)
    # External phrases
    ap.add_argument("--forbid_phrase_file", type=str, default=None)
    # Scoring
    ap.add_argument("--score_weights", type=str, default='{"reached_review_bonus":0.5,"w_severe":-1,"w_timeout":-0.5,"w_longwait":-0.5,"w_budget_over":-0.5,"w_budget_met":0.5}')
    ap.add_argument("--score_bias", type=float, default=0.4)  # 평균 3~4를 목표
    # Human tone (light)
    ap.add_argument("--humanize", action="store_true")
    ap.add_argument("--humanize_intensity", type=int, default=1)
    # Concurrency & seed
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()
    GLOBAL_STOP_MARKERS[:] = [m.strip().lower() for m in args.stop_markers.split(",") if m.strip()]
    asyncio.run(main(args))
