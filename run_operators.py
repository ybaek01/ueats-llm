# run_operators.py  (FINAL)
# - Uber Eats mobile (iPhone) UX agent + persona-based analysis (English reports)
# - PRE-CHECKOUT ONLY (never click Pay / Checkout / Apple Pay / Place order)
# - Human-centered issues first; network talk minimized
# - Sections: Persona / What Worked Well / Critical Issues / Minor Friction / Suggested Improvements
# - Strong de-dup: global per-section stores + intra-section unique + cooldown
# - Dynamic axis suggestions (target 30/axis) + rewrite for variety
# - GPT-5 param guard (no top_p, etc.), max_tokens→max_completion_tokens auto-map

import argparse, asyncio, json, pathlib, random, re, time, difflib
from typing import Dict, Any, List, Optional, Tuple, Set
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from openai import AsyncOpenAI, BadRequestError

client = AsyncOpenAI()

MAX_STEPS_DEFAULT = 60
PROHIBITED_CLICK_PAT = re.compile(r"(place\s*order|apple\s*pay|google\s*pay|\bpay\b|\bcheckout\b)", re.I)
GLOBAL_STOP_MARKERS: List[str] = ["your cart", "review order", "review your order", "cart subtotal", "summary"]

COMPACT_SYSTEM = """
You are a mobile UX agent on Uber Eats (iPhone). Output STRICT JSON only:
{"action":"click|type|wait_ms|wait_for|note","selector":"<css|text>","text":"<opt>","ms":<opt>,"state":"<opt>","tag":"<opt>","detail":"<opt>"}

Rules:
- PRE-CHECKOUT ONLY. NEVER click: "Place order", "Pay", "Apple Pay", "Checkout".
- Prefer state waits: wait_for {state: visible|attached|hidden|detached} (≤1500ms).
  wait_ms only if necessary (avoid exactly 500ms; prefer 350ms or 700ms).
- Parse persona for budget (e.g., "under $12"). If an item exceeds budget,
  emit: {"action":"note","tag":"budget_over","detail":"$14 > $12 target"}; if within budget, emit "budget_met".
- Use diet/allergen filters/search; verify item details vs persona needs.
- On human-centered issues, emit ONE note(tag in: diet_mismatch, allergen_missing, filter_missing,
  label_ambiguous, fee_transparency, upsell_overwhelming, aria_missing, contrast_low, tiny_tap_target; + detail).
- Record milestones: milestone_item_added, milestone_cart_open, milestone_review.
- Stop when cart/review is visible or a blocker occurs.
Think step-by-step silently. OUTPUT JSON ONLY.
""".strip()

ANALYSIS_SYSTEM = r"""
You are a meticulous UX auditor. Analyze the session 'history' AS the persona (English output).
Focus on human-centered issues related to THIS persona’s constraints (diet/allergen/accessibility/budget).
Avoid network latency unless repeated or >2s; the test stops before real checkout. If no critical issue, say "None observed."

Return STRICT JSON with EXACTLY three string keys: "score", "description", "markdown".
- "score": one of {"1","2","3","4","5"} (integer string). If pre-checkout reached and constraints respected, prefer "5".
- "description": ONE line referencing persona constraints, e.g., "<age>yo in <location> (<income>; vegan/large-text); goal: <goal>."
- "markdown": <=200 words, first-person, headers exactly:

## Persona
(plain)

## What Worked Well
(bullets, tie to my constraints)

## Critical Issues
(ONE most impactful issue pointing to a 'history' note, e.g., "step 12 note: diet_mismatch — …"; or "None observed.")

## Minor Friction
(1–3 bullets, concrete, not generic)

## Suggested Improvements
(1–3 bullets, persona-tailored, concrete)
"""

REWRITE_SYSTEM = """
You are a UX report rewriter. Rewrite the given markdown to avoid overlapping wording with prior reports.
Keep the same structure and headers (## Persona, ## What Worked Well, ## Critical Issues, ## Minor Friction, ## Suggested Improvements),
stay under 200 words, preserve the same facts and evidence, but use different wording and light contractions.
Output ONLY the markdown (no JSON).
""".strip()

# ---------------- Param guards (GPT-5 family) ----------------
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

        for k in ("temperature","top_p","presence_penalty","frequency_penalty","response_format"):
            if k in params and (k in msg or "unsupported" in msg.lower()):
                drop(k)
        _normalize_token_arg(model, params, max_tokens)
        if cleaned:
            return await _try(params)
        for k in list(params.keys()):
            if k in ("max_tokens","max_completion_tokens"): continue
            if f"'{k}'" in msg or "unsupported" in msg.lower():
                drop(k)
        _normalize_token_arg(model, params, max_tokens)
        return await _try(params)

# ---------------- Similarity / normalization ----------------
_STOPWORDS = set("the a an to for of on in at by with and or but so that as is are was were be been being it this those these my your our their from into over under within before after between across".split())

def _simple_stem(w: str) -> str:
    w = w.lower()
    for suf in ("ing","ed","ly","es","s"):
        if w.endswith(suf) and len(w) > len(suf)+2:
            w = w[: -len(suf)]
    return w

def normalize_line(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    toks = [_simple_stem(t) for t in toks]
    return " ".join(toks).strip()

def char_sim_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

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

def combined_similar(a: str, b: str, *, n: int = 4, j_thresh: float = 0.70, c_thresh: float = 0.82) -> bool:
    tj = jaccard(ngrams(tokens(a), n=n), ngrams(tokens(b), n=n))
    cr = char_sim_ratio(a, b)
    return (tj >= j_thresh) or (cr >= c_thresh)

def sim_against_corpus(text: str, corpus_texts: List[str], n: int = 4) -> float:
    t = ngrams(tokens(text), n=n)
    best = 0.0
    for c in corpus_texts:
        s = jaccard(t, ngrams(tokens(c), n=n))
        if s > best: best = s
    return best

# ---------------- Small helpers ----------------
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
            elif a in ('wait','wait_ms'):
                slim.append({"step": h.get("step"), "action": a})
            elif a == 'wait_for':
                slim.append({"step": h.get("step"), "action": a, "state": h.get("state")})
            elif a == "note":
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

def enforce_spacing_exact_one(md: str) -> str:
    headers = ["## What Worked Well","## Critical Issues","## Minor Friction","## Suggested Improvements"]
    for h in headers:
        md = re.sub(rf"\n*(?={re.escape(h)})", "\n", md)
        md = md.replace("\n" + h, "\n\n" + h)
        md = re.sub(rf"\n\n\n+{re.escape(h)}", "\n\n" + h, md)
    md = re.sub(r"(\n## Minor Friction\n(?:.*?))\n\n+(## Suggested Improvements)", r"\1\n\n\2", md, flags=re.S)
    return md.strip() + "\n"

# ---------------- Persona / signals ----------------
ALLERGEN_KEYWORDS = ["gluten-free","nut-free","soy-free","lactose-free","shellfish-free","egg-free","low-sodium"]
STRICT_DIET = ["vegan","vegetarian","pescatarian","halal","kosher"]

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

# ---------------- Pools (positives/minor/improvements) ----------------
POSITIVES_POOL = [
    "Diet filters surfaced early and felt dependable.",
    "Clear badges let me verify restrictions at a glance.",
    "Search results aligned with my intent without much tweaking.",
    "The add-to-cart step was straightforward and snappy.",
    "Prices were visible soon enough to guide choices.",
    "Delivery and service fees appeared before review.",
    "Labels and micro-copy made comparisons quick.",
    "Add-ons were present but didn’t derail my task.",
    "I reached review quickly with minimal detours.",
    "Cuisine and diet chips combined to narrow choices fast.",
    "Portion and price info appeared on cards when I needed them.",
    "Saved filters persisted across screens during my session."
]

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
        "Some filter chips didn’t sound distinct to screen readers.",
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
    ],
    "info_density": [
        "Some cards packed many elements, slowing quick scanning.",
        "Dense layouts made it harder to spot key diet details."
    ],
    "redundant_steps": [
        "I repeated similar confirmations across two screens.",
        "I hit the same choice twice before seeing the cart."
    ],
    "thumb_reach": [
        "Primary actions occasionally sat outside comfortable thumb reach.",
        "Important toggles landed too near the screen’s top edge."
    ]
}

SUGGESTION_POOLS = {
    "too_many_taps": [
        "Combine related steps so I reach review faster.",
        "Compress confirmations into a single clear step."
    ],
    "filter_discoverability": [
        "Surface diet and price chips at the top of the listing.",
        "Make filter entry points more prominent than sort."
    ],
    "label_ambiguity": [
        "Align naming between list cards and item details.",
        "Add short descriptors to distinguish similar options."
    ],
    "loading_feedback": [
        "Show a clear applied state when filters update.",
        "Add a brief toast to confirm changes took effect."
    ],
    "stall": [
        "Provide a subtle skeleton/loading cue during updates.",
        "Allow quick ‘retry’ if a panel doesn’t open promptly."
    ],
    "fee_transparency": [
        "Expose estimated fees on the listing cards earlier.",
        "Show a toggle to include fees in the price preview."
    ],
    "upsell_pressure": [
        "Tuck add-ons behind a ‘Customize’ affordance by default.",
        "Reduce the frequency and size of upsell prompts pre-checkout."
    ],
    "diet_badges": [
        "Standardize vegan/vegetarian badges on list and item pages.",
        "Add dedicated diet chips filterable at the top bar."
    ],
    "allergen_flags": [
        "Introduce at-a-glance allergen chips on item cards.",
        "Place a prominent allergen section near the ‘Add’ button."
    ],
    "aria_accessibility": [
        "Announce filter state changes to screen readers.",
        "Add clear ARIA labels for chips and selected states."
    ],
    "contrast_low": [
        "Increase contrast for badges and selected states.",
        "Avoid color-only signals; add icons or underlines."
    ],
    "tap_target_small": [
        "Enlarge tap targets for chips and add-on toggles.",
        "Increase spacing to reduce accidental taps."
    ],
    "budget_visibility": [
        "Let me set a hard price cap before seeing results.",
        "Add an ‘Under my budget’ quick filter on the list."
    ],
    "price_sorting": [
        "Offer a one-tap ‘Cheapest first’ sorting.",
        "Expose delivery-included totals in card prices."
    ],
    "info_density": [
        "Reduce visual density on cards; elevate the most decision-critical info.",
        "Allow hiding secondary metadata until expanded."
    ],
    "redundant_steps": [
        "Merge duplicate confirmations into a single, explicit step.",
        "Skip repeated choices by remembering recent selections."
    ],
    "thumb_reach": [
        "Keep primary actions within thumb zone on mobile.",
        "Anchor filter chips near the bottom for easier reach."
    ],
}

SUGGESTIONS_AXIS = {
  "vegan": [
    "Add a 'strict vegan only' toggle that removes items with dairy or eggs by default.",
    "Show a plant-based certification badge with a short tooltip explaining criteria.",
    "Offer a 'swap to vegan protein' quick action on item cards."
  ],
  "vegetarian": [
    "Label rennet and gelatin clearly; provide veggie-safe cheese info on item pages.",
    "Provide a one-tap 'vegetarian only' filter at the top bar."
  ],
  "halal": [
    "Display a ‘Halal-certified’ badge with cert body and inspection date on the item.",
    "Add a 'no alcohol in preparation' note where applicable."
  ],
  "kosher": [
    "Surface kosher certification with meat/dairy/pareve classification at a glance.",
    "Add a 'kosher kitchens near me' smart filter on the listing."
  ],
  "gluten-free": [
    "Show cross-contact warnings near the 'Add' button instead of deep in details.",
    "Offer a 'gluten-free crust' quick switch on pizza cards."
  ],
  "lactose-free": [
    "Add 'swap dairy to lactose-free' options inline with price delta.",
    "Expose lactose content in sauces and cheeses with tiny info chips."
  ],
  "pescatarian": [
    "Provide a 'seafood-only' facet and mark items cooked on shared grills."
  ],
  "low-sodium": [
    "Expose sodium estimates on cards; add a '≤ X mg' quick cap."
  ],
  "nut-free": [
    "Highlight nut-free prep and shared-facility warnings right by the 'Add' button."
  ],
  "low-carb": [
    "Offer 'low-carb swaps' (cauli rice, lettuce wrap) inline on the card."
  ],
  "keto": [
    "Show net carbs per item and add a '≤ N g net carbs' toggle up top."
  ],
  "low-FODMAP": [
    "Flag high-FODMAP ingredients with an icon and offer low-FODMAP swaps."
  ],
  "screen-reader": [
    "Ensure ARIA names on primary actions (e.g., 'Add to cart: <item>').",
    "Announce price and fee updates via live regions when filters change."
  ],
  "large-text": [
    "Provide a persistent text-size control the site remembers.",
    "Avoid truncation; wrap long item names instead of ellipsizing."
  ],
  "colorblind": [
    "Never rely on color alone; add icons and labels for selected chips.",
    "Increase contrast for price and fee chips per WCAG AA."
  ],
  "motor-impairment": [
    "Use large primary 'Add' buttons instead of tiny plus icons; reduce precise swipes.",
    "Keep important actions within thumb zone on mobile."
  ],
  "reduced-motion": [
    "Offer a 'reduced motion' preference that also disables skeleton shimmer loops."
  ],
  "dyslexia-friendly": [
    "Provide a dyslexia-friendly font option and 1.5x line spacing on menus."
  ],
  "budget": [
    "Expose an 'Under $<budget>' cap as a top chip that includes fees in the preview.",
    "Show fees-included totals on listing cards, not just at review."
  ],
  "nav": [
    "Pin 'Filters' above 'Sort' on mobile and remember last-used facets.",
    "Add cuisine × diet quick chips (e.g., Vegan × Thai) at the top."
  ],
  "generic": [
    "Add a 'Save for later' option to reduce decision pressure.",
    "Provide a 'Compare' mode for two items with key differences highlighted."
  ],
}

# ---------------- Global used stores & cooldown ----------------
def _load_used_set(root: pathlib.Path, fname: str) -> Set[str]:
    f = root / fname
    if f.exists():
        try: return set(json.loads(f.read_text(encoding="utf-8")))
        except Exception: return set()
    return set()

def _save_used_set(root: pathlib.Path, fname: str, used: Set[str]):
    (root / fname).write_text(json.dumps(sorted(list(used)), ensure_ascii=False, indent=2), encoding="utf-8")

def _load_used_counts(root: pathlib.Path, fname: str) -> Dict[str,int]:
    f = root / fname
    if f.exists():
        try: return dict(json.loads(f.read_text(encoding="utf-8")))
        except Exception: return {}
    return {}

def _save_used_counts(root: pathlib.Path, fname: str, d: Dict[str,int]):
    (root / fname).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- Agent loop ----------------
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
                       dom_chars: int, use_history: bool, history_k: int,
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
                temperature=agent_temp, max_tokens=220
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

# ---------------- Suggestion machinery ----------------
async def generate_axis_suggestions(persona: Dict[str, Any], axis: str, *,
                                    model: str, temp: float,
                                    need: int,
                                    forbid_phrases: Set[str],
                                    corpus_phrases: Set[str],
                                    ngram_n: int = 4, thresh: float = 0.70) -> List[str]:
    persona_line = digest_persona(persona)
    forbidden = "\n".join(sorted(list(forbid_phrases))[:80])

    sys = ("You are a seasoned mobile UX researcher writing concise, concrete product suggestions. "
           "Generate persona-tailored improvements that a designer can ship. Output STRICT JSON: "
           "{\"suggestions\":[\"...\", \"...\"]}. No commentary.")
    usr = (
        f"Persona: {persona_line}\n"
        f"Axis: {axis}\n"
        "Write 15–20 unique suggestions (12–18 words each). Be concrete and human-centered:\n"
        "- Prefer diet/allergen/accessibility/budget/navigation insights relevant to the axis.\n"
        "- Avoid vague phrasing; avoid repeating ideas; each line should propose a distinct fix.\n"
        "- Vary verbs and structure; sound like a person, not a template.\n"
        "- Avoid phrases in FORBIDDEN and anything already used.\n\n"
        "FORBIDDEN:\n" + forbidden + "\n"
        "Return JSON only."
    )
    try:
        resp = await chat_create_safe(
            model,
            [{"role":"system","content":sys},{"role":"user","content":usr}],
            want_json=True,
            temperature=temp,
            max_tokens=700
        )
        obj = json.loads(resp.choices[0].message.content or "{}")
        cand = obj.get("suggestions") or []
        out: List[str] = []
        for s in cand:
            s = (str(s) or "").strip()
            if not s: continue
            nrm = normalize_line(s)
            if not nrm: continue
            if nrm in corpus_phrases or nrm in forbid_phrases:
                continue
            if any(combined_similar(s, x) for x in out):
                continue
            out.append(s)
            if len(out) >= need: break
        return out
    except Exception:
        return []

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

def unique_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for s in lines:
        if not any(combined_similar(s, t) for t in out):
            out.append(s)
    return out

def categorize_bullet(text: str) -> str:
    t = normalize_line(text)
    kw = [
        ("too_many_taps", ["too many taps","longer than expected","confirm choices"]),
        ("filter_discoverability", ["filter placement","find where to filter","controls felt buried"]),
        ("label_ambiguity", ["generic","wording","didnt convey"]),
        ("loading_feedback", ["loading","feedback","applied","updated","acknowledgment"]),
        ("stall", ["stall","pause","retrying"]),
        ("fee_transparency", ["fee","breakdown"]),
        ("upsell_pressure", ["upsell","add-on","add on","crowded"]),
        ("diet_badges", ["diet badge","vegan marker","vegetarian marker"]),
        ("allergen_flags", ["allergen"]),
        ("aria_accessibility", ["screen reader","aria","assistive"]),
        ("contrast_low", ["contrast","color alone"]),
        ("tap_target_small", ["tap target","mis-tap"]),
        ("budget_visibility", ["budget","cap results","under my target"]),
        ("price_sorting", ["sorting by price","cheapest"]),
        ("info_density", ["dense","packed","many elements"]),
        ("redundant_steps", ["repeat","repeated","same choice"]),
        ("thumb_reach", ["thumb zone","near the top"])
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
        cand = [x for x in pool if not any(combined_similar(x, y) for y in out)]
        if not cand: cand = pool
        out.append(rng.choice(cand))
        if len(out) >= k: break
    if len(out) < k:
        all_cats = list(SUGGESTION_POOLS.keys()); rng.shuffle(all_cats)
        for c in all_cats:
            out.append(rng.choice(SUGGESTION_POOLS[c]))
            if len(out) >= k: break
    return unique_lines(out)[:k]

def candidate_axes(persona: dict) -> List[str]:
    axes = []
    diet = (persona.get("diet") or "none").lower()
    if diet in SUGGESTIONS_AXIS: axes.append(diet)
    acc = (persona.get("accessibility") or "none").lower()
    if acc in SUGGESTIONS_AXIS: axes.append(acc)
    axes.append("budget")
    axes.append("nav")
    return axes

def persona_toned_positive(sig: Dict[str, Any], prof: Dict[str, Any]) -> List[str]:
    out = []
    if sig.get("budget_met"):
        out.append("Prices stayed within my target without much effort.")
    if prof["is_strict_diet"] and sig.get("m_item"):
        out.append("Diet labels aligned with my choice, so I felt confident adding it.")
    if prof["has_allergen"] and (sig.get("m_item") or sig.get("m_cart")):
        out.append("Allergen info was close enough to the Add button to double-check quickly.")
    if prof["acc_is_screenreader"]:
        out.append("Key actions were labeled clearly for assistive tech.")
    if prof["acc_is_largetext"]:
        out.append("Text size and spacing made scanning options quick.")
    if prof["acc_is_colorblind"]:
        out.append("Selected states didn’t rely only on color.")
    if sig.get("m_cart") or sig.get("m_review") or sig.get("prechk_stop"):
        out.append("I reached the review step with only a few detours.")
    return [x for x in out if x]

async def rewrite_markdown_to_avoid(rewrite_model: str, rewrite_temp: float,
                                    persona: Dict[str, Any], md: str, forbid: List[str]) -> str:
    forbid_list = "\n".join(f"- {p}" for p in forbid[:80])
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"PERSONA: {digest_persona(persona)}\n\nFORBIDDEN:\n{forbid_list}\n\nMARKDOWN TO REWRITE:\n{md}"}
    ]
    resp = await chat_create_safe(
        rewrite_model, messages,
        want_json=False,
        temperature=rewrite_temp,
        max_tokens=420
    )
    return resp.choices[0].message.content.strip()

async def choose_suggestions(persona: dict,
                             used_global: Set[str],
                             need: int,
                             *,
                             external_axis: Dict[str, List[str]],
                             suggestions_model: str,
                             suggestions_temp: float,
                             suggestions_per_axis: int,
                             forbid_phrases: Set[str],
                             corpus_phrases: Set[str],
                             ngram_n: int = 4,
                             thresh: float = 0.70) -> List[str]:
    order = candidate_axes(persona) + ["generic"]
    rng = rng_for_persona(persona)
    picked: List[str] = []

    async def axis_pool(ax: str) -> List[str]:
        base = list(SUGGESTIONS_AXIS.get(ax, [])) + list(external_axis.get(ax, []))
        cleaned: List[str] = []
        for s in base:
            s = s.strip()
            if not s: continue
            nrm = normalize_line(s)
            if not nrm or nrm in forbid_phrases or nrm in corpus_phrases:
                continue
            if any(combined_similar(s, t) for t in cleaned):
                continue
            cleaned.append(s)

        if len(cleaned) < suggestions_per_axis:
            need_more = suggestions_per_axis - len(cleaned)
            dyn = await generate_axis_suggestions(
                persona, ax, model=suggestions_model, temp=suggestions_temp,
                need=need_more*3,
                forbid_phrases=forbid_phrases|corpus_phrases,
                corpus_phrases=corpus_phrases,
                ngram_n=4, thresh=0.70
            )
            for s in dyn:
                if any(combined_similar(s, t) for t in cleaned):
                    continue
                cleaned.append(s)
                if len(cleaned) >= suggestions_per_axis:
                    break
        rng.shuffle(cleaned)
        return cleaned

    for ax in order:
        pool = await axis_pool(ax)
        for s in pool:
            if any(combined_similar(s, u) for u in used_global):
                continue
            if any(combined_similar(s, u) for u in picked):
                continue
            picked.append(s)
            if len(picked) >= need:
                return picked

    while len(picked) < need:
        g = "Offer a clearer, mobile-first control for my constraint with concise labeling."
        if not any(combined_similar(g, u) for u in used_global|set(picked)):
            picked.append(g)
        else:
            picked.append(g + " Include brief examples on the first tap.")
    return picked[:need]

# ---------------- Analysis & write ----------------
async def analyze_and_save(
    root: pathlib.Path, persona: Dict[str, Any], run: Dict[str, Any], pid: str,
    *, analysis_model: str, analysis_temp: float,
    corpus_texts: List[str], corpus_phrases: Set[str], forbid_phrases: Set[str],
    used_minor_categories: Set[str], used_good_categories: Set[str],
    diversify_threshold: float, diversify_retries: int,
    unique_minor_global: bool, min_unique_minor: int,
    unique_sugg_global: bool, min_unique_sugg: int,
    rewrite_model: Optional[str], rewrite_temp: float,
    ngram_n: int, score_weights: Optional[Dict[str, float]],
    score_bias: float, humanize: bool,
    used_suggestions_global: Set[str],
    suggestions_axis_external: Dict[str, List[str]],
    suggestions_model: str, suggestions_temp: float, suggestions_per_axis: int,
    cooldown_max_per_phrase: int, cooldown_max_per_category: int
):
    try:
        analysis_resp = await chat_create_safe(
            analysis_model,
            [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user",   "content": json.dumps({"persona": persona, **run}, ensure_ascii=False)}
            ],
            want_json=True,
            temperature=analysis_temp,
            max_tokens=420
        )
        aobj = json.loads(analysis_resp.choices[0].message.content)
    except Exception:
        aobj = {}

    desc = aobj.get("description") if isinstance(aobj.get("description"), str) else None
    if not desc: desc = (f"{persona.get('age','?')}yo in {persona.get('location','?')} "
                         f"({persona.get('income','?')}; {persona.get('diet','none')}/{persona.get('accessibility','none')}); "
                         f"goal: {persona.get('goal','n/a')}")

    sig = collect_signals(run.get("history", []))
    rng = rng_for_persona(persona)
    prof = persona_profile(persona)

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
        positives = unique_lines(positives)[:3]

        frs = []
        if prof["is_strict_diet"]:
            frs.append("I wanted clearer vegan/vegetarian markers on list cards.")
        if prof["has_allergen"]:
            frs.append("Allergen flags could appear earlier than deep in the details.")
        if prof["acc_is_screenreader"]:
            frs.append("Some filter chips didn’t sound distinct to screen readers.")
        if prof["acc_is_largetext"]:
            frs.append("A few labels felt dense for quick reading.")
        if prof["acc_is_colorblind"]:
            frs.append("A couple of states relied heavily on color to signal selection.")
        while len(frs) < 2:
            frs.append(random.choice(MINOR_CATEGORY_POOLS["label_ambiguity"]))
        frs = unique_lines(frs)[:3]

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

    def too_similar(md_text: str) -> bool:
        sim = sim_against_corpus(md_text, corpus_texts, n=ngram_n) if corpus_texts else 0.0
        overlap = 0
        for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
            for b in extract_bullets(md_text, h):
                nb = normalize_line(b)
                if nb in corpus_phrases or nb in forbid_phrases:
                    overlap += 1
        return sim >= diversify_threshold or overlap >= 2

    tries = 0
    while too_similar(md) and tries < diversify_retries and rewrite_model:
        forbid = list((corpus_phrases | forbid_phrases))[:80]
        md2 = await rewrite_markdown_to_avoid(
            rewrite_model, rewrite_temp, persona, md, forbid
        )
        if md2 and not too_similar(md2):
            md = md2; break
        tries += 1

    # ---------- Intra-section de-dup & global uniqueness ----------
    used_good_global  = _load_used_set(root, "_used_good.json")
    used_minor_global = _load_used_set(root, "_used_minor.json")
    used_sugg_global  = _load_used_set(root, "_used_sugg.json")
    used_sugg_counts  = _load_used_counts(root, "_used_sugg_counts.json")
    used_sugg_catcnt  = _load_used_counts(root, "_used_sugg_cat_counts.json")

    def bump_count(d: Dict[str,int], key: str):
        d[key] = int(d.get(key, 0)) + 1

    # What Worked Well
    goods = unique_lines(extract_bullets(md, "## What Worked Well"))
    filt_goods = []
    for g in goods:
        if not any(combined_similar(g, u) for u in used_good_global):
            filt_goods.append(g)
            used_good_global.add(g)
    if filt_goods:
        md = replace_section(md, "## What Worked Well", ["- " + x for x in filt_goods])

    # Minor Friction
    minors = unique_lines(extract_bullets(md, "## Minor Friction"))
    unique_minors = []
    for m in minors:
        if not any(combined_similar(m, u) for u in used_minor_global):
            unique_minors.append(m)
            used_minor_global.add(m)
    if unique_minors:
        md = replace_section(md, "## Minor Friction", ["- " + x for x in unique_minors])

    # Suggested Improvements with cooldown
    imps = unique_lines(extract_bullets(md, "## Suggested Improvements"))
    final_imps = []
    for s in imps:
        cat = categorize_bullet(s)
        nrm = normalize_line(s)
        over_phrase = used_sugg_counts.get(nrm, 0) >= cooldown_max_per_phrase
        over_cat    = used_sugg_catcnt.get(cat, 0) >= cooldown_max_per_category
        if any(combined_similar(s, u) for u in used_sugg_global) or over_phrase or over_cat:
            continue
        final_imps.append(s)
        used_sugg_global.add(s)
        bump_count(used_sugg_counts, nrm)
        bump_count(used_sugg_catcnt, cat)

    # 보충 필요 시 축 기반 선택
    need_more = max(0, min_unique_sugg - len(final_imps))
    if need_more > 0:
        more = await choose_suggestions(
            persona, used_sugg_global, need_more,
            external_axis=suggestions_axis_external,
            suggestions_model=suggestions_model,
            suggestions_temp=suggestions_temp,
            suggestions_per_axis=suggestions_per_axis,
            forbid_phrases=forbid_phrases, corpus_phrases=corpus_phrases,
            ngram_n=ngram_n, thresh=diversify_threshold
        )
        for s in more:
            final_imps.append(s)
            used_sugg_global.add(s)
            bump_count(used_sugg_counts, normalize_line(s))
            bump_count(used_sugg_catcnt, categorize_bullet(s))

    if final_imps:
        md = replace_section(md, "## Suggested Improvements", ["- " + x for x in final_imps])

    md = enforce_spacing_exact_one(md)

    # ---------- Score ----------
    score_int = score_from_signals(sig, rng, weights=score_weights)
    score_int = max(1, min(5, int(round(score_int + (score_bias or 0.0)))))

    # ---------- Save ----------
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

    corpus_texts.append(md)
    for h in ["## What Worked Well","## Minor Friction","## Suggested Improvements"]:
        for b in extract_bullets(md, h):
            corpus_phrases.add(normalize_line(b))

    _save_used_set(root, "_used_good.json", used_good_global)
    _save_used_set(root, "_used_minor.json", used_minor_global)
    _save_used_set(root, "_used_sugg.json", used_sugg_global)
    _save_used_counts(root, "_used_sugg_counts.json", used_sugg_counts)
    _save_used_counts(root, "_used_sugg_cat_counts.json", used_sugg_catcnt)

# ---------------- Baseline / corpus ----------------
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

# ---------------- Runner ----------------
async def run_one(play, persona: Dict[str, Any], *,
                  engine: str, headful: bool,
                  agent_model: str, agent_temp: float,
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
            agent_model=agent_model, agent_temp=agent_temp,
            dom_chars=dom_chars, use_history=use_history, history_k=history_k,
            max_steps=max_steps
        )
    finally:
        await browser.close()
    return result

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

    # suggestions axis loader
    def load_suggestions_file(path: Optional[str]) -> Dict[str, List[str]]:
        if not path: return {}
        fp = pathlib.Path(path)
        if not fp.exists(): return {}
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            out: Dict[str, List[str]] = {}
            for k, v in (data or {}).items():
                key = (k or "").strip().lower()
                if not key: continue
                if isinstance(v, list):
                    out[key] = [str(x).strip() for x in v if str(x).strip()]
            return out
        except Exception:
            return {}

    suggestions_axis_external = load_suggestions_file(args.suggestions_file)
    used_suggestions_global: Set[str] = _load_used_set(root, "_used_suggestions.json")

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
                    agent_model=args.agent_model, agent_temp=args.agent_temp,
                    dom_chars=args.dom_chars, use_history=args.use_history, history_k=args.history_k,
                    max_steps=args.max_steps, goto_timeout_ms=args.goto_timeout_ms, retry_goto=args.retry_goto
                )
            await analyze_and_save(
                root, persona, result, pid,
                analysis_model=args.analysis_model, analysis_temp=args.analysis_temp,
                corpus_texts=corpus_texts, corpus_phrases=corpus_phrases, forbid_phrases=forbid_from_file,
                used_minor_categories=used_minor_categories, used_good_categories=used_good_categories,
                diversify_threshold=args.diversify_threshold, diversify_retries=args.diversify_retries,
                unique_minor_global=args.unique_minor_global, min_unique_minor=args.min_unique_minor,
                unique_sugg_global=args.unique_suggestions_global, min_unique_sugg=args.min_unique_suggestions,
                rewrite_model=args.rewrite_model, rewrite_temp=args.rewrite_temp,
                ngram_n=args.ngram_n, score_weights=json.loads(args.score_weights) if args.score_weights else None,
                score_bias=args.score_bias, humanize=args.humanize,
                used_suggestions_global=used_suggestions_global,
                suggestions_axis_external=suggestions_axis_external,
                suggestions_model=args.suggestions_model,
                suggestions_temp=args.suggestions_temp,
                suggestions_per_axis=args.suggestions_per_axis,
                cooldown_max_per_phrase=args.cooldown_max_per_phrase,
                cooldown_max_per_category=args.cooldown_max_per_category
            )

            # persist global suggestion set for resume-ability
            _save_used_set(root, "_used_suggestions.json", used_suggestions_global)

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

    ap.add_argument("--engine", choices=["webkit","chromium","firefox"], default="webkit")
    ap.add_argument("--headful", action="store_true")

    ap.add_argument("--agent_model", default="gpt-5-mini")
    ap.add_argument("--analysis_model", default="gpt-5")
    ap.add_argument("--rewrite_model", default="gpt-5-mini")

    ap.add_argument("--agent_temp", type=float, default=0.40)
    ap.add_argument("--analysis_temp", type=float, default=1.00)
    ap.add_argument("--rewrite_temp", type=float, default=1.20)

    ap.add_argument("--use_history", action="store_true")
    ap.add_argument("--history_k", type=int, default=6)
    ap.add_argument("--dom_chars", type=int, default=3500)
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)

    ap.add_argument("--stop_markers", type=str, default="your cart,review order,review your order,cart subtotal,summary")
    ap.add_argument("--goto_timeout_ms", type=int, default=120000)
    ap.add_argument("--retry_goto", type=int, default=2)

    ap.add_argument("--baseline_dir", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--dedupe_against_existing", action="store_true")
    ap.add_argument("--diversify_threshold", type=float, default=0.72)
    ap.add_argument("--diversify_retries", type=int, default=6)
    ap.add_argument("--ngram_n", type=int, default=4)

    ap.add_argument("--unique_minor_global", action="store_true")
    ap.add_argument("--min_unique_minor", type=int, default=3)
    ap.add_argument("--unique_suggestions_global", action="store_true")
    ap.add_argument("--min_unique_suggestions", type=int, default=2)

    ap.add_argument("--forbid_phrase_file", type=str, default=None)

    ap.add_argument("--score_weights", type=str, default='{"reached_review_bonus":0.5,"w_severe":-1,"w_timeout":-0.5,"w_longwait":-0.5,"w_budget_over":-0.5,"w_budget_met":0.5}')
    ap.add_argument("--score_bias", type=float, default=0.8)
    ap.add_argument("--humanize", action="store_true")

    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)

    # axis suggestions
    ap.add_argument("--suggestions_file", type=str, default=None,
                    help="Axis suggestions JSON. keys like 'vegan','screen-reader','budget','nav','generic'")
    ap.add_argument("--suggestions_model", type=str, default="gpt-5-mini")
    ap.add_argument("--suggestions_per_axis", type=int, default=30)
    ap.add_argument("--suggestions_temp", type=float, default=1.20)

    # cooldowns
    ap.add_argument("--cooldown_max_per_phrase", type=int, default=1,
                    help="Same improvement phrase can appear at most N times globally before replacement.")
    ap.add_argument("--cooldown_max_per_category", type=int, default=3,
                    help="Same improvement category can appear at most N times globally before replacement.")

    args = ap.parse_args()
    GLOBAL_STOP_MARKERS[:] = [m.strip().lower() for m in args.stop_markers.split(",") if m.strip()]
    asyncio.run(main(args))
