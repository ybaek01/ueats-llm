"""
Persona generator with underscore file naming and enforced diversity.

Schema per object:
  id, condition, age, income, location, diet, accessibility, goal

Usage:
  python generate_personas.py --condition uniform --count 5
  python generate_personas.py --condition diet --count 7
  python generate_personas.py --condition diverse --count 7
  # Custom file name:
  python generate_personas.py --condition diet --count 7 --out personas_diet_custom.json
"""
import argparse, json, os, random
from typing import List, Dict, Set, Optional

# ----------------------------- Config ---------------------------------
EAST_COAST = [
    # MA
    "Boston, MA","Cambridge, MA","Somerville, MA","Worcester, MA","Lowell, MA","Springfield, MA",
    # CT / RI
    "Hartford, CT","New Haven, CT","Stamford, CT","Providence, RI",
    # NY / NJ
    "New York, NY","Brooklyn, NY","Queens, NY","Bronx, NY","Staten Island, NY",
    "Albany, NY","Buffalo, NY","Rochester, NY","Syracuse, NY",
    "Jersey City, NJ","Newark, NJ","Hoboken, NJ",
    # PA / MD / DC
    "Philadelphia, PA","Pittsburgh, PA","Baltimore, MD","Towson, MD",
    "Washington, DC","Alexandria, VA","Arlington, VA","Richmond, VA",
    # NC / SC / GA
    "Raleigh, NC","Durham, NC","Chapel Hill, NC","Charlotte, NC",
    "Wilmington, NC","Charleston, SC","Savannah, GA","Atlanta, GA","Athens, GA",
    # FL
    "Orlando, FL","Tampa, FL","St. Petersburg, FL","Miami, FL","Fort Lauderdale, FL",
    "Gainesville, FL","Jacksonville, FL","Tallahassee, FL"
]

WORLDWIDE = [
    # Korea / Japan / Taiwan / SEA
    "Seoul, KR","Busan, KR","Incheon, KR","Daegu, KR","Daejeon, KR","Gwangju, KR",
    "Tokyo, JP","Kyoto, JP","Osaka, JP","Fukuoka, JP","Sapporo, JP",
    "Taipei, TW","Taichung, TW","Tainan, TW",
    "Singapore, SG","Kuala Lumpur, MY","Jakarta, ID","Bandung, ID","Bangkok, TH","Chiang Mai, TH",
    "Hanoi, VN","Ho Chi Minh City, VN","Manila, PH","Cebu, PH",
    # China & HK
    "Hong Kong, HK","Macau, MO","Shenzhen, CN","Guangzhou, CN","Shanghai, CN","Beijing, CN",
    # ANZ
    "Sydney, AU","Melbourne, AU","Brisbane, AU","Perth, AU","Adelaide, AU",
    "Auckland, NZ","Wellington, NZ",
    # Europe (West/North/South/East)
    "London, UK","Manchester, UK","Birmingham, UK","Edinburgh, UK","Glasgow, UK",
    "Paris, FR","Lyon, FR","Marseille, FR","Nice, FR",
    "Berlin, DE","Munich, DE","Hamburg, DE","Cologne, DE","Frankfurt, DE",
    "Zurich, CH","Geneva, CH","Vienna, AT","Prague, CZ","Budapest, HU","Warsaw, PL","Krakow, PL",
    "Copenhagen, DK","Stockholm, SE","Oslo, NO","Helsinki, FI",
    "Amsterdam, NL","Rotterdam, NL","Brussels, BE","Antwerp, BE",
    "Madrid, ES","Barcelona, ES","Valencia, ES","Seville, ES",
    "Rome, IT","Milan, IT","Florence, IT","Naples, IT",
    "Lisbon, PT","Porto, PT","Athens, GR","Thessaloniki, GR",
    # Middle East
    "Istanbul, TR","Ankara, TR","Izmir, TR",
    "Dubai, AE","Abu Dhabi, AE","Doha, QA","Kuwait City, KW","Manama, BH","Muscat, OM","Riyadh, SA","Jeddah, SA",
    "Amman, JO","Beirut, LB","Tel Aviv, IL",
    # Africa
    "Cairo, EG","Alexandria, EG","Casablanca, MA","Rabat, MA","Tunis, TN",
    "Nairobi, KE","Mombasa, KE","Kampala, UG","Addis Ababa, ET","Dar es Salaam, TZ",
    "Lagos, NG","Abuja, NG","Accra, GH","Johannesburg, ZA","Cape Town, ZA",
    # Americas
    "Toronto, CA","Vancouver, CA","Montreal, CA","Ottawa, CA","Calgary, CA",
    "Mexico City, MX","Guadalajara, MX","Monterrey, MX",
    "São Paulo, BR","Rio de Janeiro, BR","Brasília, BR","Belo Horizonte, BR","Curitiba, BR","Porto Alegre, BR",
    "Buenos Aires, AR","Córdoba, AR","Rosario, AR",
    "Santiago, CL","Valparaíso, CL","Lima, PE","Cusco, PE",
    "Bogotá, CO","Medellín, CO","Quito, EC","Guayaquil, EC",
    "Montevideo, UY","Asunción, PY","La Paz, BO"
]

STUDENT_INCOMES = [
    "work-study ($200/week)","campus dining (~$600/mo)","library aide (~$650/mo)",
    "part-time barista (~$800/mo)","retail associate (~$900/mo)","RA stipend ($900/mo)",
    "intern stipend ($1,200/mo)","tutoring (~$18/h, 6h/wk)","delivery rider (~$16/h, 8h/wk)",
    "IT helpdesk (~$1,100/mo)","lab assistant (~$700/mo)","campus ambassador (~$500/mo)",
    "freelance design (~$900/mo)","gig work (~$400/mo)","TA stipend (~$1,400/mo)"
]

DIETS_CYCLIC = [
    "vegan","vegetarian","pescatarian","halal","kosher",
    "gluten-free","lactose-free","low-sodium","nut-free",
    "soy-free","egg-free","shellfish-free","low-fat"
]
DIETS_DIVERSE = DIETS_CYCLIC + [
    "none","low-carb","keto","low-FODMAP","paleo","whole30","diabetic-friendly","heart-healthy"
]

ACCESS_LIST_DIVERSE = [
    "none","screen-reader","large-text","colorblind",
    "reduced-motion","dyslexia-friendly","motor-impairment","hearing-impaired"
]

FOODS = [
    "burrito bowl","ramen","poke bowl","salad","wrap","sandwich","pizza","sushi set","noodle soup","rice bowl",
    "grilled chicken bowl","falafel wrap","tacos","curry","bibimbap","katsu curry","gyudon","pho","banh mi",
    "shawarma plate","hummus bowl","kebab wrap","mezze box","butter chicken","paneer tikka wrap","samosa combo",
    "pad thai","green curry","laksa","kimchi stew combo","Japchae bowl","teriyaki bowl","yakisoba",
    "grain bowl","cauliflower rice bowl","plant-based burger","veggie dumplings","poke burrito",
    "southern fried chicken sandwich","fish and chips","carbonara","margherita pizza","sicilian slice",
    "arepa","empanadas","ceviche bowl","poutine","smørrebrød"
]

CUISINES = [
    "American","Southern","BBQ","Mexican","Tex-Mex","Peruvian","Brazilian","Argentinian","Colombian",
    "Italian","Sicilian","Mediterranean","Middle Eastern","Turkish","Greek","Israeli","Lebanese",
    "Indian","Pakistani","Bangladeshi","Sri Lankan","Nepalese",
    "Thai","Vietnamese","Filipino","Indonesian","Malaysian","Singaporean",
    "Korean","Japanese","Chinese","Taiwanese","Hong Kong",
    "French","Spanish","Portuguese","German","Austrian","Swiss","Belgian","Dutch","Nordic"
]

REGIONAL_CUISINE_HINTS = {
  # North America
  "US": ["American","Southern","BBQ","Mexican","Tex-Mex","Italian","Chinese","Japanese","Mediterranean","Korean"],
  "CA": ["American","Chinese","Japanese","Mediterranean","Indian","French","Lebanese"],
  # Europe
  "UK": ["Indian","Mediterranean","Chinese","Japanese","American","Turkish"],
  "EU": ["Italian","Mediterranean","Vietnamese","Japanese","Indian","Greek","Spanish","Portuguese","Turkish","Lebanese","Nordic"],
  # East Asia
  "KR": ["Korean","Japanese","Chinese","American","Mediterranean"],
  "JP": ["Japanese","Korean","Chinese","Italian"],
  "TW": ["Taiwanese","Japanese","Chinese","American"],
  "HK": ["Hong Kong","Chinese","Japanese","Korean"],
  # SE Asia / South Asia
  "SEA": ["Thai","Vietnamese","Malaysian","Singaporean","Indonesian","Japanese","Korean","Chinese"],
  "SA": ["Indian","Pakistani","Bangladeshi","Sri Lankan","Nepalese","Middle Eastern"],
  # Middle East
  "ME": ["Middle Eastern","Mediterranean","Turkish","Indian","Pakistani"],
  # Latin America & Africa
  "LATAM": ["Mexican","Peruvian","Brazilian","Argentinian","Colombian","Spanish","Portuguese"],
  "AFR": ["Ethiopian","Moroccan","Tunisian","Egyptian","Nigerian","Ghanaian","South African","Lebanese","Turkish"]
}

BUDGET_BY_INCOME = {
  "very_low": [7,8,9,10],
  "low":      [8,9,10,11,12],
  "student":  [9,10,11,12,13],
  "mid":      [12,13,14,15,16,18],
  "high":     [15,16,18,20,22,24]
}

ACCESS_SUFFIX = {
  "none": "",
  "screen-reader": " with clear, screen-reader-friendly labels",
  "large-text": " with easy-to-read, larger menus",
  "colorblind": " with high-contrast, non-color-only cues",
  "reduced-motion": " with reduced motion and simple transitions",
  "dyslexia-friendly": " with dyslexia-friendly fonts and spacing",
  "motor-impairment": " with larger tap targets and fewer precise swipes",
  "hearing-impaired": " with visual cues for status changes"
}

# -------------------------- LLM (optional) ----------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI()
    except Exception:
        USE_OPENAI = False

def llm_goal_one(diet: str, location: str, delivery_or_pickup: str, budget: int, access: str) -> str:
    if not USE_OPENAI:
        return ""
    prompt = (
        "Write ONE concrete food-ordering goal (8-12 words), imperative voice.\n"
        f"Must include a dish or cuisine, diet='{diet}', mode='{delivery_or_pickup}', "
        f"budget under ${budget}, fit location='{location}'. "
        "Start with 'Order'. No quotes, no emojis, no extra commentary. "
        f"Append this accessibility suffix verbatim at the end if non-empty: '{ACCESS_SUFFIX.get(access,'')}'."
    )
    try:
        r = oai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.6
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""

# ----------------------- Local goal generation ------------------------
def _income_band(income: str) -> str:
    s = income.lower()
    if any(k in s for k in ["work-study","~$400","~$600","$700","$800","$900"]): return "low"
    if any(k in s for k in ["$1,200","$1,400","$1,800"]): return "mid"
    return "mid"

def _country_hint(loc: str) -> str:
    if ", KR" in loc: return "KR"
    if ", JP" in loc: return "JP"
    if ", AU" in loc: return "AU"
    if ", NZ" in loc: return "NZ"
    if ", UK" in loc: return "UK"
    if ", CA" in loc: return "CA"
    if any(x in loc for x in [", FR", ", DE", ", IT", ", ES", ", PT"]): return "EU"
    if any(x in loc for x in [", SG", ", TW", ", VN", ", TH", ", ID"]): return "SEA"
    if any(x in loc for x in [", AE", ", TR", ", EG"]): return "ME"
    return "US"

def _pick_cuisine_for(loc: str) -> str:
    hint = _country_hint(loc)
    pool = REGIONAL_CUISINE_HINTS.get(hint, CUISINES)
    return random.choice(pool) if pool else random.choice(CUISINES)

def _mode() -> str:
    # 표현 통일: 항상 "Order ..."로 시작. 수단만 Delivery/Pickup 랜덤.
    return random.choice(["Delivery","Pickup"])

def local_goal(diet: str, location: str, income: str, access: str) -> str:
    """LLM 없이도 일관/현실적인 목표문 생성: Order + (Delivery/Pickup), 예산, 음식/요리, 간단 맥락, 접근성."""
    mode = _mode()
    diet_tag = "option" if diet == "none" else diet
    price = random.choice(BUDGET_BY_INCOME[_income_band(income)])
    cuisine_or_food = random.choice([random.choice(FOODS), _pick_cuisine_for(location)])
    ctx = random.choice([
        "near campus","between classes","for a late-night snack",
        "within 20 minutes","with low delivery fee","before study session"
    ])
    suffix = ACCESS_SUFFIX.get(access, "")
    return f"Order a {diet_tag} {cuisine_or_food} ({mode}, under ${price}) {ctx}{suffix}"

def _normalize_goal(s: str) -> str:
    # 예산 숫자/모드까지 포함해 중복 방지 강도↑
    return " ".join(s.lower().strip().split())

def _valid_goal(g: str) -> bool:
    if not g: return False
    ws = g.split()
    if len(ws) < 5 or len(ws) > 16: return False
    if '"' in g or "'" in g: return False
    if any(bad in g.lower() for bad in ["http://","https://","emoji"]): return False
    return True

def unique_goal(diet: str, location: str, income: str, access: str, used: Set[str]) -> str:
    for _ in range(8):
        mode = _mode()
        budget = random.choice(BUDGET_BY_INCOME[_income_band(income)])
        g = llm_goal_one(diet, location, mode, budget, access)
        if not g:
            g = local_goal(diet, location, income, access)
        if _valid_goal(g):
            key = _normalize_goal(g)
            if key not in used:
                used.add(key)
                return g
    g = local_goal(diet, location, income, access) + f" #{len(used)+1}"
    used.add(_normalize_goal(g))
    return g

# ---------------------------- Builders --------------------------------
def make_id(prefix: str, n: int) -> str:
    return f"{prefix}-{n:02}"

def build_uniform(n: int) -> Dict:
    return {
        "id": make_id("U", n),
        "condition": "uniform",
        "age": random.randint(19, 24),
        "income": random.choice(STUDENT_INCOMES),
        "location": random.choice(EAST_COAST),
        "diet": "none",
        "accessibility": "none",
        "goal": ""  # later
    }

def build_diet_only(n: int, diet: str) -> Dict:
    return {
        "id": make_id("D", n),
        "condition": "diet",
        "age": random.randint(19, 30),
        "income": random.choice(STUDENT_INCOMES),
        "location": random.choice(EAST_COAST),
        "diet": diet,
        "accessibility": "none",
        "goal": ""  # later
    }

def build_diverse(n: int,
                  forced_diet: Optional[str] = None,
                  forced_location: Optional[str] = None,
                  forced_access: Optional[str] = None) -> Dict:
    return {
        "id": make_id("F", n),
        "condition": "diverse",
        "age": random.randint(18, 65),
        "income": random.choice([
            "student stipend ($700/mo)","gig work (~$400/mo)","family support (~$1,000/mo)",
            "part-time dev ($1,800/mo)","scholarship + TA ($1,400/mo)","freelance design (~$900/mo)"
        ]),
        "location": forced_location or random.choice(WORLDWIDE),
        "diet": forced_diet or random.choice(DIETS_DIVERSE),
        "accessibility": forced_access or random.choice(ACCESS_LIST_DIVERSE),
        "goal": ""  # later
    }

# --------------------------- Validation --------------------------------
def validate_uniform(objs: List[Dict]):
    assert all(o["diet"] == "none" and o["accessibility"] == "none" for o in objs)
    assert all(19 <= o["age"] <= 24 for o in objs)
    assert all(o["location"] in EAST_COAST for o in objs)
    assert len({o["goal"] for o in objs}) == len(objs)
    assert all(isinstance(o["income"], str) and o["income"] for o in objs)

def validate_diet(objs: List[Dict]):
    assert all(19 <= o["age"] <= 30 for o in objs)
    assert all(o["accessibility"] == "none" for o in objs)
    assert all(o["location"] in EAST_COAST for o in objs)
    assert len({o["goal"] for o in objs}) == len(objs)
    assert len({o["diet"] for o in objs}) >= min(5, len(objs)), "Diet variety too low."

def validate_diverse(objs: List[Dict]):
    assert len({o["location"] for o in objs}) >= min(6, len(objs)), "Location variety too low."
    assert len({o["diet"] for o in objs}) >= min(6, len(objs)), "Diet variety too low."
    assert len({o["accessibility"] for o in objs}) >= min(3, len(objs)), "Accessibility variety too low."
    ages = [o["age"] for o in objs]
    assert min(ages) != max(ages), "Age spread too narrow."
    assert len({o["goal"] for o in objs}) == len(objs), "Goals must be unique."

# ------------------------------ Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["uniform","diet","diverse"], required=True)
    ap.add_argument("--count", type=int, default=7)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="출력 파일명 (기본: personas_<condition>.json)")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    used_goals: Set[str] = set()
    out: List[Dict] = []

    if args.condition == "uniform":
        for i in range(args.count):
            p = build_uniform(i+1)
            p["goal"] = unique_goal(p["diet"], p["location"], p["income"], p["accessibility"], used_goals)
            out.append(p)
        validate_uniform(out)

    elif args.condition == "diet":
        for i in range(args.count):
            diet = DIETS_CYCLIC[i % len(DIETS_CYCLIC)]
            p = build_diet_only(i+1, diet)
            p["goal"] = unique_goal(p["diet"], p["location"], p["income"], p["accessibility"], used_goals)
            out.append(p)
        validate_diet(out)

    else:  # diverse
        k = min(6, args.count)
        diet_pool = random.sample(DIETS_DIVERSE, k=min(len(DIETS_DIVERSE), args.count))
        loc_pool  = random.sample(WORLDWIDE,     k=min(len(WORLDWIDE), args.count))
        acc_needed = min(3, args.count)
        acc_pool = (ACCESS_LIST_DIVERSE * ((acc_needed + len(ACCESS_LIST_DIVERSE)-1)//len(ACCESS_LIST_DIVERSE)))[:args.count]

        for i in range(args.count):
            forced_diet = diet_pool[i] if i < len(diet_pool) else random.choice(DIETS_DIVERSE)
            forced_loc  = loc_pool[i]  if i < len(loc_pool)  else random.choice(WORLDWIDE)
            forced_acc  = acc_pool[i]  if i < len(acc_pool)  else random.choice(ACCESS_LIST_DIVERSE)

            p = build_diverse(i+1, forced_diet=forced_diet,
                              forced_location=forced_loc,
                              forced_access=forced_acc)
            p["goal"] = unique_goal(p["diet"], p["location"], p["income"], p["accessibility"], used_goals)
            out.append(p)
        validate_diverse(out)

    out_path = args.out or f"personas_{args.condition}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    main()
