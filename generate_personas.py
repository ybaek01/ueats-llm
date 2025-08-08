"""
Persona generator with underscore file naming and enforced diversity.

Schema per object:
  id, condition, age, income, location, diet, accessibility, goal

Usage:
  python generate_personas.py --condition uniform --count 7
  python generate_personas.py --condition diet --count 7
  python generate_personas.py --condition diverse --count 7
  # 수동 파일명 지정:
  python generate_personas.py --condition diet --count 7 --out personas_diet_custom.json

Env (optional):
  OPENAI_API_KEY=...  # 있으면 goal을 LLM으로 생성, 없으면 로컬 규칙
"""
import argparse, json, os, random
from typing import List, Dict, Set, Optional

# ----------------------------- Config ---------------------------------
EAST_COAST = [
    "Boston, MA","Cambridge, MA","New York, NY","Brooklyn, NY","Queens, NY",
    "Jersey City, NJ","Newark, NJ","Philadelphia, PA","Baltimore, MD",
    "Washington, DC","Alexandria, VA","Arlington, VA","Richmond, VA",
    "Raleigh, NC","Charlotte, NC","Charleston, SC","Savannah, GA",
    "Atlanta, GA","Orlando, FL","Tampa, FL","Miami, FL","Gainesville, FL"
]

WORLDWIDE = [
    "Seoul, KR","Busan, KR","Tokyo, JP","Kyoto, JP","Taipei, TW","Singapore, SG",
    "Sydney, AU","Melbourne, AU","Auckland, NZ","London, UK","Manchester, UK",
    "Paris, FR","Berlin, DE","Munich, DE","Toronto, CA","Vancouver, CA",
    "Mexico City, MX","São Paulo, BR","Buenos Aires, AR","Istanbul, TR",
    "Dubai, AE","Nairobi, KE","Cairo, EG","Johannesburg, ZA","Mumbai, IN",
    "Bengaluru, IN","Jakarta, ID","Bangkok, TH","Hanoi, VN","Barcelona, ES",
    "Rome, IT","Lisbon, PT"
]

STUDENT_INCOMES = [
    "work-study ($200/week)","campus dining (~$600/mo)","library aide (~$650/mo)",
    "part-time barista (~$800/mo)","retail associate (~$900/mo)",
    "RA stipend ($900/mo)","intern stipend ($1,200/mo)","tutoring (~$18/h, 6h/wk)"
]

DIETS_CYCLIC = [
    "vegan","vegetarian","halal","kosher","gluten-free",
    "lactose-free","pescatarian","low-sodium","nut-free"
]

DIETS_DIVERSE = DIETS_CYCLIC + ["none","low-carb","keto","low-FODMAP"]

ACCESS_LIST_DIVERSE = ["none","screen-reader","large-text","colorblind"]

GOAL_TEMPLATES = [
    "Find an under ${price} {diet_tag} {food} near campus",
    "Order a quick {diet_tag} {food} for pickup within 20 minutes",
    "Locate a late-night {diet_tag} {food} under ${price}",
    "Get a healthy {diet_tag} {food} with delivery fee under ${fee}",
    "Try a new {diet_tag} {cuisine} place within 2 miles",
]
FOODS = ["burrito bowl","ramen","poke bowl","salad","wrap","sandwich","pizza","sushi set","noodle soup","rice bowl"]
CUISINES = ["Thai","Korean","Japanese","Mexican","Mediterranean","Indian","Italian","Vietnamese"]

# -------------------------- LLM (optional) ----------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI()
    except Exception:
        USE_OPENAI = False

def llm_goal_one(diet: str, location: str) -> str:
    if not USE_OPENAI:
        return ""
    prompt = (
        "Write ONE short, concrete food-ordering goal (<=12 words) for a student. "
        f"Diet='{diet}', Location='{location}'. Avoid quotes/emojis/explanations."
    )
    try:
        r = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""

def local_goal(diet: str) -> str:
    tpl = random.choice(GOAL_TEMPLATES)
    diet_tag = "option" if diet == "none" else diet
    return tpl.format(
        diet_tag=diet_tag,
        food=random.choice(FOODS),
        cuisine=random.choice(CUISINES),
        price=random.choice([10,12,14,15]),
        fee=random.choice([2,3,4,5])
    )

def unique_goal(diet: str, location: str, used: Set[str]) -> str:
    for _ in range(8):
        g = llm_goal_one(diet, location) or local_goal(diet)
        if g and g not in used:
            used.add(g)
            return g
    g = (llm_goal_one(diet, location) or local_goal(diet)) + f" #{len(used)+1}"
    used.add(g)
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
        "age": random.randint(18, 35),
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
    # same diet/accessibility (none), ages 19-24, East-Coast, varied student incomes, unique goals
    assert all(o["diet"] == "none" and o["accessibility"] == "none" for o in objs)
    assert all(19 <= o["age"] <= 24 for o in objs)
    assert all(o["location"] in EAST_COAST for o in objs)
    assert len({o["goal"] for o in objs}) == len(objs)
    assert all(isinstance(o["income"], str) and o["income"] for o in objs)

def validate_diet(objs: List[Dict]):
    # only diet changes systematically; ages 19-30, East-Coast, varied student incomes, accessibility none, unique goals
    assert all(19 <= o["age"] <= 30 for o in objs)
    assert all(o["accessibility"] == "none" for o in objs)
    assert all(o["location"] in EAST_COAST for o in objs)
    assert len({o["goal"] for o in objs}) == len(objs)
    # ensure diet variety across set
    assert len({o["diet"] for o in objs}) >= min(5, len(objs)), "Diet variety too low."

def validate_diverse(objs: List[Dict]):
    # every key field varies widely
    assert len({o["location"] for o in objs}) >= min(6, len(objs)), "Location variety too low."
    assert len({o["diet"] for o in objs}) >= min(6, len(objs)), "Diet variety too low."
    assert len({o["accessibility"] for o in objs}) >= min(3, len(objs)), "Accessibility variety too low."
    ages = [o["age"] for o in objs]
    assert min(ages) != max(ages), "Age spread too narrow."

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
            p["goal"] = unique_goal(p["diet"], p["location"], used_goals)
            out.append(p)
        validate_uniform(out)

    elif args.condition == "diet":
        # cycle diets to make diet the systematic axis
        for i in range(args.count):
            diet = DIETS_CYCLIC[i % len(DIETS_CYCLIC)]
            p = build_diet_only(i+1, diet)
            p["goal"] = unique_goal(p["diet"], p["location"], used_goals)
            out.append(p)
        validate_diet(out)

    else:  # diverse — enforce diversity deterministically to avoid assertion failures
        # Force at least min(6, count) unique diets/locations/accessibilities
        k = min(6, args.count)
        diet_pool = random.sample(DIETS_DIVERSE, k=min(len(DIETS_DIVERSE), args.count))
        loc_pool  = random.sample(WORLDWIDE,     k=min(len(WORLDWIDE), args.count))
        # ensure at least 3 unique accessibility values if possible
        acc_needed = min(3, args.count)
        acc_pool = (ACCESS_LIST_DIVERSE * ((acc_needed + len(ACCESS_LIST_DIVERSE)-1)//len(ACCESS_LIST_DIVERSE)))[:args.count]

        for i in range(args.count):
            forced_diet = diet_pool[i] if i < len(diet_pool) else random.choice(DIETS_DIVERSE)
            forced_loc  = loc_pool[i]  if i < len(loc_pool)  else random.choice(WORLDWIDE)
            forced_acc  = acc_pool[i]  if i < len(acc_pool)  else random.choice(ACCESS_LIST_DIVERSE)

            p = build_diverse(i+1, forced_diet=forced_diet,
                              forced_location=forced_loc,
                              forced_access=forced_acc)
            p["goal"] = unique_goal(p["diet"], p["location"], used_goals)
            out.append(p)
        validate_diverse(out)

    # 파일 저장 (underscore)
    out_path = args.out or f"personas_{args.condition}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    main()
