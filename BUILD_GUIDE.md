# Brim Burgers Chatbot — Step-by-Step Build Guide

Build order: project setup → knowledge layer → tests → tools → agent → API + chat page →
test scripts → Docker and docs. Every step ends with a check so you know it works before moving on.

All commands are for **Windows PowerShell** with **uv**. Run them from the project folder.

---

## Step 0 — Project setup

### 0.1 Create the folders

```powershell
mkdir brim-chatbot
cd brim-chatbot
mkdir app, data, tests, scripts, static
New-Item app\__init__.py, tests\__init__.py -ItemType File
```

Copy your `brim_knowledge_base.json` into `data\`.

Final layout:

```
brim-chatbot/
├── app/
│   ├── __init__.py        (empty)
│   ├── config.py          Step 1
│   ├── knowledge.py       Step 2
│   ├── tools.py           Step 4
│   ├── agent.py           Step 5
│   └── main.py            Step 6
├── data/brim_knowledge_base.json
├── static/index.html      Step 6
├── tests/
│   ├── __init__.py        (empty)
│   └── test_knowledge.py  Step 3
├── scripts/
│   ├── questions.txt      Step 7
│   └── ask_questions.py   Step 7
├── .env.example  .env  .gitignore  .dockerignore
├── requirements.txt  pytest.ini  Dockerfile  README.md
```

### 0.2 Install uv, create the environment, install packages

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.12
.venv\Scripts\activate
```

**`requirements.txt`**

```text
langchain-core>=0.3
langchain-groq>=0.3
opik>=1.7
fastapi>=0.115
uvicorn[standard]>=0.30
python-dotenv>=1.0
pydantic>=2.7
pytest>=8
```

```powershell
uv pip install -r requirements.txt
```

### 0.3 Settings files

**`.env.example`**

```bash
# Groq (free key: https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Opik cloud (key: https://www.comet.com/opik -> your profile -> API key)
OPIK_API_KEY=your_opik_key_here
OPIK_WORKSPACE=your_workspace_name
OPIK_PROJECT_NAME=brim-burgers-chatbot

# Comma-separated origins allowed to call the API (the client's website)
ALLOWED_ORIGINS=http://localhost:8000,https://brimburgers.pk
```

```powershell
copy .env.example .env
notepad .env
```

Put your real Groq key in `.env` (free at https://console.groq.com/keys). The Opik key can wait
until Step 5; without it the bot simply runs without tracing.

**`.gitignore`**

```text
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.log
.DS_Store
```

**`pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

`pythonpath = .` lets the tests import `app.knowledge`.

---

## Step 1 — Settings (`app/config.py`)

Loads `.env` once, sets the Opik project name, switches tracing off when there is no Opik key,
and gives a Pakistan-time clock. Pakistan has no daylight saving, so a fixed UTC+5 is used
(this avoids needing the `tzdata` package on Windows).

**`app/config.py`**

```python
"""Settings loaded from .env. Import this module before opik so its env vars are set."""

import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

os.environ.setdefault("OPIK_PROJECT_NAME", "brim-burgers-chatbot")
# No Opik key -> run without tracing instead of failing or prompting.
if not os.getenv("OPIK_API_KEY"):
    os.environ.setdefault("OPIK_TRACK_DISABLE", "true")

MAX_TOOL_ROUNDS = 5
MEMORY_MESSAGES = 10

# Pakistan has no daylight saving: a fixed UTC+5 works on every OS
# (zoneinfo needs the extra tzdata package on Windows).
PKT = timezone(timedelta(hours=5), name="PKT")


def pk_now() -> datetime:
    """Current date and time in Pakistan (naive datetime, PKT wall clock)."""
    return datetime.now(PKT).replace(tzinfo=None)
```

---

## Step 2 — Knowledge layer (`app/knowledge.py`)

Plain Python, no LLM. This is the most important file: **every price and total the bot ever
shows is computed here.** What's inside:

- `normalize()` — lowercase, drop apostrophes, punctuation → spaces, so `Nash'd` = `nashd`.
- `BM25` — a small keyword ranker. One text chunk per menu item, branch, add-on group and brand
  section. Good for exact names like "Mozzy Meltdown" and "Askari 11", where a vector DB adds nothing.
- `find_items()` — fuzzy lookup in 3 stages: exact name (returns *all* items with that name, so
  both "Holy Chipotle" items come back) → contains → close spelling (similarity ≥ 0.82).
  The threshold is strict on purpose so "chicken biryani" is reported as unknown instead of
  silently becoming "Chicken Run".
- `match_category()` — aliases: burgers = Beef + Chicken, shakes/sundaes = Sips & Scoops,
  drinks = Beverages (+ Coolers), kids = Brim Junior, wings = We Winged It, deals = deal categories.
- `filter_items()` — category, max price, keyword, cheapest / most expensive first.
- `meal_options()` — Make It A Meal totals plus item + each frequently-bought side, with an
  optional budget. `link_for()` finds the side's order link (the data says "Brim Box Chicken",
  the menu says "Chicken Brim Box", so it also tries the words reversed).
- `order_total()` — `[{name, quantity, size}]` → lines + subtotal; unknown items listed separately.
  It also understands sizes inside the name ("Serious Brim Burger").
- `check_delivery()` — exact area match first, near misses only if there's no exact match,
  and always returns the city (Model Town exists in Lahore, Multan and Sargodha).
- `is_available_now()` — parses deal limits like `Mon-Thu, 12:00 PM - 6:00 PM` and overnight
  windows like `11:00 AM - 3:00 AM`, so the LLM never has to reason about time.

**`app/knowledge.py`**

```python
"""Knowledge layer for the Brim Burgers chatbot.

Plain Python, no LLM. Loads the JSON knowledge base, builds BM25-indexed text
chunks, and provides structured lookups. Every price and total the bot shows
is computed here, never by the LLM.
"""

import json
import math
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "brim_knowledge_base.json"

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "is",
    "are", "do", "does", "you", "your", "i", "me", "my", "what", "which", "how",
    "much", "have", "has", "it", "at", "by", "from", "can", "please", "any",
    "kya", "hai", "ka", "ki", "ke", "mein", "me", "aur", "se",
}


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn punctuation into spaces."""
    text = str(text).lower().replace("'", "").replace("’", "")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def rs(amount) -> str:
    """Format a PKR amount: 1499 -> 'Rs 1,499'."""
    if isinstance(amount, (int, float)):
        return f"Rs {int(amount):,}"
    return f"Rs {amount}"


# ---------------------------------------------------------------------------
# BM25 (pure Python)
# ---------------------------------------------------------------------------


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = [Counter(t) for t in self.docs_tokens]
        df = Counter()
        for toks in self.docs_tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += self.idf[w] * f * (self.k1 + 1) / denom
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Category aliases
# ---------------------------------------------------------------------------

DEAL_CATEGORIES = ["Brim Off Hour Deals", "Spicy Saturdays Offer"]

CATEGORY_ALIASES = {
    "burger": ["Beef Burgers", "Chicken Burger"],
    "burgers": ["Beef Burgers", "Chicken Burger"],
    "beef": ["Beef Burgers"],
    "beef burger": ["Beef Burgers"],
    "beef burgers": ["Beef Burgers"],
    "chicken burger": ["Chicken Burger"],
    "chicken burgers": ["Chicken Burger"],
    "shake": ["Sips & Scoops"],
    "shakes": ["Sips & Scoops"],
    "milkshake": ["Sips & Scoops"],
    "milkshakes": ["Sips & Scoops"],
    "sundae": ["Sips & Scoops"],
    "sundaes": ["Sips & Scoops"],
    "ice cream": ["Sips & Scoops"],
    "drink": ["Beverages", "Coolers"],
    "drinks": ["Beverages", "Coolers"],
    "soft drink": ["Beverages"],
    "soft drinks": ["Beverages"],
    "cold drink": ["Beverages"],
    "cold drinks": ["Beverages"],
    "kids": ["Brim Junior"],
    "kid": ["Brim Junior"],
    "kids meal": ["Brim Junior"],
    "junior": ["Brim Junior"],
    "wings": ["We Winged It"],
    "wing": ["We Winged It"],
    "deal": DEAL_CATEGORIES,
    "deals": DEAL_CATEGORIES,
    "offer": DEAL_CATEGORIES,
    "offers": DEAL_CATEGORIES,
    "wrap": ["Wraps", "Lil Wrapstars"],
    "wraps": ["Wraps", "Lil Wrapstars"],
    "small wraps": ["Lil Wrapstars"],
    "hot dog": ["Hot Dogs"],
    "hotdog": ["Hot Dogs"],
    "hotdogs": ["Hot Dogs"],
    "fries": ["Fries", "Loaded Fries", "Fries Of The Month"],
    "sides": ["Fries", "Appetizers", "Loaded Fries"],
    "starters": ["Appetizers"],
    "dessert": ["Desserts", "Sips & Scoops"],
    "desserts": ["Desserts", "Sips & Scoops"],
    "sweet": ["Desserts", "Sips & Scoops"],
    "salad": ["Naked Box"],
    "salads": ["Naked Box"],
    "sauce": ["Dips"],
    "sauces": ["Dips"],
}


def parse_time(text: str) -> int:
    """'12:00 PM' -> minutes after midnight."""
    t = datetime.strptime(text.strip().upper(), "%I:%M %p")
    return t.hour * 60 + t.minute


DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _parse_days(text: str) -> set[int]:
    days = set()
    for part in text.split(","):
        part = part.strip().lower()[:]
        if "-" in part:
            a, b = [p.strip()[:3] for p in part.split("-")]
            i, j = DAY_NAMES.index(a), DAY_NAMES.index(b)
            k = i
            while True:
                days.add(k)
                if k == j:
                    break
                k = (k + 1) % 7
        elif part[:3] in DAY_NAMES:
            days.add(DAY_NAMES.index(part[:3]))
    return days


def is_available_now(available: str | None, now: datetime) -> bool | None:
    """Check an 'available' string such as 'Mon-Thu, 12:00 PM - 6:00 PM'.

    Returns True/False, or None when there is no limit or it can't be parsed.
    A time window that crosses midnight (11:00 AM - 3:00 AM) belongs to the day
    it starts on.
    """
    if not available:
        return None
    m = re.match(
        r"^(?:(?P<days>[A-Za-z ,\-]+?),\s*)?(?P<start>\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(?P<end>\d{1,2}:\d{2}\s*[AP]M)$",
        available.strip(),
    )
    if not m:
        return None
    start, end = parse_time(m["start"]), parse_time(m["end"])
    days = _parse_days(m["days"]) if m["days"] else set(range(7))
    t = now.hour * 60 + now.minute
    today = now.weekday()
    if start < end:
        return today in days and start <= t < end
    # crosses midnight
    if t >= start:
        return today in days
    if t < end:
        return (today - 1) % 7 in days
    return False


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    def __init__(self, path: Path | str = DATA_PATH):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.brand: dict = self.data["brand"]
        self.menu: list[dict] = self.data["menu"]
        self.add_ons: dict = self.data.get("add_ons", {})
        self.branches: list[dict] = self.data["branches"]
        self.other_locations: list[dict] = self.data.get("other_locations", [])
        self.categories: list[str] = sorted({i["category"] for i in self.menu})
        self._norm_names = [normalize(i["name"]) for i in self.menu]
        self._build_index()

    # ----------------------------------------------------------- chunks/BM25
    def _item_text(self, item: dict) -> str:
        lines = [f"{item['name']} ({item['category']}): {rs(item['price'])}"]
        if item.get("original_price"):
            lines[0] += f" (was {rs(item['original_price'])})"
        if item.get("description"):
            lines.append(item["description"])
        if item.get("sizes"):
            lines.append("Sizes: " + ", ".join(f"{s['name']} {rs(s['price'])}" for s in item["sizes"]))
        for group, opts in (item.get("choices") or {}).items():
            lines.append(f"{group}: " + ", ".join(opts))
        if item.get("make_it_a_meal"):
            lines.append("Make it a meal: " + ", ".join(
                f"{m['meal']} {rs(m['total_price'])}" for m in item["make_it_a_meal"]))
        if item.get("available"):
            lines.append(f"Available: {item['available']}")
        lines.append(f"Order: {item['order_link']}")
        return "\n".join(lines)

    def _branch_text(self, b: dict) -> str:
        return (
            f"{b['name']}, {b['city']}. Address: {b['address']}. Phone: {b['phone']}. "
            f"Hours: {b['hours']}. Services: {b['services']}. Delivery fee: Rs {b['delivery_fee']}. "
            f"Minimum order: Rs {b['minimum_order']}. Delivery time: {b['delivery_time_minutes']} min. "
            f"Delivers to: {', '.join(b['delivers_to'])}. Map: {b['map']}"
        )

    def _build_index(self):
        self.chunks: list[dict] = []
        for item in self.menu:
            self.chunks.append({"type": "menu_item", "title": item["name"], "text": self._item_text(item)})
        for b in self.branches:
            self.chunks.append({"type": "branch", "title": b["name"], "text": self._branch_text(b)})
        for group, opts in self.add_ons.items():
            text = f"Add-ons - {group}: " + ", ".join(f"{o['name']} {rs(o['price'])}" for o in opts)
            self.chunks.append({"type": "add_on", "title": group, "text": text})
        for key, value in self.brand.items():
            if isinstance(value, dict):
                value = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in value.items())
            elif isinstance(value, list):
                value = ", ".join(value)
            self.chunks.append({"type": "brand", "title": key,
                                "text": f"Brim Burgers {key.replace('_', ' ')}: {value}"})
        for loc in self.other_locations:
            self.chunks.append({"type": "other_location", "title": loc["name"],
                                "text": f"{loc['name']}: {loc['note']}"})
        self.bm25 = BM25([c["title"] + " " + c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """BM25 retrieval over all chunks. Returns the top-k with a score."""
        scores = self.bm25.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{**self.chunks[i], "score": round(scores[i], 3)} for i in ranked[:k] if scores[i] > 0]

    # ----------------------------------------------------------- item lookup
    def find_items(self, name: str, limit: int = 5) -> list[dict]:
        """Fuzzy item lookup by name.

        1. exact normalized name (returns every item with that name, e.g. both
           'Holy Chipotle' items)
        2. query contained in the name, or name contained in the query
        3. token overlap / similarity ratio
        """
        q = normalize(name)
        if not q:
            return []
        exact = [it for it, n in zip(self.menu, self._norm_names) if n == q]
        if exact:
            return exact
        q_compact = q.replace(" ", "")
        contains = [it for it, n in zip(self.menu, self._norm_names)
                    if q in n or q_compact in n.replace(" ", "") or (len(n.split()) >= 2 and n in q)]
        if contains:
            # shortest name first = closest match
            return sorted(contains, key=lambda it: len(it["name"]))[:limit]
        q_tokens = set(q.split())
        scored = []
        for it, n in zip(self.menu, self._norm_names):
            n_tokens = set(n.split())
            # every query word found in the name, or the spelling is close
            overlap = 1.0 if q_tokens <= n_tokens else 0.0
            ratio = SequenceMatcher(None, q, n).ratio()
            score = max(overlap, ratio)
            if score >= 0.82:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return []
        best = scored[0][0]
        return [it for s, it in scored if s >= best - 0.1][:limit]

    def find_item(self, name: str) -> dict | None:
        found = self.find_items(name)
        return found[0] if found else None

    # ----------------------------------------------------------- categories
    def match_category(self, text: str) -> list[str]:
        """Map user wording to real category names (aliases included)."""
        q = normalize(text)
        if not q:
            return []
        if q in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[q]
        for cat in self.categories:
            if normalize(cat) == q:
                return [cat]
        matches = [c for c in self.categories if q in normalize(c) or normalize(c) in q]
        if matches:
            return matches
        if q.endswith("s") and q[:-1] in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[q[:-1]]
        return []

    def filter_items(self, category: str | None = None, max_price: float | None = None,
                     keyword: str | None = None, sort: str = "cheapest",
                     limit: int = 15) -> dict:
        """Filter the menu. sort = 'cheapest' or 'most_expensive'."""
        items = self.menu
        cats: list[str] = []
        if category:
            cats = self.match_category(category)
            if not cats:
                return {"error": f"Unknown category '{category}'.", "categories": self.categories}
            items = [i for i in items if i["category"] in cats]
        if max_price is not None:
            items = [i for i in items if i["price"] <= max_price]
        if keyword:
            kw = normalize(keyword)
            items = [i for i in items if kw in normalize(
                i["name"] + " " + i.get("description", "") + " " + i["category"])]
        reverse = sort.lower().startswith(("most", "exp", "high", "desc"))
        items = sorted(items, key=lambda i: i["price"], reverse=reverse)
        return {"categories_matched": cats, "count": len(items), "items": items[:limit]}

    # ----------------------------------------------------------- meals
    def _price_for_size(self, item: dict, size: str | None) -> tuple[int, int | None, str | None]:
        """Return (price, original_price, size_name) for an item and optional size."""
        if size and item.get("sizes"):
            s_norm = normalize(size)
            for s in item["sizes"]:
                if normalize(s["name"]) == s_norm or s_norm in normalize(s["name"]):
                    return s["price"], s.get("original_price"), s["name"]
            return None, None, size  # unknown size
        return item["price"], item.get("original_price"), None

    def link_for(self, name: str) -> str | None:
        it = self.find_item(name)
        if not it:
            # data uses "Brim Box Chicken" for the menu's "Chicken Brim Box"
            words = normalize(name).split()
            it = self.find_item(" ".join(reversed(words)))
        return it["order_link"] if it else None

    def meal_options(self, name: str, budget: float | None = None) -> dict:
        """Make-It-A-Meal totals plus item + each frequently-bought item."""
        found = self.find_items(name)
        if not found:
            return {"error": f"No menu item matches '{name}'."}
        results = []
        for item in found[:3]:
            price = item["price"]
            options = []
            for m in item.get("make_it_a_meal", []):
                options.append({
                    "type": "make_it_a_meal",
                    "description": f"{item['name']} + {m['meal']}",
                    "total": m["total_price"],
                    "order_link": item["order_link"],
                })
            for fb in item.get("frequently_bought_together", []):
                if normalize(fb["name"]) == normalize(item["name"]):
                    continue
                options.append({
                    "type": "add_side",
                    "description": f"{item['name']} ({rs(price)}) + {fb['name']} ({rs(fb['price'])})",
                    "total": price + fb["price"],
                    "order_link": item["order_link"],
                    "side_order_link": self.link_for(fb["name"]),
                })
            if budget is not None:
                options = [o for o in options if o["total"] <= budget]
            options.sort(key=lambda o: o["total"])
            results.append({
                "item": item["name"],
                "category": item["category"],
                "price": price,
                "original_price": item.get("original_price"),
                "order_link": item["order_link"],
                "budget": budget,
                "options": options,
                "note": None if options else (
                    "No meal option fits this budget." if budget is not None
                    else "No meal options listed for this item."),
            })
        return {"results": results}

    def meals_under_budget(self, budget: float, category: str | None = None, limit: int = 8) -> list[dict]:
        """Best Make-It-A-Meal combos (any item) within a budget."""
        items = self.menu
        if category:
            cats = self.match_category(category)
            items = [i for i in items if i["category"] in cats]
        combos = []
        for it in items:
            for m in it.get("make_it_a_meal", []):
                if m["total_price"] <= budget:
                    combos.append({"description": f"{it['name']} + {m['meal']}",
                                   "total": m["total_price"], "order_link": it["order_link"]})
        combos.sort(key=lambda c: -c["total"])
        return combos[:limit]

    # ----------------------------------------------------------- order total
    def order_total(self, items: list[dict]) -> dict:
        """items: [{name, quantity, size}] -> line items, subtotal, unknown items."""
        lines, unknown = [], []
        subtotal = 0
        for entry in items:
            name = str(entry.get("name", "")).strip()
            qty = int(entry.get("quantity") or 1)
            size = entry.get("size")
            item = None
            # "Serious Brim Burger" / "6 pcs wings": pull the size out of the name
            if not size:
                for s_name in ("serious", "standard", "6 pcs", "3 pcs"):
                    if s_name in normalize(name):
                        cand = self.find_item(normalize(name).replace(s_name, "").strip())
                        if cand and cand.get("sizes"):
                            item, size = cand, s_name
                            break
            if item is None:
                item = self.find_item(name)
            if item is None:
                unknown.append(name)
                continue
            price, original, size_name = self._price_for_size(item, size)
            if price is None:
                unknown.append(f"{name} (size '{size}' not found; sizes: "
                               f"{', '.join(s['name'] for s in item['sizes'])})")
                continue
            line_total = price * qty
            subtotal += line_total
            lines.append({
                "name": item["name"], "size": size_name, "quantity": qty,
                "unit_price": price, "original_unit_price": original,
                "line_total": line_total, "order_link": item["order_link"],
            })
        return {
            "lines": lines,
            "subtotal": subtotal,
            "unknown_items": unknown,
            "note": "Delivery fee is extra (usually Rs 100).",
            "advance_payment_needed": subtotal > 10000,
        }

    # ----------------------------------------------------------- branches
    def find_branches(self, query: str | None = None) -> list[dict]:
        """Branches by city or by (part of) the branch name. Empty query = all."""
        if not query:
            return self.branches
        q = normalize(query)
        by_city = [b for b in self.branches if normalize(b["city"]) == q]
        if by_city:
            return by_city
        return [b for b in self.branches if q in normalize(b["name"]) or q in normalize(b["address"])]

    def check_delivery(self, area: str, city: str | None = None) -> dict:
        """Which branches deliver to an area. Exact match first; near misses
        only when there is no exact match."""
        q = normalize(area)
        branches = self.branches
        if city:
            branches = [b for b in branches if normalize(b["city"]) == normalize(city)] or branches

        def hit(b, a):
            return {"area": a, "branch": b["name"], "city": b["city"], "phone": b["phone"],
                    "delivery_fee": b["delivery_fee"], "minimum_order": b["minimum_order"],
                    "delivery_time_minutes": b["delivery_time_minutes"]}

        exact = [hit(b, a) for b in branches for a in b["delivers_to"] if normalize(a) == q]
        if exact:
            return {"query": area, "covered": True, "matches": exact}

        near = []
        for b in branches:
            for a in b["delivers_to"]:
                na = normalize(a)
                if (len(q) >= 3 and q in na) or SequenceMatcher(None, q, na).ratio() >= 0.8:
                    near.append(hit(b, a))
        return {
            "query": area,
            "covered": False,
            "near_matches": near[:10],
            "note": ("No exact match. These areas are similar; ask the customer which one they mean."
                     if near else "This area is not in any branch's delivery list."),
        }

    # ----------------------------------------------------------- deals
    def deal_status(self, item: dict, now: datetime) -> str | None:
        ok = is_available_now(item.get("available"), now)
        if ok is None:
            return None
        return "available now" if ok else f"not available right now (only {item['available']})"


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Load the knowledge base once and reuse it."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB
```

**Check it:**

```powershell
python -c "from app.knowledge import get_kb; kb=get_kb(); i=kb.find_item('mozzy meltdown beef'); print(i['price'], i['original_price']); print(kb.check_delivery('Askari 11')['matches'][0]['branch'])"
```

Expected: `1139 1265` and `Brim Burgers DHA Phase 6`.

---

## Step 3 — Offline tests (`tests/test_knowledge.py`)

No API keys needed. Covers lookups, categories, filters, meal totals, order totals, branches,
delivery areas, deal times and BM25 — including all five expected checks from the brief.

**`tests/test_knowledge.py`**

```python
"""Offline tests for the knowledge layer. No API keys needed.

Run:  python -m pytest -q
"""

from datetime import datetime

import pytest

from app.knowledge import KnowledgeBase, is_available_now, normalize


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


# ------------------------------------------------------------- loading
def test_loads_data(kb):
    assert kb.brand["name"] == "Brim Burgers"
    assert len(kb.menu) >= 100
    assert len(kb.branches) == 15
    assert all(i.get("order_link") for i in kb.menu)


def test_normalize():
    assert normalize("Nash'd Tenders (4 Pcs)") == "nashd tenders 4 pcs"


# ------------------------------------------------------------- lookups
def test_exact_lookup_mozzy(kb):
    item = kb.find_item("Mozzy Meltdown Beef")
    assert item["price"] == 1139
    assert item["original_price"] == 1265


def test_fuzzy_lookup_nashd(kb):
    item = kb.find_item("nashd tenders")
    assert item["name"] == "Nash'd Tenders (4 Pcs)"


def test_typo_lookup(kb):
    assert kb.find_item("mozy meltdown beef")["name"] == "Mozzy Meltdown Beef"


def test_holy_chipotle_returns_both(kb):
    found = kb.find_items("Holy Chipotle")
    cats = sorted(i["category"] for i in found)
    assert cats == ["Lil Wrapstars", "Wraps"]


def test_unknown_item(kb):
    assert kb.find_items("pizza margherita") == []


# ------------------------------------------------------------- categories
def test_category_aliases(kb):
    assert set(kb.match_category("burgers")) == {"Beef Burgers", "Chicken Burger"}
    assert kb.match_category("shakes") == ["Sips & Scoops"]
    assert "Beverages" in kb.match_category("drinks")
    assert kb.match_category("kids") == ["Brim Junior"]
    assert kb.match_category("wings") == ["We Winged It"]
    assert "Brim Off Hour Deals" in kb.match_category("deals")


def test_filter_cheapest_burgers(kb):
    res = kb.filter_items(category="burgers", sort="cheapest")
    prices = [i["price"] for i in res["items"]]
    assert prices == sorted(prices)
    assert res["items"][0]["name"] == "Brim Burger"


def test_filter_max_price(kb):
    res = kb.filter_items(category="shakes", max_price=600)
    assert res["count"] > 0
    assert all(i["price"] <= 600 for i in res["items"])


def test_filter_most_expensive(kb):
    res = kb.filter_items(category="beef burgers", sort="most_expensive")
    assert res["items"][0]["name"] == "Brim Maniac"


def test_filter_keyword(kb):
    res = kb.filter_items(keyword="jalapeno")
    assert res["count"] > 0


def test_unknown_category(kb):
    assert "error" in kb.filter_items(category="sushi")


# ------------------------------------------------------------- meals
def test_meal_mozzy_classic(kb):
    res = kb.meal_options("Mozzy Meltdown Beef")["results"][0]
    meal = next(o for o in res["options"] if o["description"].endswith("Classic Fries & Soft Drink"))
    assert meal["total"] == 1499


def test_meal_side_totals(kb):
    res = kb.meal_options("Mozzy Meltdown Beef")["results"][0]
    side = next(o for o in res["options"] if "Oreo Shake" in o["description"])
    assert side["total"] == 1139 + 590
    assert side["side_order_link"].endswith("oreo-shake-1461483")


def test_meal_budget_filter(kb):
    res = kb.meal_options("Mozzy Meltdown Beef", budget=1500)["results"][0]
    assert res["options"]
    assert all(o["total"] <= 1500 for o in res["options"])


def test_meal_budget_too_low(kb):
    res = kb.meal_options("Brim Maniac", budget=1000)["results"][0]
    assert res["options"] == []
    assert res["note"]


def test_brim_box_chicken_link_resolves(kb):
    # data says "Brim Box Chicken", menu says "Chicken Brim Box"
    assert kb.link_for("Brim Box Chicken").endswith("chicken-brim-box-1461497")


# ------------------------------------------------------------- order totals
def test_order_total_expected(kb):
    res = kb.order_total([
        {"name": "Brim Burger", "quantity": 2, "size": "Serious"},
        {"name": "Oreo Shake", "quantity": 1},
    ])
    assert res["subtotal"] == 2958
    assert res["unknown_items"] == []


def test_order_total_size_in_name(kb):
    res = kb.order_total([{"name": "Serious Brim Burger", "quantity": 2},
                          {"name": "oreo shake", "quantity": 1}])
    assert res["subtotal"] == 2958


def test_order_total_unknown_item(kb):
    res = kb.order_total([{"name": "Oreo Shake", "quantity": 1},
                          {"name": "Chicken Biryani", "quantity": 1}])
    assert res["subtotal"] == 590
    assert res["unknown_items"] == ["Chicken Biryani"]


def test_order_total_bad_size(kb):
    res = kb.order_total([{"name": "Brim Burger", "quantity": 1, "size": "Mega"}])
    assert res["subtotal"] == 0
    assert len(res["unknown_items"]) == 1


# ------------------------------------------------------------- branches
def test_branches_by_city(kb):
    names = [b["name"] for b in kb.find_branches("Islamabad")]
    assert len(names) == 3


def test_branch_by_name(kb):
    assert kb.find_branches("johar")[0]["name"] == "Brim Burgers Johar Town"


def test_delivery_askari_11(kb):
    res = kb.check_delivery("Askari 11")
    assert res["covered"] is True
    assert {m["branch"] for m in res["matches"]} == {"Brim Burgers DHA Phase 6"}
    assert res["matches"][0]["city"] == "Lahore"


def test_delivery_clifton_not_covered(kb):
    res = kb.check_delivery("Clifton")
    assert res["covered"] is False
    assert res["near_matches"] == []


def test_delivery_repeated_area_shows_city(kb):
    res = kb.check_delivery("Model Town")
    cities = {m["city"] for m in res["matches"]}
    assert {"Lahore", "Multan", "Sargodha"} <= cities


def test_delivery_near_miss(kb):
    res = kb.check_delivery("Askari")
    assert res["covered"] is False
    assert res["near_matches"]


# ------------------------------------------------------------- deal times
def test_deal_time_window():
    rule = "Mon-Thu, 12:00 PM - 6:00 PM"
    assert is_available_now(rule, datetime(2026, 10, 26, 14, 0)) is True    # Monday 2 PM
    assert is_available_now(rule, datetime(2026, 10, 26, 19, 0)) is False   # Monday 7 PM
    assert is_available_now(rule, datetime(2026, 10, 30, 14, 0)) is False   # Friday


def test_overnight_window():
    rule = "11:00 AM - 3:00 AM"
    assert is_available_now(rule, datetime(2026, 10, 26, 1, 0)) is True
    assert is_available_now(rule, datetime(2026, 10, 26, 5, 0)) is False


# ------------------------------------------------------------- BM25
def test_bm25_search(kb):
    top = kb.search("Askari 11 delivery")[0]
    assert top["type"] == "branch"
    assert "DHA Phase 6" in top["title"]
    assert kb.search("advance payment")[0]["type"] == "brand"
```

**Check it:**

```powershell
python -m pytest -q
```

Expected: `31 passed`. Don't move on until this is green.

---

## Step 4 — LangChain tools (`app/tools.py`)

Seven tools the LLM can call. Notes:

- Decorator order is `@tool` (outermost) then `@track(name=..., type="tool")`. Opik wraps the
  plain function; LangChain reads its signature and type hints to build the schema the LLM sees.
- **No** `from __future__ import annotations` in this file — it turns type hints into strings
  and breaks the schema.
- Docstrings are written for the LLM: they say when to use the tool and show example arguments.
- Tools return short, ready-to-quote text with prices already formatted (`Rs 1,139 (was Rs 1,265)`),
  totals labelled `TOTAL`, deal status (`available now` / `not available right now`) and order links.
- `calculate_order_total` uses a Pydantic `OrderItem` model so `items` gets a proper JSON schema.

**`app/tools.py`**

```python
"""LangChain tools for the Brim Burgers agent.

Decorator order matters:
    @tool                                   <- LangChain, outermost
    @track(name="...", type="tool")         <- Opik, wraps the plain function
    def fn(...)

Do NOT add `from __future__ import annotations` here: @tool reads the real
type hints to build the argument schema the LLM sees.
"""

import json
from typing import Optional

from langchain_core.tools import tool
from opik import track
from pydantic import BaseModel, Field

from app.config import pk_now
from app.knowledge import get_kb, rs

kb = get_kb()


# ---------------------------------------------------------------------------
# Formatting helpers (prices are formatted here, not by the LLM)
# ---------------------------------------------------------------------------


def _price_text(price, original=None) -> str:
    if original and original > price:
        return f"{rs(price)} (was {rs(original)})"
    return rs(price)


def _item_line(item: dict) -> str:
    line = f"- {item['name']} [{item['category']}]: {_price_text(item['price'], item.get('original_price'))}"
    if item.get("sizes"):
        line += " | sizes: " + ", ".join(
            f"{s['name']} {_price_text(s['price'], s.get('original_price'))}" for s in item["sizes"])
    status = kb.deal_status(item, pk_now())
    if status:
        line += f" | {status}"
    return line + f" | order: {item['order_link']}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
@track(name="list_menu", type="tool")
def list_menu(category: Optional[str] = None, max_price: Optional[int] = None,
              keyword: Optional[str] = None, sort: str = "cheapest", limit: int = 15) -> str:
    """List menu items, optionally filtered and sorted by price.

    Args:
        category: Category or alias, e.g. "burgers", "beef burgers", "chicken burgers",
            "wraps", "hot dogs", "fries", "loaded fries", "appetizers", "wings", "shakes",
            "sundaes", "desserts", "drinks", "coolers", "kids", "deals", "naked box",
            "build your own", "dips". Leave empty for the whole menu.
        max_price: Only items costing this many PKR or less.
        keyword: Word that must appear in the name/description, e.g. "spicy", "cheese", "jalapeno".
        sort: "cheapest" (default) or "most_expensive".
        limit: Max items to return (default 15).
    """
    res = kb.filter_items(category=category, max_price=max_price, keyword=keyword,
                          sort=sort or "cheapest", limit=limit or 15)
    if "error" in res:
        return f"{res['error']} Available categories: {', '.join(res['categories'])}"
    if not res["items"]:
        return "No items match these filters."
    head = f"{res['count']} item(s) found"
    if res["count"] > len(res["items"]):
        head += f", showing {len(res['items'])}"
    return head + ":\n" + "\n".join(_item_line(i) for i in res["items"])


@tool
@track(name="get_item_details", type="tool")
def get_item_details(name: str) -> str:
    """Full details for a menu item by name (fuzzy match, typos are fine): price, old price,
    description, sizes, choices, Make It A Meal prices, deal timing and order link.
    Several items can share a name (e.g. "Holy Chipotle" wrap and Lil Wrapstars) - all are returned.

    Args:
        name: Item name as the customer wrote it, e.g. "mozzy meltdown", "nashd tenders".
    """
    found = kb.find_items(name)
    if not found:
        return f"No menu item matches '{name}'. Ask the customer to rephrase or call 042 111 222 746."
    out = []
    now = pk_now()
    for item in found[:4]:
        lines = [f"{item['name']} [{item['category']}]: {_price_text(item['price'], item.get('original_price'))}"]
        if item.get("description"):
            lines.append(f"Description: {item['description']}")
        if item.get("sizes"):
            lines.append("Sizes: " + ", ".join(
                f"{s['name']} {_price_text(s['price'], s.get('original_price'))}" for s in item["sizes"]))
        for group, opts in (item.get("choices") or {}).items():
            lines.append(f"{group}: {', '.join(opts)}")
        if item.get("make_it_a_meal"):
            lines.append("Make It A Meal (total price): " + ", ".join(
                f"{m['meal']} = {rs(m['total_price'])}" for m in item["make_it_a_meal"]))
        if item.get("available"):
            lines.append(f"Available: {item['available']} -> {kb.deal_status(item, now)}")
        lines.append(f"Order link: {item['order_link']}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


@tool
@track(name="suggest_meal", type="tool")
def suggest_meal(item_name: Optional[str] = None, budget: Optional[int] = None,
                 category: Optional[str] = None) -> str:
    """Suggest meals with TOTAL prices (computed in code).

    - With item_name: Make It A Meal options for that item (fries + drink) and the item
      paired with each frequently-bought side, with totals. Optional budget filters them.
    - With only budget (and optional category like "burgers"): the best meal combos
      on the whole menu that fit the budget.

    Args:
        item_name: The main item, e.g. "Mozzy Meltdown Beef".
        budget: Maximum total in PKR, e.g. 1500.
        category: Limit budget search to a category, e.g. "burgers", "wraps", "hot dogs".
    """
    if item_name:
        res = kb.meal_options(item_name, budget)
        if "error" in res:
            return res["error"]
        blocks = []
        for r in res["results"]:
            head = (f"{r['item']} [{r['category']}] alone: {_price_text(r['price'], r['original_price'])}"
                    f" | order: {r['order_link']}")
            if not r["options"]:
                blocks.append(head + f"\n{r['note']}")
                continue
            lines = [head, "Options (TOTAL):"]
            for o in r["options"][:10]:
                line = f"- {o['description']} = TOTAL {rs(o['total'])}"
                if o.get("side_order_link"):
                    line += f" | side link: {o['side_order_link']}"
                lines.append(line)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
    if budget:
        combos = kb.meals_under_budget(budget, category)
        if not combos:
            return f"No Make It A Meal combo fits Rs {budget:,}. Suggest single items with list_menu."
        return f"Meal combos within {rs(budget)} (highest first):\n" + "\n".join(
            f"- {c['description']} = TOTAL {rs(c['total'])} | order: {c['order_link']}" for c in combos)
    return "Give item_name or budget."


class OrderItem(BaseModel):
    name: str = Field(description="Menu item name, e.g. 'Brim Burger'")
    quantity: int = Field(default=1, ge=1, le=50, description="How many")
    size: Optional[str] = Field(default=None, description="Size if the item has sizes: 'Standard', 'Serious', '3 pcs', '6 pcs'")


@tool
@track(name="calculate_order_total", type="tool")
def calculate_order_total(items: list[OrderItem]) -> str:
    """Calculate the total for several items (delivery fee is extra).
    Always use this when the customer wants more than one item or asks "how much for ...".

    Args:
        items: List of {name, quantity, size}. Example:
            [{"name": "Brim Burger", "quantity": 2, "size": "Serious"}, {"name": "Oreo Shake", "quantity": 1}]
    """
    rows = [i.model_dump() if isinstance(i, BaseModel) else dict(i) for i in items]
    res = kb.order_total(rows)
    lines = []
    for ln in res["lines"]:
        size = f" ({ln['size']})" if ln["size"] else ""
        lines.append(f"- {ln['quantity']} x {ln['name']}{size} @ {rs(ln['unit_price'])} = "
                     f"{rs(ln['line_total'])} | order: {ln['order_link']}")
    out = "Order:\n" + ("\n".join(lines) if lines else "(no known items)")
    out += f"\nSUBTOTAL: {rs(res['subtotal'])} (+ delivery fee, usually Rs 100)"
    if res["unknown_items"]:
        out += "\nNot found on the menu: " + "; ".join(res["unknown_items"])
    if res["advance_payment_needed"]:
        out += "\nOrders above Rs 10,000 need 50% advance payment."
    return out


@tool
@track(name="find_branches", type="tool")
def find_branches(city_or_name: Optional[str] = None) -> str:
    """Find Brim Burgers branches in Pakistan by city ("Lahore", "Islamabad", "Faisalabad",
    "Gujranwala", "Multan", "Sargodha") or by part of the branch name ("Johar", "DHA Phase 6").
    Returns address, phone, hours, delivery fee, minimum order and map link.
    Leave empty to list every branch.

    Args:
        city_or_name: City or branch name.
    """
    found = kb.find_branches(city_or_name)
    if not found:
        cities = sorted({b["city"] for b in kb.branches})
        return (f"No branch matches '{city_or_name}'. Branches are in: {', '.join(cities)}. "
                f"Coming soon: {', '.join(kb.brand.get('coming_soon', []))}.")
    out = [
        f"- {b['name']} ({b['city']}): {b['address']} | phone {b['phone']} | {b['hours']} | "
        f"{b['services']} | delivery fee Rs {b['delivery_fee']}, min order Rs {b['minimum_order']}, "
        f"~{b['delivery_time_minutes']} min | map: {b['map']}"
        for b in found
    ]
    if not city_or_name and kb.other_locations:
        out.append("Dine-in only (no online ordering): " + ", ".join(o["name"] for o in kb.other_locations))
    return "\n".join(out)


@tool
@track(name="check_delivery_area", type="tool")
def check_delivery_area(area: str, city: Optional[str] = None) -> str:
    """Check whether Brim Burgers delivers to an area and which branch covers it.
    Exact match first; near misses only if nothing matches exactly.

    Args:
        area: Area / society / sector, e.g. "Askari 11", "DHA Phase 5", "F-7/2".
        city: Optional city to narrow it down (area names repeat across cities).
    """
    res = kb.check_delivery(area, city)
    if res["covered"]:
        return f"YES, '{area}' is covered:\n" + "\n".join(
            f"- {m['branch']} ({m['city']}) | fee Rs {m['delivery_fee']}, min order Rs {m['minimum_order']}, "
            f"~{m['delivery_time_minutes']} min | phone {m['phone']}" for m in res["matches"])
    if res["near_matches"]:
        return (f"No exact match for '{area}'. Similar areas (ask which one they mean):\n" + "\n".join(
            f"- {m['area']} -> {m['branch']} ({m['city']})" for m in res["near_matches"]))
    return f"NO, '{area}' is not in any branch's delivery list. Suggest pickup or calling 042 111 222 746."


@tool
@track(name="search_knowledge", type="tool")
def search_knowledge(query: str) -> str:
    """Keyword (BM25) search over everything: menu items, branches, add-ons and brand info.
    Use it when no other tool fits, e.g. "which burgers have mushrooms", "coolers", "Dolmen Mall".

    Args:
        query: Search words.
    """
    hits = kb.search(query, k=5)
    if not hits:
        return "Nothing found. Suggest calling 042 111 222 746."
    return "\n\n".join(f"[{h['type']}] {h['text']}" for h in hits)


TOOLS = [list_menu, get_item_details, suggest_meal, calculate_order_total,
         find_branches, check_delivery_area, search_knowledge]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


if __name__ == "__main__":
    # quick manual check: python -m app.tools
    print(get_item_details.invoke({"name": "mozzy meltdown beef"}))
    print(calculate_order_total.invoke({"items": [
        {"name": "Brim Burger", "quantity": 2, "size": "Serious"},
        {"name": "Oreo Shake", "quantity": 1}]}))
    print(json.dumps(check_delivery_area.args_schema.model_json_schema(), indent=1)[:300])
```

**Check it (no keys needed):**

```powershell
python -m app.tools
```

You should see the Mozzy Meltdown Beef details and `SUBTOTAL: Rs 2,958`.

---

## Step 5 — The agent (`app/agent.py`)

- `build_system_prompt()` — rules + current Pakistan date/time + brand info + add-on prices,
  rebuilt on every message so the time is always fresh.
- `call_llm()` — one Groq call = one Opik `llm` span. `capture_input=False` stops Opik from
  logging the long system prompt; the span is filled manually with the conversation (without
  the system message), model, `provider="groq"` and token usage from `usage_metadata`.
- `run_tool()` — runs a tool call; if the LLM sent bad arguments, the error text goes back as the
  tool result so the LLM can fix it and retry.
- `answer()` — the `handle_query` trace. Sets `thread_id = session_id`, runs up to 5 tool rounds,
  then forces a final answer if needed. Any Groq failure → polite fallback with the order link
  and phone number, trace tagged `error`.
- Memory — last 10 messages per session in a dict guarded by a lock. Fine for one process;
  move it to Redis when you run several workers or servers.

**`app/agent.py`**

```python
"""Tool-calling agent: Groq LLM + 7 tools, traced with Opik."""

import json
import logging
import threading
from collections import defaultdict, deque

from app.config import GROQ_API_KEY, GROQ_MODEL, MAX_TOOL_ROUNDS, MEMORY_MESSAGES, pk_now  # loads .env first

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from opik import opik_context, track

from app.knowledge import get_kb
from app.tools import TOOLS, TOOLS_BY_NAME

log = logging.getLogger("brim.agent")
kb = get_kb()
brand = kb.brand

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _add_on_text() -> str:
    lines = []
    for group, opts in kb.add_ons.items():
        lines.append(f"- {group}: " + ", ".join(
            f"{o['name']} " + ("free" if o["price"] == 0 else f"+Rs {o['price']}") for o in opts))
    return "\n".join(lines)


def _brand_text() -> str:
    o = brand["ordering"]
    return f"""- Website: {brand['website']}
- Order online: {brand['order_online']}
- Phone orders / customer service: {brand['phone_orders']}
- Complaints & suggestions: {brand['complaints_and_suggestions']}
- Emails: {', '.join(brand['emails'])}
- Android app: {brand['android_app']}
- iPhone app: {brand['iphone_app']}
- Social: {', '.join(brand['social_media'])}
- Opening hours: {brand['opening_hours']}
- How to order: {o['how_to_order']}
- Advance payment: {o['advance_payment']}
- Delivery fee: {o['delivery_fee']}
- Minimum order: {o['minimum_order']}
- Delivery time: {o['delivery_time']}
- Payment methods: {o['payment_methods']}
- Coming soon: {', '.join(brand['coming_soon'])}
- Cities with branches: {', '.join(sorted({b['city'] for b in kb.branches}))} (Pakistan only)"""


BRAND_TEXT = _brand_text()
ADD_ON_TEXT = _add_on_text()


def build_system_prompt() -> str:
    now = pk_now()
    return f"""You are the friendly customer assistant for Brim Burgers, a smash-burger chain in Pakistan.
Current Pakistan date and time: {now.strftime('%A, %d %B %Y, %I:%M %p')} (PKT).

RULES
1. For ANY question about menu, items, prices, deals, meals, branches, timings or delivery areas,
   call a tool first. Never invent items, prices, branches or links. If a tool can't answer,
   give the phone number {brand['phone_orders']}.
2. Show the price the customer pays. If there is an old price, mention it (e.g. "Rs 1,139, was Rs 1,265").
3. Always include the order link when you recommend or describe an item.
4. Meal requests -> suggest_meal and always state the TOTAL. Several items -> calculate_order_total
   and say the delivery fee is extra. Copy prices and totals exactly from tool results; never do maths yourself.
5. Deals have day/time limits. Tool results say "available now" or "not available right now";
   only offer a deal that is available now, otherwise say when it runs.
6. You cannot place, change or cancel orders. Send the order link, the app, or the phone number instead.
7. Keep replies short (a few lines or a short list). Reply in the customer's language:
   English, Urdu, or Roman Urdu.
8. Only talk about Brim Burgers. Politely decline anything else (homework, coding, politics, other restaurants).
9. If the area or item is ambiguous, ask one short question.

BRAND INFO (no tool needed for these)
{BRAND_TEXT}

ADD-ON PRICES (same on every item)
{ADD_ON_TEXT}
"""


FALLBACK_REPLY = (
    "Sorry, I'm having trouble right now. You can order directly at "
    f"{brand['order_online']} or call {brand['phone_orders']}."
)

# ---------------------------------------------------------------------------
# Session memory (in process). For more than one worker/instance, move this to Redis.
# ---------------------------------------------------------------------------

_memory: dict[str, deque] = defaultdict(lambda: deque(maxlen=MEMORY_MESSAGES))
_memory_lock = threading.Lock()


def get_history(session_id: str) -> list[BaseMessage]:
    with _memory_lock:
        return list(_memory[session_id])


def save_turn(session_id: str, user_text: str, reply: str) -> None:
    with _memory_lock:
        _memory[session_id].append(HumanMessage(content=user_text))
        _memory[session_id].append(AIMessage(content=reply))


def reset_session(session_id: str) -> None:
    with _memory_lock:
        _memory.pop(session_id, None)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = None
_llm_with_tools = None


def get_llms():
    global _llm, _llm_with_tools
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY or None,
                        temperature=0.2, max_retries=2, timeout=30)
        _llm_with_tools = _llm.bind_tools(TOOLS)
    return _llm, _llm_with_tools


def _msg_to_dict(m: BaseMessage) -> dict:
    d = {"role": m.type, "content": m.content}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        d["tool_call_id"] = m.tool_call_id
    return d


@track(name="groq_chat", type="llm", capture_input=False, capture_output=False)
def call_llm(messages: list[BaseMessage], use_tools: bool = True) -> AIMessage:
    """One LLM call = one Opik 'llm' span with model, provider and token usage."""
    llm, llm_with_tools = get_llms()
    runnable = llm_with_tools if use_tools else llm
    response = runnable.invoke(messages)

    usage = response.usage_metadata or {}
    opik_context.update_current_span(
        input={"messages": [_msg_to_dict(m) for m in messages if not isinstance(m, SystemMessage)]},
        output=_msg_to_dict(response),
        model=GROQ_MODEL,
        provider="groq",
        usage={
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )
    return response


def run_tool(call: dict) -> ToolMessage:
    """Run one tool call. Bad arguments come back to the LLM as text so it can retry."""
    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        content = f"Error: unknown tool '{call['name']}'. Available: {', '.join(TOOLS_BY_NAME)}."
    else:
        try:
            content = str(tool.invoke(call["args"]))
        except Exception as e:  # validation errors, wrong types, etc.
            log.warning("Tool %s failed: %s", call["name"], e)
            content = f"Error calling {call['name']} with {json.dumps(call['args'])}: {e}. Fix the arguments and try again."
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@track(name="handle_query")
def answer(message: str, session_id: str) -> str:
    """Answer one customer message. One Opik trace per call, grouped by session."""
    opik_context.update_current_trace(thread_id=session_id, metadata={"model": GROQ_MODEL})

    messages: list[BaseMessage] = [SystemMessage(content=build_system_prompt())]
    messages += get_history(session_id)
    messages.append(HumanMessage(content=message))

    try:
        reply = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = call_llm(messages)
            messages.append(response)
            if not response.tool_calls:
                reply = response.content
                break
            for call in response.tool_calls:
                messages.append(run_tool(call))
        if reply is None:
            # Out of tool rounds: ask for a final answer without tools.
            messages.append(HumanMessage(content="Answer the customer now using the tool results above."))
            reply = call_llm(messages, use_tools=False).content
        reply = (reply or "").strip() or FALLBACK_REPLY
    except Exception as e:  # Groq down, rate limit, bad key, network...
        log.exception("LLM failure")
        opik_context.update_current_trace(tags=["error"], metadata={"error": f"{type(e).__name__}: {e}"})
        return FALLBACK_REPLY

    save_turn(session_id, message, reply)
    return reply


if __name__ == "__main__":
    # quick manual check: python -m app.agent
    import opik

    print(answer("How much is the Mozzy Meltdown Beef?", "cli-test"))
    opik.flush_tracker()
```

**Check it (needs `GROQ_API_KEY` in `.env`):**

```powershell
python -m app.agent
```

You should get a short answer with Rs 1,139 (was Rs 1,265) and the order link.

Now add Opik: sign up at https://www.comet.com/opik, copy your API key and workspace name into
`.env`, run the command again, and open project **brim-burgers-chatbot**. You should see a
`handle_query` trace with `groq_chat` and `get_item_details` spans inside it.

---

## Step 6 — API and chat page

### 6.1 `app/main.py`

- `POST /chat` → `{reply, session_id}`. A new `session_id` is created if none is sent.
- The agent is blocking (HTTP call to Groq), so it runs in a threadpool via `run_in_threadpool`.
- CORS origins come from `ALLOWED_ORIGINS` in `.env`.
- The `lifespan` hook loads the data at startup and flushes Opik traces on shutdown.

**`app/main.py`**

```python
"""FastAPI server for the Brim Burgers chatbot.

Run:  uvicorn app.main:app --reload
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import ALLOWED_ORIGINS, GROQ_MODEL  # loads .env first

import opik
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import answer, reset_session
from app.knowledge import get_kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_kb()  # load and index the data once at startup
    yield
    opik.flush_tracker()  # send any traces still in the queue


app = FastAPI(title="Brim Burgers Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex
    # answer() is blocking (HTTP call to Groq), so keep it off the event loop
    reply = await run_in_threadpool(answer, req.message.strip(), session_id)
    return ChatResponse(reply=reply, session_id=session_id)


@app.post("/reset")
async def reset(req: ResetRequest) -> dict:
    reset_session(req.session_id)
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    kb = get_kb()
    return {"status": "ok", "model": GROQ_MODEL, "menu_items": len(kb.menu), "branches": len(kb.branches)}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
```

### 6.2 `static/index.html`

A simple chat page. It keeps the session id in `sessionStorage`, turns links into clickable
links, escapes HTML (so nothing from the reply can inject script), and has a "New chat" button
that calls `/reset`.

**`static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brim Burgers Assistant</title>
<style>
  :root { --brand:#e4002b; --bg:#f6f4f1; --card:#fff; --text:#1d1d1d; --muted:#777; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--text);
         height:100vh; display:flex; flex-direction:column; }
  header { background:var(--brand); color:#fff; padding:14px 18px; display:flex; justify-content:space-between; align-items:center; }
  header h1 { margin:0; font-size:18px; }
  header button { background:transparent; color:#fff; border:1px solid #fff8; border-radius:6px; padding:6px 10px; cursor:pointer; }
  #chat { flex:1; overflow-y:auto; padding:16px; max-width:760px; width:100%; margin:0 auto; }
  .msg { padding:10px 14px; border-radius:14px; margin:8px 0; max-width:85%; white-space:pre-wrap; line-height:1.45; word-wrap:break-word; }
  .user { background:var(--brand); color:#fff; margin-left:auto; border-bottom-right-radius:4px; }
  .bot { background:var(--card); border-bottom-left-radius:4px; box-shadow:0 1px 2px #0001; }
  .bot a { color:var(--brand); }
  .typing { color:var(--muted); font-style:italic; }
  form { display:flex; gap:8px; padding:12px; max-width:760px; width:100%; margin:0 auto; }
  input { flex:1; padding:12px 14px; border:1px solid #ccc; border-radius:24px; font-size:15px; }
  form button { background:var(--brand); color:#fff; border:0; border-radius:24px; padding:0 20px; font-size:15px; cursor:pointer; }
  form button:disabled { opacity:.5; }
</style>
</head>
<body>
<header>
  <h1>🍔 Brim Burgers Assistant</h1>
  <button id="reset" type="button">New chat</button>
</header>
<div id="chat"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Ask about the menu, deals, branches, delivery..." maxlength="1000">
  <button id="send">Send</button>
</form>
<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
let sessionId = null;
try { sessionId = sessionStorage.getItem("brim_session"); } catch (e) {}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function linkify(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s)<]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
}
function add(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  if (who === "bot") div.innerHTML = linkify(text); else div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

add("Hi! 👋 I can help with the Brim Burgers menu, meal deals, prices, branches and delivery areas. What would you like?", "bot");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  add(text, "user");
  input.value = "";
  send.disabled = true;
  const typing = add("Typing...", "bot typing");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text, session_id: sessionId}),
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    sessionId = data.session_id;
    try { sessionStorage.setItem("brim_session", sessionId); } catch (e) {}
    add(data.reply, "bot");
  } catch (err) {
    typing.remove();
    add("Sorry, something went wrong. Please call 042 111 222 746.", "bot");
  } finally {
    send.disabled = false;
    input.focus();
  }
});

document.getElementById("reset").addEventListener("click", async () => {
  if (sessionId) {
    fetch("/reset", {method: "POST", headers: {"Content-Type": "application/json"},
                     body: JSON.stringify({session_id: sessionId})}).catch(() => {});
  }
  sessionId = null;
  try { sessionStorage.removeItem("brim_session"); } catch (e) {}
  chat.innerHTML = "";
  add("New chat started. How can I help?", "bot");
});
</script>
</body>
</html>
```

**Check it:**

```powershell
uvicorn app.main:app --reload
```

- http://localhost:8000/health → `{"status":"ok", ... "menu_items":105, "branches":15}`
- http://localhost:8000 → chat page. Try: "2 serious brim burgers and an oreo shake" → Rs 2,958.
- http://localhost:8000/docs → FastAPI's built-in API tester.

From a second PowerShell window:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -ContentType "application/json" -Body '{"message":"Do you deliver to Clifton?"}'
```

---

## Step 7 — Test questions and runner

### 7.1 `scripts/questions.txt`

50 questions in 12 conversations. A blank line starts a new session, so follow-ups like
"What's in it?" test the memory. Lines starting with `#` are comments.

**`scripts/questions.txt`**

```text
# 50 test questions. A blank line starts a new conversation (new session).
# Lines starting with # are ignored.

# --- Menu & prices
Hi! What burgers do you have?
Which one is the cheapest?
How much is the Mozzy Meltdown Beef?
What's in it?
Do you have a bigger size of the Brim Burger?

# --- Fuzzy names
nashd tenders price?
Tell me about the Holy Chipotle
mozy meltdown chiken kitne ka hai?

# --- Meals & budgets
I want the Mozzy Meltdown Beef as a meal with classic fries and a soft drink. What's the total?
What about with curly fries and a cooler instead?
Suggest a meal with Oklahoma Onion Smash under Rs 1,400
I have Rs 1,200, what burger meal can I get?
Something for my kid under Rs 700?

# --- Order totals
How much for 2 Serious Brim Burgers and 1 Oreo Shake?
Add 2 Classic Fries to that
And a chicken biryani too
Can you place this order for me?

# --- Deals & timing
What deals do you have right now?
Is the Double Down Deal available on Friday?
What is the Spicy Saturdays Offer and when can I get it?

# --- Filters
Show me shakes under Rs 600
What's the most expensive item on the menu?
Which items have jalapenos?
Do you have wings? What flavours?
What desserts do you have?

# --- Add-ons
How much is extra cheese on a burger?
How much does an extra patty cost?
Can I upgrade my meal fries to garlic mayo fries? How much extra?

# --- Branches
Where are your branches in Islamabad?
Do you have a branch in Multan?
What's the address and phone number of the Johar Town branch?
Are you in Karachi?
What time does the Gulberg branch close?

# --- Delivery areas
Do you deliver to Askari 11?
Which branch delivers to F-7/2?
Do you deliver to Clifton?
Do you deliver to Askari?
Model Town delivery? I'm in Lahore.
What is the delivery fee and minimum order?

# --- Contact & ordering
What's your phone number?
How can I complain about my order?
Do you have an app?
Can I pay online?
My order is above Rs 12,000, anything I should know?

# --- Roman Urdu
Assalam o alaikum, sab se sasta burger konsa hai?
Mujhe 1500 mein meal chahiye, kya milega?
DHA Phase 5 mein delivery hoti hai?

# --- Off-topic
Can you help me with my Python homework?
Which is better, Brim or McDonald's?
What's the weather in Lahore today?
```

### 7.2 `scripts/ask_questions.py`

**`scripts/ask_questions.py`**

```python
"""Run the test questions through the agent and print the answers.

    python scripts/ask_questions.py                 # all questions
    python scripts/ask_questions.py --limit 10      # first 10 only
    python scripts/ask_questions.py --pause 3       # slower, for rate limits

A blank line in questions.txt starts a new conversation (new session id),
so follow-up questions share memory. Lines starting with # are skipped.
Every answer shows up as a trace in Opik, grouped by session (thread).
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import opik  # noqa: E402

from app.agent import answer  # noqa: E402

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.txt"


def load_conversations(path: Path) -> list[list[str]]:
    conversations, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if not line:
            if current:
                conversations.append(current)
                current = []
            continue
        current.append(line)
    if current:
        conversations.append(current)
    return conversations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="stop after N questions")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between calls (Groq free tier)")
    args = parser.parse_args()

    conversations = load_conversations(QUESTIONS_FILE)
    total = sum(len(c) for c in conversations)
    print(f"{total} questions in {len(conversations)} conversations\n")

    asked = 0
    try:
        for c_idx, convo in enumerate(conversations, 1):
            session_id = f"test-{c_idx}-{uuid.uuid4().hex[:6]}"
            print(f"===== Conversation {c_idx} ({session_id}) =====")
            for q in convo:
                asked += 1
                started = time.perf_counter()
                reply = answer(q, session_id)
                took = time.perf_counter() - started
                print(f"\n[{asked}/{total}] Q: {q}\nA ({took:.1f}s): {reply}\n")
                if args.limit and asked >= args.limit:
                    return
                time.sleep(args.pause)
    finally:
        print("Flushing Opik traces...")
        opik.flush_tracker()


if __name__ == "__main__":
    main()
```

**Check it:**

```powershell
python scripts/ask_questions.py --limit 5     # quick run
python scripts/ask_questions.py               # all 50 (about 3-5 minutes)
```

Then review in Opik → **Threads**: one thread per conversation. Things to check:

- Prices in answers match the tool output exactly.
- Every recommended item has an order link.
- Deals offered only when "available now".
- "chicken biryani" is reported as not on the menu.
- Roman Urdu questions get Roman Urdu answers; off-topic questions are declined.

If you hit Groq rate limits, use `--pause 4`.

---

## Step 8 — Docker and docs

**`Dockerfile`**

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY static ./static

# run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

# 1 worker: session memory lives in process (use Redis before scaling out)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
```

**`.dockerignore`**

```text
.env
.venv
__pycache__
.pytest_cache
.git
tests
scripts
*.md
```

```powershell
docker build -t brim-chatbot .
docker run -p 8000:8000 --env-file .env brim-chatbot
```

One worker, because session memory lives in the process. The `README.md` in the project covers
setup, running, Opik, the API and Docker for whoever takes over the project.

---

## Expected checks (all covered by the tests)

| Question | Expected |
|---|---|
| Mozzy Meltdown Beef | Rs 1,139 (was Rs 1,265) |
| + Classic Fries & Soft Drink | Rs 1,499 total |
| 2 Serious Brim Burgers + 1 Oreo Shake | Rs 2,958 |
| Askari 11 | Delivered by DHA Phase 6 (Lahore) |
| Clifton | Not covered |

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: app` when running tests | Run from the project folder; check `pytest.ini` exists |
| `.venv\Scripts\activate` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Bot always replies "Sorry, I'm having trouble…" | Wrong/missing `GROQ_API_KEY`; the real error is in the server log and the Opik trace (tag `error`) |
| No traces in Opik | Check `OPIK_API_KEY` and `OPIK_WORKSPACE`; traces are sent in the background, so wait a few seconds |
| 429 from Groq | Free-tier rate limit; slow down with `--pause 4` |
| Browser CORS error from the client site | Add that site's origin to `ALLOWED_ORIGINS` |
