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


GENERIC_WORDS = {"burger", "burgers", "wing", "wings", "the", "item", "meal", "wala", "wali"}


def _word_match(word: str, name_tokens: set[str]) -> bool:
    """Exact word, or a close spelling for words of 4+ letters (mozy ~ mozzy)."""
    if word in name_tokens:
        return True
    if len(word) < 4:
        return False
    return any(SequenceMatcher(None, word, t).ratio() >= 0.85 for t in name_tokens)


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
        # Generic words customers add ("oklahoma burger") don't have to match,
        # as long as at least one real word is left.
        q_tokens = set(q.split())
        required = q_tokens - GENERIC_WORDS or q_tokens
        scored = []
        for it, n in zip(self.menu, self._norm_names):
            n_tokens = set(n.split())
            # every required word is in the name, allowing small typos (mozy -> mozzy)
            overlap = 1.0 if all(_word_match(w, n_tokens) for w in required) else 0.0
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
                    combos.append({"description": f"{it['name']} + {m['meal']}", "category": it["category"],
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

    def find_other_locations(self, query: str) -> list[dict]:
        """Dine-in-only locations (no online ordering), e.g. Dolmen Mall."""
        q = normalize(query or "")
        return [o for o in self.other_locations if q and q in normalize(o["name"])]

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
        if ok:
            return "AVAILABLE NOW"
        return f"NOT AVAILABLE NOW - it only runs {item['available']} (Pakistan time). Do not offer it now"


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Load the knowledge base once and reuse it."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB