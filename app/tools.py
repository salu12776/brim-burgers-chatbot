"""LangChain tools for the Brim Burgers agent.

Decorator order matters:
    @tool                                   <- LangChain, outermost
    @track(name="...", type="tool")         <- Opik, wraps the plain function
    def fn(...)

Do NOT add `from __future__ import annotations` here: @tool reads the real
type hints to build the argument schema the LLM sees.
"""

import json
import re
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
              keyword: Optional[str] = None, sort: str = "cheapest", limit: Optional[int] = None) -> str:
    """List menu items, optionally filtered and sorted by price.

    Args:
        category: Category or alias, e.g. "burgers", "beef burgers", "chicken burgers",
            "wraps", "hot dogs", "fries", "loaded fries", "appetizers", "wings", "shakes",
            "sundaes", "desserts", "drinks", "coolers", "kids", "deals", "naked box",
            "build your own", "dips". Leave empty for the whole menu.
        max_price: Only items costing this many PKR or less.
        keyword: Word that must appear in the name/description, e.g. "spicy", "cheese", "jalapeno".
        sort: "cheapest" (default) or "most_expensive".
        limit: Max items to return (default 30 with a category, 15 without).
    """
    res = kb.filter_items(category=category, max_price=max_price, keyword=keyword,
                          sort=sort or "cheapest", limit=limit or (30 if category else 15))
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
        if re.search(r"\bveg|vegetarian|sabzi", name, re.I):
            return ("Brim has NO vegetarian burgers, wraps or boxes; all of them contain beef or chicken. "
                    "Meat-free options: fries, curly fries, onion rings, mozzarella bites/dipper, shakes, "
                    "sundaes, churros and drinks (use list_menu to show prices).")
        return f"No menu item matches '{name}'. It is NOT on the menu. Say so; don't suggest a made-up item."
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
        category: Limit budget search to a category, e.g. "chicken burgers", "beef burgers", "wraps",
            "hot dogs". ALWAYS pass it when the customer says chicken, beef, wrap, etc.
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
            f"- {c['description']} [{c['category']}] = TOTAL {rs(c['total'])} | order: {c['order_link']}" for c in combos)
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
        orig = ln["original_unit_price"]
        line_total = _price_text(ln["line_total"], orig * ln["quantity"] if orig else None)
        lines.append(f"- {ln['quantity']} x {ln['name']}{size} @ {_price_text(ln['unit_price'], orig)} = "
                     f"{line_total} | order: {ln['order_link']}")
    out = "Order:\n" + ("\n".join(lines) if lines else "(no known items)")
    out += f"\nSUBTOTAL: {rs(res['subtotal'])} (+ delivery fee, usually Rs 100)"
    if res["lines"] and all(ln["original_unit_price"] for ln in res["lines"]):
        before = sum(ln["original_unit_price"] * ln["quantity"] for ln in res["lines"])
        if before > res["subtotal"]:
            out += f" | was {rs(before)}, you save {rs(before - res['subtotal'])}"
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
    other = kb.find_other_locations(city_or_name) if city_or_name else []
    if other:
        found = found or []
        extra = "\n".join(f"- {o['name']}: {o['note']}" for o in other)
        if not found:
            return extra
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
    if other:
        out.append(extra)
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
        cities = sorted({m["city"] for m in res["matches"]})
        if len(cities) > 1 and not city:
            return (f"'{area}' exists in {len(cities)} cities: {', '.join(cities)}. "
                    f"Ask the customer which city before answering.\n" + "\n".join(
                        f"- {m['city']}: {m['branch']}" for m in res["matches"]))
        return f"YES, '{area}' is covered:\n" + "\n".join(
            f"- {m['branch']} ({m['city']}) | fee Rs {m['delivery_fee']}, min order Rs {m['minimum_order']}, "
            f"~{m['delivery_time_minutes']} min | phone {m['phone']}" for m in res["matches"])
    if res["near_matches"]:
        return (f"No exact match for '{area}'. Similar areas (ask which one they mean):\n" + "\n".join(
            f"- {m['area']} -> {m['branch']} ({m['city']})" for m in res["near_matches"]))
    cities = ", ".join(sorted({b["city"] for b in kb.branches}))
    return (f"NO, '{area}' is not in any branch's delivery list. Brim Burgers has branches only in: {cities} "
            f"(coming soon: {', '.join(kb.brand.get('coming_soon', []))}). Do NOT suggest a branch in another "
            f"city. If the customer is in one of these cities, ask for a nearby area or suggest pickup there.")


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