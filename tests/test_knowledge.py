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
