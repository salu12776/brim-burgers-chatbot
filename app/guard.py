"""Reply checker: catches made-up prices, made-up order links and the wrong script
BEFORE a reply reaches the customer. Plain Python, no LLM.

- Every Brim order link in the reply must exist in the knowledge base.
- Every "Rs N" amount in the reply must appear in this turn's tool results, the system
  prompt, the conversation, or be a tool total plus a delivery fee (Rs 100 / Rs 150).
- If the customer wrote in Latin letters (English / Roman Urdu), the reply must not
  switch to Urdu script.

The agent asks the model to rewrite once when a check fails (see agent.answer).
"""

import re

from app.knowledge import get_kb

LINK_RE = re.compile(r"https?://brimburgers\.blinkco\.io/product/[^\s)\]>\"'<]+")
MONEY_RE = re.compile(r"(?:Rs\.?|PKR)\s*([\d][\d,]*)|([\d][\d,]{2,})\s*(?:PKR|rupees|rs\b)", re.I)
ARABIC_RE = re.compile(r"[؀-ۿ]")
DELIVERY_FEES = (100, 150)
FREE_RE = re.compile(r"\bfree\b|\bmuft\b|\bmufat\b|\u0645\u0641\u062a", re.I)

_VALID_LINKS: set[str] | None = None


def valid_links() -> set[str]:
    global _VALID_LINKS
    if _VALID_LINKS is None:
        _VALID_LINKS = {i["order_link"].rstrip("/") for i in get_kb().menu}
    return _VALID_LINKS


def amounts(text: str) -> set[int]:
    out = set()
    for a, b in MONEY_RE.findall(text or ""):
        raw = (a or b).replace(",", "")
        if raw.isdigit():
            out.add(int(raw))
    return out


def all_numbers(text: str) -> set[int]:
    """Every number in a text (tool results also contain bare totals like '= 1,499')."""
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text or "") if n.replace(",", "").isdigit()}


def mostly_urdu_script(text: str) -> bool:
    """True when the reply has real Urdu-script wording (not just one stray character)."""
    return len(ARABIC_RE.findall(text or "")) >= 8


NEGATION_RE = re.compile(r"\b(nahi|nahin|nhi|nai|not|isn'?t|no|never)\b", re.I)


def says_free(text: str) -> bool:
    """True if the reply claims something is free. Ignores "free nahi", "not free", "gluten-free"."""
    for m in FREE_RE.finditer(text or ""):
        if m.start() > 0 and text[m.start() - 1] == "-":
            continue  # gluten-free, sugar-free
        window = text[max(0, m.start() - 15): m.end() + 15]
        if NEGATION_RE.search(window):
            continue
        return True
    return False


def check_reply(reply: str, user_message: str, context_texts: list[str], tool_texts: list[str]) -> list[str]:
    """Return a list of problems (empty = reply is fine)."""
    problems = []

    bad_links = [u for u in LINK_RE.findall(reply) if u.rstrip("/.,") not in valid_links()]
    if bad_links:
        problems.append("These order links do not exist: " + ", ".join(bad_links[:5]))

    known = set()
    for t in context_texts + tool_texts + [user_message]:
        known |= all_numbers(t)
    tool_totals = set()
    for t in tool_texts:
        tool_totals |= all_numbers(t)
    known |= {t + fee for t in tool_totals for fee in DELIVERY_FEES}
    unknown = sorted(a for a in amounts(reply) if a not in known)
    if unknown:
        problems.append("These prices are not in any tool result: " + ", ".join(f"Rs {a:,}" for a in unknown[:6]))

    if says_free(reply) and not any(FREE_RE.search(t) for t in tool_texts):
        problems.append("The reply says something is free, but nothing on the menu is free. A meal price "
                        "already includes its fries and drink; say that instead (e.g. 'the meal with fries "
                        "and a drink is Rs X'). If the customer claimed it is free, politely correct them")

    if not ARABIC_RE.search(user_message or "") and mostly_urdu_script(reply):
        problems.append("The customer wrote in Latin letters but the reply is in Urdu script")
    return problems


def correction_prompt(problems: list[str]) -> str:
    return (
        "Your last reply has mistakes:\n- " + "\n- ".join(problems) +
        "\nRewrite the reply for the customer. Use ONLY items, prices and links that appear in the "
        "tool results (call a tool again if you need to). If something doesn't exist, say so. "
        "Reply in the same language and script as the customer's message (Roman Urdu = Latin letters). "
        "Do not mention this correction."
    )


def strip_bad_links(reply: str) -> str:
    """Last resort: replace unknown order links with the main ordering page."""
    main = get_kb().brand["order_online"]
    return LINK_RE.sub(lambda m: m.group(0) if m.group(0).rstrip("/.,") in valid_links() else main, reply)