"""Tool-calling agent: Groq LLM (with automatic backup) + 7 tools, traced with Opik."""

import json
import logging
import threading
from collections import defaultdict, deque

from app.config import GROQ_MODEL, MAX_TOOL_ROUNDS, MEMORY_MESSAGES, pk_now  # loads .env first

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from opik import opik_context, track

from app.guard import check_reply, correction_prompt, strip_bad_links
from app.knowledge import get_kb
from app.llm_router import LLMRouter, build_providers
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
            f"{o['name']} " + ("(no extra charge)" if o["price"] == 0 else f"+Rs {o['price']}") for o in opts))
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
7. Keep replies short (a few lines or a short list).
8. LANGUAGE: reply in the same language and script as the customer's LAST message.
   - Roman Urdu in (e.g. "kitne ka hai?", "delivery hoti hai?") -> Roman Urdu out
     (e.g. "Ji haan, Askari 11 mein delivery hoti hai."). Item names, prices and links stay as they are.
   - Urdu script in -> Urdu script out. English in -> English out.
9. Only talk about Brim Burgers. Politely decline anything else (homework, coding, politics, other restaurants).
10. If the area or item is ambiguous, ask one short question.
11. Give the phone number only when it helps (tool can't answer, complaints, phone orders),
    not at the end of every reply.
12. Only mention an old price ("was ...") if it appears in the tool result.
13. Copy every link EXACTLY and in full from the tool result. Never shorten, edit or rebuild a URL.
14. Describe items only with the description and choices in the tool result. Don't add ingredients,
    options, sides or drinks that aren't listed there. Nothing is free unless a tool result says so.
15. VEGETARIAN: Brim has NO vegetarian burgers, wraps or boxes. Every burger, wrap, hot dog, box and
    tender has beef or chicken (Smashed Shrooms is a BEEF burger with mushrooms on top). For a
    vegetarian customer, say this clearly and only suggest meat-free sides: fries, curly fries,
    onion rings, mozzarella sticks/bites, shakes, sundaes, churros, drinks. Never guess.
16. HEALTHY / LIGHTER: the only lighter option is the Naked Box (a burger turned into a salad box,
    no bun). Don't call anything else healthy, low-fat or protein-rich.
17. NEVER AGREE JUST BECAUSE THE CUSTOMER SAID IT. If the customer states a price, a size, a free
    item, a discount or any other fact, check it with a tool first. If it's wrong, politely correct it
    ("Actually, ..."). Only offer sizes that the tool result lists; if an item has no sizes, say so.
18. Item names: use only names that appear in a tool result in this conversation. Never make up a
    menu item, combo or "meal" name.
19. DRINKS IN MEALS: a Make It A Meal price already INCLUDES the fries and the drink. Say "the meal
    (fries + drink) is Rs X"; never call the drink "free". A burger on its own comes with no drink.
    Only list drink choices that appear in the add-on list above.
20. Chicken vs beef: if the customer asks for chicken, show only items whose category is chicken
    (Chicken Burger, wraps, tenders...); for beef, only Beef Burgers. Check the [category] in tool results.
21. FORMAT: no tables. Use short lines or a simple "- " list.
22. "Show the whole menu": don't refuse. Give the category list (burgers, wraps, hot dogs, fries,
    loaded fries, appetizers, wings, Naked Box, kids, shakes & sundaes, desserts, drinks, deals)
    and ask which one to show.

BRAND INFO (no tool needed for these)
{BRAND_TEXT}

ADD-ON PRICES (same on every item). "No extra charge" means it is a choice INSIDE a meal or deal
whose price already includes it. It does NOT mean free.
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


MEMORY_REPLY_CHARS = 500  # long replies (e.g. a 16-burger list) are shortened in memory to save tokens


def save_turn(session_id: str, user_text: str, reply: str) -> None:
    # The full reply goes to the customer; memory keeps only the start, which is enough
    # for follow-ups like "what's in it?" and keeps every LLM call small (Groq rate limits).
    if len(reply) > MEMORY_REPLY_CHARS:
        reply = reply[:MEMORY_REPLY_CHARS].rsplit("\n", 1)[0] + "\n[...reply shortened...]"
    with _memory_lock:
        _memory[session_id].append(HumanMessage(content=user_text[:500]))
        _memory[session_id].append(AIMessage(content=reply))


def reset_session(session_id: str) -> None:
    with _memory_lock:
        _memory.pop(session_id, None)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_router: LLMRouter | None = None
_router_lock = threading.Lock()


def get_router() -> LLMRouter:
    """Groq main model first, backups after it (see app/llm_router.py)."""
    global _router
    with _router_lock:
        if _router is None:
            _router = LLMRouter(build_providers(), TOOLS)
            log.info("LLM order: %s", " -> ".join(p.name for p in _router.providers))
    return _router


def _msg_to_dict(m: BaseMessage) -> dict:
    d = {"role": m.type, "content": m.content}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        d["tool_call_id"] = m.tool_call_id
    return d


@track(name="llm_chat", type="llm", capture_input=False, capture_output=False)
def call_llm(messages: list[BaseMessage], use_tools: bool = True) -> AIMessage:
    """One LLM call = one Opik 'llm' span with the model that actually answered and its token usage."""
    response, used, skipped = get_router().invoke(messages, use_tools)

    usage = response.usage_metadata or {}
    is_backup = used is not get_router().providers[0]
    opik_context.update_current_span(
        input={"messages": [_msg_to_dict(m) for m in messages if not isinstance(m, SystemMessage)]},
        output=_msg_to_dict(response),
        model=used.model,
        provider=used.provider,
        usage={
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        metadata={"llm": used.name, "fallback": is_backup, "skipped": skipped},
        tags=["fallback"] if is_backup else None,
    )
    if is_backup:
        opik_context.update_current_trace(tags=["fallback"])
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


def run_loop(messages: list[BaseMessage], rounds: int) -> str:
    """Tool-calling loop: call the LLM, run the tools it asks for, repeat until it answers."""
    for _ in range(rounds):
        response = call_llm(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content or ""
        for call in response.tool_calls:
            messages.append(run_tool(call))
    # Out of tool rounds: ask for a final answer without tools.
    messages.append(HumanMessage(content="Answer the customer now using the tool results above."))
    return call_llm(messages, use_tools=False).content or ""


def tool_results(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, ToolMessage)]


@track(name="handle_query")
def answer(message: str, session_id: str) -> str:
    """Answer one customer message. One Opik trace per call, grouped by session."""
    opik_context.update_current_trace(thread_id=session_id, metadata={"model": GROQ_MODEL})

    messages: list[BaseMessage] = [SystemMessage(content=build_system_prompt())]
    messages += get_history(session_id)
    messages.append(HumanMessage(content=message))

    try:
        context_texts = [m.content for m in messages if isinstance(m.content, str)]
        reply = run_loop(messages, MAX_TOOL_ROUNDS)

        # Check the reply against the data; ask for ONE rewrite if it invented something.
        problems = check_reply(reply, message, context_texts, tool_results(messages))
        if problems:
            log.warning("Reply failed checks, asking for a rewrite: %s", problems)
            opik_context.update_current_trace(tags=["corrected"], metadata={"check_problems": problems})
            messages.append(AIMessage(content=reply))
            messages.append(HumanMessage(content=correction_prompt(problems)))
            retry = run_loop(messages, 2)
            still = check_reply(retry, message, context_texts, tool_results(messages))
            if still:
                log.warning("Rewrite still has problems: %s", still)
                opik_context.update_current_trace(tags=["check_failed"], metadata={"check_problems_after": still})
            reply = strip_bad_links(retry)
        reply = reply.strip() or FALLBACK_REPLY
    except Exception as e:  # every provider failed (Groq and backups)
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