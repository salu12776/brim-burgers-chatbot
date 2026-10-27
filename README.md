# Brim Burgers Chatbot

A customer chatbot for Brim Burgers (Pakistan). Customers ask about the menu, get meal
suggestions with total prices, check delivery areas and branches, and get a direct order link
for every item.

**Stack:** Python 3.12 · LangChain (langchain-core) · Groq `llama-3.3-70b-versatile` ·
FastAPI + uvicorn · Opik (Comet cloud) · BM25 keyword retrieval in pure Python (no vector DB) · Docker

## How it works

```
Browser chat page ──POST /chat──► FastAPI (app/main.py)
                                     │  threadpool
                                     ▼
                              Agent (app/agent.py)          ── Opik trace "handle_query"
                              Groq LLM + tool loop (≤5)       ├─ span "groq_chat" (llm, tokens)
                                     │                        └─ span per tool call
                                     ▼
                              7 tools (app/tools.py)
                                     │
                                     ▼
                         Knowledge layer (app/knowledge.py)
                         JSON → lookups + BM25, all prices computed in code
```

| Layer | File | Job |
|---|---|---|
| Knowledge | `app/knowledge.py` | Loads the JSON, fuzzy item lookup, category aliases, filters, meal totals, order totals, branches, delivery check, deal time check, BM25 search. No LLM. |
| Tools | `app/tools.py` | `list_menu`, `get_item_details`, `suggest_meal`, `calculate_order_total`, `find_branches`, `check_delivery_area`, `search_knowledge` |
| Agent | `app/agent.py` | System prompt (PK time, brand info, add-on prices), tool loop, error handling, per-session memory (last 10 messages) |
| API | `app/main.py` | `/chat`, `/reset`, `/health`, `/` (chat page) |

## Setup on Windows (uv)

Open PowerShell in the project folder.

```powershell
# 1. Install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create a Python 3.12 virtual environment and activate it
uv venv --python 3.12
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Create your .env and fill in the keys
copy .env.example .env
notepad .env
```

Keys you need:
- `GROQ_API_KEY`: free at https://console.groq.com/keys
- `OPIK_API_KEY` and `OPIK_WORKSPACE`: sign up at https://www.comet.com/opik, copy the API key
  from your account settings; the workspace is your Comet username/workspace name.
  Without an Opik key the bot still works, tracing is just switched off.

## Run

```powershell
# Offline tests (no keys needed)
python -m pytest -q

# Quick tool check (no keys needed)
python -m app.tools

# Start the server
uvicorn app.main:app --reload
```

Open http://localhost:8000 and chat.

Run the 50 test questions (needs the Groq key; pauses 2 s between calls for the free tier):

```powershell
python scripts/ask_questions.py
python scripts/ask_questions.py --limit 10
```

## What to look at in Opik

Open https://www.comet.com/opik → project **brim-burgers-chatbot**.

- **Traces**: one `handle_query` trace per customer message. Expand it to see each
  `groq_chat` LLM span and every tool span with its input and output.
- **Threads**: traces are grouped by `session_id`, so you can read a whole conversation.
- **Token usage**: each `groq_chat` span carries model, provider `groq` and prompt /
  completion / total tokens.
- **Errors**: Groq failures are tagged `error` on the trace; filter by that tag.
- Check that tool outputs match the reply: prices in the answer should be copied from tool
  results, never calculated by the model.

## API

`POST /chat`
```json
{"message": "How much is the Mozzy Meltdown Beef?", "session_id": "optional"}
```
→
```json
{"reply": "...", "session_id": "3f2c..."}
```
Leave `session_id` empty on the first message and send back the one you receive.

`POST /reset` — `{"session_id": "..."}` clears that conversation's memory.

`GET /health` — `{"status":"ok","model":"...","menu_items":105,"branches":15}`

`GET /` — the chat page.

Call it from the client website (add the site to `ALLOWED_ORIGINS`):
```js
const res = await fetch("https://your-server/chat", {
  method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({message, session_id})
});
```

## Docker

```powershell
docker build -t brim-chatbot .
docker run -p 8000:8000 --env-file .env brim-chatbot
```

The image runs one uvicorn worker. Session memory is kept in process, so before running
several workers or instances, move it to Redis (`get_history` / `save_turn` in `app/agent.py`).

## Expected checks

| Question | Expected |
|---|---|
| Mozzy Meltdown Beef price | Rs 1,139 (was Rs 1,265) |
| …with Classic Fries & Soft Drink | Rs 1,499 total |
| 2 Serious Brim Burgers + 1 Oreo Shake | Rs 2,958 |
| Delivery to Askari 11 | DHA Phase 6 branch, Lahore |
| Delivery to Clifton | Not covered |

All of these are covered by `tests/test_knowledge.py`.

## Updating the data

Replace `data/brim_knowledge_base.json` with a new export (same structure), run
`python -m pytest -q`, and restart the server. Prices, links and branches all come from that file.
