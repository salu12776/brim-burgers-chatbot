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
