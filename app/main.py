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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import answer, get_router, reset_session
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
    return {"status": "ok", "model": GROQ_MODEL, "menu_items": len(kb.menu), "branches": len(kb.branches),
            "llm": get_router().status()}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# /static/widget.js -> floating chat button for the client's website
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")