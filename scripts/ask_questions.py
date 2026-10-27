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
