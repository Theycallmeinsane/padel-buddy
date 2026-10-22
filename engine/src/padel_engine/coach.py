"""A written coaching summary of a match report, by Claude.

Only runs when ANTHROPIC_API_KEY is set (keep it in the git-ignored .env)
and the ``anthropic`` package is installed (``pip install -e ".[coach]"``).
Only the report's numbers are sent: players are P1, P2, ..., with no names, video
or images. Any failure leaves the summary out; the report never depends on it.
"""

from __future__ import annotations

import json
import logging
import os

from .schemas import MatchReport

log = logging.getLogger(__name__)

SYSTEM = """You are a padel coach writing a short summary of one session for the amateur players in it.
People may rotate in and swap partners: each pair ("P1+P3") is two players who played points together.
You get the numbers as JSON: points, shots by kind, winners and errors, each player's points, partners and
success rate per kind of shot (kept in play or won outright, rated strong, solid or weak once there are
enough), time at the net, each pair's formation (both at the net, split, both back), movement during
points, and rule-based tips.

Write 120 to 200 words of plain prose, no headings or bullet points. Start with how the session went, then
give each player or pair one or two concrete things to work on, grounded in the numbers. Call players by
their numbers (P1, P2, ...).
Winners, errors and shot kinds are estimates from video; don't build big conclusions on small counts.
"""


def summary(report: MatchReport, model: str) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("ANTHROPIC_API_KEY is set but the anthropic package isn't installed; skipping the summary")
        return None

    numbers = report.model_dump(exclude={"coach_summary", "outputs", "engine_version", "shorts"})
    for p in numbers["players"]:
        p.pop("highlights", None)
        for key in ("reel", "winning_shots", "winners_reel"):
            p.pop(key, None)
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "medium"},
            # If the model declines, the API retries on a fallback model inside this call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(numbers)}],
        )
    except anthropic.APIStatusError as e:
        log.warning("Coach summary failed (%s): %s", e.status_code, e.message)
        return None
    except anthropic.APIConnectionError:
        log.warning("Coach summary failed: could not reach the Claude API")
        return None
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None
