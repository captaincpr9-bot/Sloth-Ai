"""
Sloth Mode — a lazy little assistant that does the boring work for you.

Paste anything messy (an email, a long chat thread, a brain-dump to-do list,
or buggy code). Sloth Mode asks a LOCAL model to classify it, then runs the
matching task automatically:

    reply   -> a ready-to-send reply draft
    summary -> a short summary + bullet list of action items
    plan    -> a clean, prioritised plan (most important first)
    fix     -> a suggested fix with a one-line explanation

Everything runs on your own laptop via Ollama + Llama 3.2.
No API keys. No external services. No data leaves your machine.

Run:  python app.py   ->   open http://localhost:5000
"""

import math
import os
import re
import sys
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "llama3.2:1b"
MAX_CHARS = 30000  # keep one paste to a size a sloth can handle

TASK_LABELS = {
    "reply": "Reply draft",
    "summary": "Summary + action items",
    "plan": "Prioritised plan",
    "fix": "Suggested fix",
}

TONE_INSTRUCTIONS = {
    "ultra-lazy": (
        "\n\nTONE MODIFIER [ULTRA LAZY]: Be ultra concise, blunt, and efficient. "
        "Use minimal words, no filler, straight to the point."
    ),
    "corporate": (
        "\n\nTONE MODIFIER [CORPORATE]: Use professional executive phrasing, polished business etiquette, "
        "diplomatic clarity, and a respectful closing."
    ),
    "detailed": (
        "\n\nTONE MODIFIER [DETAILED]: Provide a comprehensive, in-depth, structured response with step-by-step "
        "reasoning and all relevant nuances covered."
    ),
    "chill": (
        "\n\nTONE MODIFIER [CHILL]: Keep it relaxed, friendly, informal, and stress-free. Sound like a helpful teammate."
    ),
}


class OllamaError(Exception):
    """An error with a friendly, user-facing message and an HTTP status."""

    def __init__(self, message, status=503):
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def call_ollama(prompt, model=None):
    """Send one prompt to the local Ollama model and return its text reply."""
    target_model = model.strip() if (model and isinstance(model, str) and model.strip()) else MODEL
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": target_model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            "Ollama is not running yet. Start Ollama first "
            "(open the Ollama app, or run `ollama serve` in a terminal), "
            f"make sure you have run `ollama pull {target_model}`, then try again."
        )
    except requests.exceptions.Timeout:
        raise OllamaError(
            "Your sloth is taking too long — the model may still be "
            "loading. Wait a moment and try again.",
            status=504,
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise OllamaError(
                f"Model '{target_model}' is not downloaded yet. Run "
                f"`ollama pull {target_model}` in a terminal, then try again."
            )
        raise OllamaError(
            f"Ollama could not complete that with '{target_model}'. Check Ollama is running, then try again.",
            status=502,
        )
    except requests.exceptions.RequestException:
        raise OllamaError(
            "Could not reach your local Ollama. Check it is running, "
            "then try again."
        )

    try:
        data = response.json()
    except ValueError:
        raise OllamaError(
            "Ollama sent back something unreadable. Please try again.",
            status=502,
        )
    text = data.get("response", "") if isinstance(data, dict) else ""
    if not isinstance(text, str) or not text.strip():
        raise OllamaError(
            "Ollama returned an empty answer. Please try again.",
            status=502,
        )
    return text.strip()


CLASSIFY_INSTRUCTION = """You are a classifier for a productivity app.

Classify the user's text into EXACTLY ONE of these four types:

- reply   : an email or message that needs a response written back
- summary : a long text or chat thread (many messages / many people) that needs to be summarised
- plan    : a messy to-do list or brain-dump of tasks that needs organising
- fix     : code, an error message, or a bug report that needs a fix

Rules:
- Code, errors or stack traces are always "fix".
- A conversation/chat with multiple messages or speakers is "summary".
- A list of things to do (even messy) is "plan".
- A single message/email asking or telling something is "reply".

Reply with ONLY one word — reply, summary, plan, or fix. No punctuation, no explanation.

User's text:
---
"""

TASK_INSTRUCTIONS = {
    "reply": (
        "Write a ready-to-send reply draft for the email/message below. "
        "Match the tone of the original (friendly if friendly, formal if formal). "
        "Keep it natural and concise. Do not invent facts, dates or commitments — "
        "if something essential is unknown, leave a simple [placeholder]. "
        "Output ONLY the reply draft, nothing else."
    ),
    "summary": (
        "Summarise the long text / chat thread below. Output in exactly this shape:\n"
        "A 2-3 line summary first, then a line that says 'Action items:' followed by "
        "short bullet points (starting with - ) for every task, decision or follow-up "
        "mentioned. Only include action items that are actually in the text — do not "
        "invent any. If there are none, write 'Action items: none found.'"
    ),
    "plan": (
        "Turn the messy to-do list / brain-dump below into a clean, prioritised plan. "
        "Order it MOST IMPORTANT FIRST, judging by urgency, deadlines and impact in the "
        "text. Output a numbered list, one clear action per line, each line short and "
        "starting with a verb. Do not invent deadlines that are not in the text. "
        "Output ONLY the plan."
    ),
    "fix": (
        "The code / bug below has a problem. Suggest a fix. Output the corrected code "
        "(or the corrected part, in a code block) first, then ONE single line starting "
        "with 'Why:' that explains the fix in plain words. Keep the same programming "
        "language and style as the original."
    ),
}


def classify_text(text, model=None):
    """Step 1: ask the model what type of input this is (reply/summary/plan/fix)."""
    raw = call_ollama(CLASSIFY_INSTRUCTION + text[:4000] + "\n---", model=model)
    match = re.search(r"\b(reply|summary|plan|fix)\b", raw.lower())
    if match:
        return match.group(1)
    return _fallback_classify(text)


def _fallback_classify(text):
    lowered = text.lower()
    code_hints = ("def ", "function ", "console.", "import ", "class ",
                  "traceback", "error:", "exception", "{", "};", "=>")
    if sum(hint in lowered for hint in code_hints) >= 2:
        return "fix"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    list_lines = sum(1 for ln in lines if ln.startswith(("-", "*", "•"))
                     or re.match(r"^\d+[.)]\s", ln))
    if lines and list_lines >= max(2, len(lines) // 2):
        return "plan"
    if len(text.split()) > 180:
        return "summary"
    return "reply"


def run_task(task, text, tone="ultra-lazy", model=None):
    """Step 2: run the prompt that matches the detected type with tone modifiers."""
    base_prompt = TASK_INSTRUCTIONS[task]
    tone_addon = TONE_INSTRUCTIONS.get(tone, "")
    prompt = base_prompt + tone_addon + "\n\n---\n" + text + "\n---"
    return call_ollama(prompt, model=model)


# ---------------------------------------------------------------------------
# Time saved + XP
# ---------------------------------------------------------------------------
_TIME_RATES = {
    "reply": (3, 90),
    "summary": (2, 150),
    "plan": (4, 80),
    "fix": (6, 50),
}


def estimate_minutes_saved(text, task):
    words = len(re.findall(r"\S+", text))
    base, per_minute = _TIME_RATES[task]
    minutes = math.ceil(base + words / per_minute)
    return max(2, min(60, minutes))


# ---------------------------------------------------------------------------
# Frontend Single Page (HTML + CSS + Vanilla JS)
# ---------------------------------------------------------------------------
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sloth Mode — Do Less. Get It Done.</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;750;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<script src="/static/lottie.min.js"></script>
<script src="/static/lottie-player.js"></script>
<style>
  :root {
    --primary: #059669;
    --primary-light: #10b981;
    --primary-dark: #047857;
    --primary-gradient: linear-gradient(135deg, #059669 0%, #10b981 100%);
    --primary-glow: rgba(16, 185, 129, 0.15);
    --accent-subtle: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.02) 100%);
    --bg-base: #f8fafc;
    --bg-surface: #ffffff;
    --bg-subtle: #f1f5f9;
    --border-subtle: #e2e8f0;
    --border-emerald: rgba(16, 185, 129, 0.24);
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --radius: 7px;
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }

  html {
    scroll-behavior: smooth;
  }

  body {
    margin: 0;
    background-color: var(--bg-base);
    background-image: 
      radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.12) 0%, rgba(248, 250, 252, 0) 55%),
      linear-gradient(180deg, #f8fafc 0%, #ffffff 40%, #f1f5f9 100%);
    color: var(--slate-900);
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    line-height: 1.55;
    overflow-x: hidden;
  }

  /* Strict 7px border radius across elements */
  .card, .btn, button, input, textarea, .badge, .status-pill, .pill,
  .xp-track, .xp-fill, .example-btn, .metric-chip, .reward-panel,
  .step-card, .mode-card, .math-banner, .math-stat-box, .privacy-item,
  .brand-icon-wrap, .level-card, .card-accent-bar, .type-badge-result,
  .loading-center-icon, .loading-pulse-ring, .reward-icon-box, .xp-awarded-pill,
  .nav-link-btn, .step-number-tag, .step-icon-wrap, .mode-icon-box, .privacy-icon,
  .copy-action-btn, .card-badge, .hero-badge, .section-badge, .mode-time-chip,
  .math-formula-box, .kbd {
    border-radius: 7px !important;
  }

  button, textarea { font: inherit; }
  button { cursor: pointer; }
  button:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--primary-light);
    outline-offset: 2px;
  }

  /* Enhanced Topbar */
  .topbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 100;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.85);
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03);
  }

  .top-inner {
    max-width: 1200px;
    margin: 0 auto;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
  }

  .brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: inherit;
  }

  .brand-icon-wrap {
    width: 36px;
    height: 36px;
    background: transparent;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
  }

  .brand-icon-wrap svg, .brand-icon-wrap lottie-player {
    width: 36px !important;
    height: 36px !important;
  }

  .brand-info {
    display: flex;
    flex-direction: column;
  }

  .brand-title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--slate-900);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .brand-pill {
    font-size: 9.5px;
    font-weight: 750;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    background: var(--primary-gradient);
    color: #ffffff;
    padding: 2px 7px;
  }

  .brand-sub {
    font-size: 10px;
    font-weight: 650;
    color: var(--slate-500);
    letter-spacing: 0.6px;
    text-transform: uppercase;
  }

  .level-card {
    flex: 1;
    max-width: 360px;
    min-width: 220px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 7px 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
  }

  .level-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11.5px;
    margin-bottom: 5px;
  }

  .level-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .level-badge svg {
    color: var(--primary);
  }

  .level-xp-text {
    font-weight: 650;
    color: var(--primary);
    font-size: 11px;
  }

  .xp-track {
    height: 6px;
    background: #e2e8f0;
    overflow: hidden;
    position: relative;
  }

  .xp-fill {
    height: 100%;
    width: 0%;
    background: var(--primary-gradient);
    transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
  }

  .level-footer {
    display: flex;
    justify-content: space-between;
    font-size: 10.5px;
    color: var(--slate-500);
    margin-top: 4px;
  }

  .metrics-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .metric-chip {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    box-shadow: 0 2px 5px rgba(15, 23, 42, 0.02);
  }

  .metric-chip svg {
    color: var(--primary);
  }

  .metric-chip strong {
    font-size: 14px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .metric-label {
    color: var(--slate-500);
    font-size: 11px;
  }

  .nav-link-btn {
    background: transparent;
    border: 1px solid var(--border-subtle);
    padding: 8px 14px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--slate-700);
    display: flex;
    align-items: center;
    gap: 6px;
    text-decoration: none;
    transition: all 0.2s ease;
  }

  .nav-link-btn:hover {
    background: var(--bg-subtle);
    border-color: var(--primary);
    color: var(--primary);
  }

  /* Hero Section: 100vh Full Immersion */
  .hero-section {
    height: 100vh !important;
    min-height: 100vh !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 72px 24px 30px;
    position: relative;
    max-width: 980px;
    margin: 0 auto;
    box-sizing: border-box;
    text-align: center;
  }

  @keyframes slothFloat {
    0%, 100% {
      transform: translateY(0px);
    }
    50% {
      transform: translateY(-9px);
    }
  }

  .hero-mascot-wrap {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    z-index: 2;
    animation: slothFloat 3s ease-in-out infinite;
    will-change: transform;
  }

  .hero-sloth-lottie {
    width: 175px;
    height: 175px;
    max-width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    filter: drop-shadow(0 14px 36px rgba(16, 185, 129, 0.28));
  }

  .hero-sloth-lottie svg,
  .hero-sloth-lottie lottie-player {
    width: 175px !important;
    height: 175px !important;
    max-width: 100%;
    display: block;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.04));
    border: 1px solid var(--border-emerald);
    color: var(--primary-dark);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
    margin-bottom: 14px;
  }

  h1 {
    font-size: clamp(34px, 5.2vw, 54px);
    font-weight: 800;
    letter-spacing: -1.8px;
    line-height: 1.1;
    margin: 0 0 14px 0;
    text-align: center;
    color: var(--slate-900);
  }

  .gradient-text {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
  }

  .hero-desc {
    font-size: clamp(15px, 1.8vw, 17.5px);
    line-height: 1.65;
    color: var(--slate-600);
    max-width: 650px;
    text-align: center;
    margin: 0 auto 28px auto;
  }

  .hero-cta-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--primary-gradient);
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 750;
    padding: 13px 28px;
    border: none;
    text-decoration: none;
    box-shadow: 0 8px 24px rgba(5, 150, 105, 0.32);
    transition: all 0.2s ease;
  }

  .hero-cta-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(5, 150, 105, 0.42);
    color: #ffffff !important;
  }

  .scroll-down-indicator {
    position: absolute;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    color: var(--slate-400);
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.2s;
  }

  .scroll-down-indicator:hover {
    color: var(--primary);
    transform: translateX(-50%) translateY(3px);
  }

  /* Separate Dedicated Tool Workspace Section — Never Cuts Off */
  .tool-section {
    max-width: 980px;
    margin: 0 auto;
    padding: 60px 20px 80px;
    scroll-margin-top: 70px;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* Clean Status Pill — No muddy yellowish lines */
  .status-pill {
    display: none;
    width: 100%;
    max-width: 860px;
    margin: 0 auto 18px;
    background: rgba(254, 243, 199, 0.55);
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 12px 18px;
    color: #92400e;
    font-size: 13px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.06);
  }

  .status-pill-inner {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .status-pill code {
    background: #ffffff;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 2px 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: #78350f;
    font-weight: 600;
  }

  /* Main Tool Card */
  .tool-card {
    width: 100%;
    max-width: 860px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    box-shadow: 0 16px 40px -10px rgba(15, 23, 42, 0.06), 0 0 0 1px rgba(16, 185, 129, 0.08);
    overflow: hidden;
    position: relative;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    opacity: 1 !important;
    visibility: visible !important;
  }

  .tool-card:focus-within {
    border-color: rgba(16, 185, 129, 0.5);
    box-shadow: 0 20px 45px -10px rgba(16, 185, 129, 0.12), 0 0 0 1px rgba(16, 185, 129, 0.3);
  }

  .card-accent-bar {
    height: 3px;
    width: 100%;
    background: var(--primary-gradient);
  }

  .card-head {
    padding: 16px 22px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }

  .card-title-group {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 15px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .card-title-group svg {
    color: var(--primary);
  }

  .card-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 700;
    color: var(--primary-dark);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.04));
    border: 1px solid var(--border-emerald);
    padding: 4px 10px;
  }

  .card-body {
    padding: 8px 22px 10px;
  }

  textarea {
    width: 100%;
    min-height: 170px;
    height: 175px;
    resize: vertical;
    border: 1px solid var(--border-subtle);
    outline: none;
    background: #fafbfc;
    color: var(--slate-900);
    font-size: 15px;
    line-height: 1.65;
    padding: 14px 16px;
    transition: background 0.2s, border-color 0.2s;
  }

  textarea:focus {
    background: #ffffff;
    border-color: rgba(16, 185, 129, 0.4);
  }

  textarea::placeholder {
    color: #94a3b8;
  }

  .meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    font-size: 11.5px;
    color: var(--slate-500);
    padding: 8px 2px 0;
  }

  .kbd {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    padding: 1px 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--slate-700);
  }

  .actions-bar {
    padding: 14px 22px;
    background: #f8fafc;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }

  .actions-hint {
    font-size: 12.5px;
    color: var(--slate-500);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .actions-hint svg {
    color: var(--primary);
  }

  .primary-btn {
    background: var(--primary-gradient);
    color: #ffffff;
    border: none;
    padding: 12px 22px;
    font-size: 14px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
  }

  .primary-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
    filter: brightness(1.05);
  }

  .primary-btn:active:not(:disabled) {
    transform: translateY(0);
  }

  .primary-btn:disabled {
    opacity: 0.7;
    cursor: wait;
    transform: none;
  }

  /* Quick Examples Bar */
  .examples-wrap {
    width: 100%;
    max-width: 860px;
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 12.5px;
    color: var(--slate-500);
  }

  .examples-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .example-btn {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--slate-700);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
    opacity: 1 !important;
    visibility: visible !important;
  }

  .example-btn svg {
    color: var(--primary);
  }

  .example-btn:hover {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.02));
    border-color: var(--primary);
    color: var(--primary-dark);
    transform: translateY(-1px);
  }

  /* Animated AI Loading State */
  .loading-panel {
    display: none;
    width: 100%;
    max-width: 860px;
    margin-top: 20px;
    background: #ffffff;
    border: 1px solid var(--border-emerald);
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.08);
    position: relative;
    overflow: hidden;
  }

  .loading-radar {
    width: 52px;
    height: 52px;
    margin: 0 auto 14px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .loading-pulse-ring {
    position: absolute;
    inset: 0;
    border: 2px solid var(--primary-light);
    animation: radarPulse 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    opacity: 0;
  }

  .loading-pulse-ring:nth-child(2) {
    animation-delay: 0.6s;
  }

  .loading-pulse-ring:nth-child(3) {
    animation-delay: 1.2s;
  }

  @keyframes radarPulse {
    0% { transform: scale(0.6); opacity: 0.8; }
    100% { transform: scale(1.6); opacity: 0; }
  }

  .loading-center-icon {
    width: 38px;
    height: 38px;
    background: var(--primary-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    z-index: 2;
  }

  .loading-msg {
    font-size: 15px;
    font-weight: 750;
    color: var(--slate-900);
    margin-bottom: 4px;
  }

  .loading-subtext {
    font-size: 12px;
    color: var(--slate-500);
  }

  /* Error Box */
  .error-box {
    display: none;
    width: 100%;
    max-width: 860px;
    margin-top: 18px;
    background: rgba(254, 242, 242, 0.85);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #991b1b;
    padding: 14px 18px;
    font-size: 13.5px;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.05);
  }

  .error-box-inner {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  /* Result Card */
  .result-card {
    display: none;
    width: 100%;
    max-width: 860px;
    margin-top: 22px;
    background: #ffffff;
    border: 1px solid var(--border-emerald);
    box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.12);
    overflow: hidden;
  }

  .result-header {
    padding: 15px 22px;
    background: #f8fafc;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .result-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .result-title-group svg {
    color: var(--primary);
  }

  .type-badge-result {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.05));
    border: 1px solid var(--border-emerald);
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
    color: var(--primary-dark);
  }

  .copy-action-btn {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 7px 14px;
    font-size: 12.5px;
    font-weight: 650;
    color: var(--slate-700);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
  }

  .copy-action-btn:hover {
    background: var(--bg-subtle);
    border-color: var(--primary);
    color: var(--primary);
  }

  .result-output-content {
    padding: 22px;
    font-size: 14.5px;
    line-height: 1.7;
    color: var(--slate-800);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 520px;
    overflow-y: auto;
    background: #ffffff;
  }

  .reward-panel {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.02) 100%);
    border-top: 1px solid var(--border-emerald);
    padding: 16px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }

  .reward-stats {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .reward-icon-box {
    width: 40px;
    height: 40px;
    background: #ffffff;
    border: 1px solid var(--border-emerald);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
  }

  .saved-time-title {
    font-size: 19px;
    font-weight: 800;
    color: var(--slate-900);
    letter-spacing: -0.5px;
  }

  .saved-time-desc {
    font-size: 11.5px;
    color: var(--slate-500);
  }

  .xp-awarded-pill {
    background: var(--primary-gradient);
    color: #ffffff;
    padding: 8px 16px;
    font-size: 13.5px;
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
  }

  /* Scroll Down Hint */
  .scroll-down-btn {
    margin-top: 22px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--slate-500);
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border: 1px solid transparent;
    transition: all 0.2s;
  }

  .scroll-down-btn:hover {
    color: var(--primary);
    border-color: var(--border-emerald);
    background: #ffffff;
  }

  /* Details Section: "How it works and how it saves time" */
  .details-section {
    max-width: 1140px;
    margin: 0 auto;
    padding: 70px 24px 80px;
    scroll-margin-top: 75px;
  }

  .section-header {
    text-align: center;
    margin-bottom: 46px;
  }

  .section-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--primary);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.04));
    border: 1px solid var(--border-emerald);
    padding: 4px 12px;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: clamp(28px, 4vw, 38px);
    font-weight: 800;
    letter-spacing: -1.2px;
    color: var(--slate-900);
    margin: 0 0 12px;
  }

  .section-subtitle {
    font-size: 15.5px;
    color: var(--slate-600);
    max-width: 620px;
    margin: 0 auto;
  }

  .steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 20px;
    margin-bottom: 50px;
  }

  .step-card {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 24px;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03);
    position: relative;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  }

  .step-card:hover {
    transform: translateY(-3px);
    border-color: var(--primary);
    box-shadow: 0 12px 30px -4px rgba(16, 185, 129, 0.12);
  }

  .step-number-tag {
    position: absolute;
    top: 20px;
    right: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: var(--primary);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.03));
    border: 1px solid var(--border-emerald);
    padding: 2px 7px;
  }

  .step-icon-wrap {
    width: 44px;
    height: 44px;
    background: var(--primary-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    margin-bottom: 16px;
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25);
  }

  .step-title {
    font-size: 17px;
    font-weight: 750;
    color: var(--slate-900);
    margin: 0 0 8px;
  }

  .step-desc {
    font-size: 13.5px;
    color: var(--slate-600);
    line-height: 1.6;
    margin: 0;
  }

  .modes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 18px;
    margin-bottom: 50px;
  }

  .mode-card {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 22px;
    transition: all 0.3s ease;
  }

  .mode-card:hover {
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 8px 24px -2px rgba(16, 185, 129, 0.08);
    transform: translateY(-2px);
  }

  .mode-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }

  .mode-icon-box {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.03));
    border: 1px solid var(--border-emerald);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
  }

  .mode-name {
    font-size: 15px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .mode-desc {
    font-size: 13px;
    color: var(--slate-600);
    line-height: 1.55;
    margin: 0 0 14px;
  }

  .mode-time-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    font-weight: 700;
    color: var(--primary-dark);
    background: #ecfdf5;
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 4px 9px;
  }

  /* Math Banner: How it saves time */
  .math-banner {
    background: #ffffff;
    border: 1px solid var(--border-emerald);
    padding: 30px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.06);
    display: flex;
    gap: 30px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 50px;
  }

  .math-banner-content {
    flex: 1;
    min-width: 280px;
  }

  .math-banner h3 {
    font-size: 21px;
    font-weight: 800;
    margin: 0 0 10px;
    color: var(--slate-900);
  }

  .math-banner p {
    font-size: 14px;
    line-height: 1.6;
    color: var(--slate-600);
    margin: 0 0 14px;
  }

  .math-formula-box {
    background: #f8fafc;
    border: 1px solid var(--border-subtle);
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: var(--slate-800);
  }

  .math-stats-cards {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }

  .math-stat-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.02));
    border: 1px solid var(--border-emerald);
    padding: 14px 18px;
    min-width: 130px;
    text-align: center;
  }

  .math-stat-val {
    font-size: 24px;
    font-weight: 800;
    color: var(--primary);
  }

  .math-stat-lbl {
    font-size: 11px;
    color: var(--slate-500);
    margin-top: 3px;
  }

  /* Privacy & Architecture Grid */
  .privacy-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 18px;
    margin-bottom: 40px;
  }

  .privacy-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 18px;
  }

  .privacy-icon {
    width: 38px;
    height: 38px;
    background: var(--primary-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    flex-shrink: 0;
  }

  .privacy-item h4 {
    margin: 0 0 4px;
    font-size: 14.5px;
    font-weight: 750;
    color: var(--slate-900);
  }

  .privacy-item p {
    margin: 0;
    font-size: 12.5px;
    color: var(--slate-600);
    line-height: 1.5;
  }

  /* Comprehensive Feature Documentation Grid */
  .features-docs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 20px;
    margin-bottom: 55px;
  }

  .feature-doc-card {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 22px;
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .feature-doc-card:hover {
    border-color: rgba(16, 185, 129, 0.45);
    box-shadow: 0 10px 28px -4px rgba(16, 185, 129, 0.1);
    transform: translateY(-2px);
  }

  .feature-doc-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .feature-doc-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(5, 150, 105, 0.04));
    border: 1px solid var(--border-emerald);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    flex-shrink: 0;
  }

  .feature-doc-title {
    margin: 0;
    font-size: 15.5px;
    font-weight: 800;
    color: var(--slate-900);
  }

  .feature-doc-badge {
    font-size: 10px;
    font-weight: 700;
    color: var(--primary-dark);
    background: rgba(16, 185, 129, 0.1);
    padding: 1px 6px;
    border: 1px solid var(--border-emerald);
    display: inline-block;
    margin-top: 2px;
  }

  .feature-doc-section {
    margin-top: 10px;
  }

  .feature-doc-section strong {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--primary-dark);
    margin-bottom: 4px;
  }

  .feature-doc-section p {
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
    color: var(--slate-600);
  }

  .feature-doc-why {
    background: #f8fafc;
    border-left: 3px solid var(--primary);
    padding: 10px 12px;
    margin-top: 12px;
  }

  .feature-doc-why strong {
    color: var(--slate-800);
  }

  .feature-doc-why p {
    color: var(--slate-700);
    font-weight: 500;
  }

  /* FAQ Section Styling */
  .faq-container {
    max-width: 900px;
    margin: 0 auto 55px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .faq-card {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .faq-card:hover {
    border-color: rgba(16, 185, 129, 0.35);
  }

  .faq-card.active {
    border-color: var(--primary);
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);
  }

  .faq-question {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: 18px 20px;
    font-size: 14.5px;
    font-weight: 750;
    color: var(--slate-800);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    cursor: pointer;
    transition: color 0.2s;
  }

  .faq-card.active .faq-question {
    color: var(--primary-dark);
    background: #fafbfc;
  }

  .faq-icon-caret {
    color: var(--slate-400);
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), color 0.2s;
    flex-shrink: 0;
  }

  .faq-card.active .faq-icon-caret {
    transform: rotate(180deg);
    color: var(--primary);
  }

  .faq-answer {
    display: none;
    padding: 0 20px 18px;
    font-size: 13.5px;
    line-height: 1.65;
    color: var(--slate-600);
    background: #fafbfc;
    border-top: 1px solid rgba(226, 232, 240, 0.6);
  }

  .faq-card.active .faq-answer {
    display: block;
    padding-top: 14px;
  }

  .faq-answer code {
    background: #f1f5f9;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--primary-dark);
    border: 1px solid var(--border-subtle);
  }

  footer {
    border-top: 1px solid var(--border-subtle);
    padding: 24px 20px;
    text-align: center;
    color: var(--slate-500);
    font-size: 12.5px;
    background: #ffffff;
  }

  /* Model Selector */
  .card-head-tools {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .model-picker-wrap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 3px 8px;
    font-size: 11.5px;
    font-weight: 650;
    color: var(--slate-700);
    transition: all 0.2s ease;
  }

  .model-picker-wrap:hover {
    border-color: var(--primary);
  }

  .model-picker-wrap svg {
    color: var(--primary);
  }

  .model-select {
    border: none;
    outline: none;
    background: transparent;
    font-family: inherit;
    font-size: 11.5px;
    font-weight: 650;
    color: var(--slate-800);
    cursor: pointer;
  }

  /* Voice Dictation Button */
  .voice-dictate-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    font-size: 11.5px;
    font-weight: 700;
    color: var(--slate-700);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .voice-dictate-btn svg {
    color: var(--primary);
  }

  .voice-dictate-btn:hover {
    border-color: var(--primary);
    color: var(--primary-dark);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.02));
  }

  .voice-dictate-btn.listening {
    background: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #dc2626 !important;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2) !important;
    animation: micPulse 1.2s infinite ease-in-out;
  }

  .voice-dictate-btn.listening svg {
    color: #dc2626 !important;
  }

  @keyframes micPulse {
    0%, 100% { box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2); }
    50% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0.35); }
  }

  /* Sloth Tone Persona Bar */
  .tone-bar-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 22px;
    background: #f8fafc;
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
  }

  .tone-bar-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    color: var(--slate-600);
  }

  .tone-bar-label svg {
    color: var(--primary);
  }

  .tone-chips {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .tone-chip {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    font-size: 11.5px;
    font-weight: 650;
    color: var(--slate-600);
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .tone-chip:hover {
    border-color: rgba(16, 185, 129, 0.4);
    color: var(--slate-900);
    background: #f8fafc;
  }

  .tone-chip.active {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.04));
    border-color: var(--primary);
    color: var(--primary-dark);
    font-weight: 750;
    box-shadow: 0 1px 4px rgba(16, 185, 129, 0.15);
  }

  /* Action Buttons in Result Header */
  .result-actions-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .action-pill-btn {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 7px 11px;
    font-size: 12px;
    font-weight: 650;
    color: var(--slate-700);
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
  }

  .action-pill-btn:hover {
    background: var(--bg-subtle);
    border-color: var(--primary);
    color: var(--primary);
    transform: translateY(-1px);
  }

  .action-pill-btn:active {
    transform: translateY(0);
  }

  .action-pill-btn svg {
    color: var(--primary);
  }

  /* Lazy Vault Drawer & Backdrop */
  .vault-toggle-btn {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
    color: var(--slate-700);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .vault-toggle-btn:hover {
    background: var(--bg-subtle);
    border-color: var(--primary);
    color: var(--primary);
  }

  .vault-toggle-btn svg {
    color: var(--primary);
  }

  .vault-count-badge {
    background: var(--primary-gradient);
    color: #ffffff;
    font-size: 10px;
    font-weight: 800;
    padding: 1px 6px;
    min-width: 14px;
    text-align: center;
  }

  .vault-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
    z-index: 999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.24s ease;
  }

  .vault-backdrop.open {
    opacity: 1;
    pointer-events: auto;
  }

  .vault-drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: 440px;
    max-width: 92vw;
    background: #ffffff;
    border-left: 1px solid var(--border-subtle);
    box-shadow: -8px 0 35px rgba(15, 23, 42, 0.12);
    z-index: 1000;
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .vault-drawer.open {
    transform: translateX(0);
  }

  .vault-drawer-head {
    padding: 16px 20px;
    background: #f8fafc;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .vault-head-title {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .vault-head-title h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 800;
    color: var(--slate-900);
    letter-spacing: -0.3px;
  }

  .vault-head-title svg {
    color: var(--primary);
  }

  .vault-offline-badge {
    font-size: 9.5px;
    font-weight: 700;
    color: var(--primary-dark);
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid var(--border-emerald);
    padding: 2px 6px;
  }

  .vault-close-btn {
    background: transparent;
    border: 1px solid var(--border-subtle);
    padding: 5px;
    cursor: pointer;
    color: var(--slate-500);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }

  .vault-close-btn:hover {
    background: #f1f5f9;
    color: var(--slate-800);
    border-color: var(--slate-400);
  }

  .vault-drawer-body {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .vault-card {
    background: #fafbfc;
    border: 1px solid var(--border-subtle);
    padding: 12px 14px;
    transition: all 0.2s ease;
  }

  .vault-card:hover {
    background: #ffffff;
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
  }

  .vault-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
  }

  .vault-card-tags {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }

  .vault-tag {
    font-size: 10px;
    font-weight: 750;
    padding: 2px 6px;
    border: 1px solid var(--border-emerald);
    background: rgba(16, 185, 129, 0.08);
    color: var(--primary-dark);
  }

  .vault-tag.tone-tag {
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: var(--slate-600);
  }

  .vault-time {
    font-size: 11px;
    color: var(--slate-400);
  }

  .vault-card-preview {
    font-size: 12px;
    line-height: 1.5;
    color: var(--slate-700);
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 8px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .vault-card-actions {
    display: flex;
    align-items: center;
    gap: 6px;
    border-top: 1px solid var(--border-subtle);
    padding-top: 7px;
  }

  .vault-mini-btn {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 3px 8px;
    font-size: 10.5px;
    font-weight: 650;
    color: var(--slate-600);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.15s ease;
  }

  .vault-mini-btn:hover {
    background: var(--bg-subtle);
    border-color: var(--primary);
    color: var(--primary);
  }

  .vault-mini-btn.danger:hover {
    border-color: #ef4444;
    color: #ef4444;
    background: #fef2f2;
  }

  .vault-empty-state {
    text-align: center;
    padding: 40px 16px;
    color: var(--slate-400);
  }

  .vault-empty-state svg {
    margin: 0 auto 10px;
    color: var(--slate-300);
  }

  .vault-empty-state h4 {
    margin: 0 0 4px;
    font-size: 14.5px;
    color: var(--slate-700);
    font-weight: 750;
  }

  .vault-empty-state p {
    margin: 0;
    font-size: 12px;
    line-height: 1.5;
  }

  .vault-drawer-foot {
    padding: 12px 18px;
    background: #f8fafc;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: flex-end;
  }

  .vault-clear-btn {
    background: transparent;
    border: 1px solid var(--border-subtle);
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 650;
    color: var(--slate-500);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .vault-clear-btn:hover {
    border-color: #ef4444;
    color: #ef4444;
    background: #fef2f2;
  }

  @media (max-width: 768px) {
    .top-inner { padding: 10px 16px; gap: 12px; }
    .level-card { order: 3; max-width: 100%; width: 100%; }
    .metrics-group { order: 2; width: 100%; justify-content: space-between; flex-wrap: wrap; }
    .hero-section { height: 100vh !important; min-height: 100vh !important; padding: 68px 16px 20px; }
    .hero-sloth-lottie { width: 110px; height: 110px; }
    .hero-sloth-lottie svg, .hero-sloth-lottie lottie-player { width: 110px !important; height: 110px !important; }
    .card-head, .card-body, .actions-bar, .result-header, .result-output-content, .reward-panel {
      padding-left: 16px; padding-right: 16px;
    }
    .card-head-tools { width: 100%; justify-content: flex-start; }
    .vault-drawer { width: 100vw; max-width: 100vw; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }
</style>
</head>
<body>

<!-- Enhanced Top Header Bar -->
<header class="topbar">
  <div class="top-inner">
    <a href="#" class="brand-group">
      <div class="brand-icon-wrap" id="brandLottieWrap">
        <lottie-player
          id="brandLottiePlayer"
          src="/static/sloth-meditation.json"
          background="transparent"
          speed="1"
          style="width: 36px; height: 36px;"
          loop
          autoplay>
        </lottie-player>
      </div>
      <div class="brand-info">
        <span class="brand-title">Sloth Mode <span class="brand-pill">Local AI</span></span>
        <span class="brand-sub">Slow is the new fast</span>
      </div>
    </a>

    <!-- Level Progression Card -->
    <div class="level-card">
      <div class="level-header">
        <div class="level-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>
            <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>
            <path d="M4 22h16"/>
            <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>
            <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>
            <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>
          </svg>
          <span id="levelName">Baby Sloth</span>
        </div>
        <span class="level-xp-text" id="xpLabel">0 / 200 XP</span>
      </div>
      <div class="xp-track">
        <div class="xp-fill" id="xpFill"></div>
      </div>
      <div class="level-footer">
        <span id="nextHint">200 XP to Couch Sloth</span>
        <span>Local Level</span>
      </div>
    </div>

    <!-- Live Metrics Counter -->
    <div class="metrics-group">
      <div class="metric-chip">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
        <div>
          <strong id="totalMinutes">0 min</strong>
          <span class="metric-label">Saved</span>
        </div>
      </div>
      <div class="metric-chip">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <div>
          <strong id="totalTasks">0</strong>
          <span class="metric-label">Tasks</span>
        </div>
      </div>
      <button type="button" class="vault-toggle-btn" id="vaultToggleBtn" aria-label="Open Lazy Vault history drawer" title="Lazy Vault (Offline History)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect width="20" height="14" x="2" y="5" rx="2"/>
          <line x1="2" y1="10" x2="22" y2="10"/>
        </svg>
        <span>Vault</span>
        <span class="vault-count-badge" id="vaultCountBadge">0</span>
      </button>
      <a href="#workspace" class="nav-link-btn">
        <span>Workspace</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
      </a>
      <a href="#how-it-works" class="nav-link-btn">
        <span>How it works</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
      </a>
    </div>
  </div>
</header>

<!-- Hero Section: 100vh Full Screen Immersion -->
<section class="hero-section" id="heroSection">
  <div class="hero-mascot-wrap">
    <div id="slothLottieContainer" class="hero-sloth-lottie" aria-label="Sloth doing meditation animation">
      <lottie-player
        id="slothLottiePlayer"
        src="/static/sloth-meditation.json"
        background="transparent"
        speed="1"
        style="width: 175px; height: 175px;"
        loop
        autoplay>
      </lottie-player>
    </div>
  </div>

  <div class="hero-badge">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
    <span>Autonomous Local AI Engine</span>
  </div>

  <h1>Do less. <span class="gradient-text">Get it done.</span></h1>
  <p class="hero-desc">
    Paste anything messy — an email, a giant chat thread, a brain-dump to-do list, or buggy code.
    Sloth Mode automatically identifies the intent and performs the heavy lifting on your machine.
  </p>

  <a href="#workspace" class="hero-cta-btn">
    <span>Open Sloth Workspace</span>
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m6 9 6 6 6-6"/></svg>
  </a>

  <a href="#workspace" class="scroll-down-indicator" aria-label="Scroll to workspace">
    <span>Scroll to tool</span>
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
  </a>
</section>

<!-- Dedicated Tool Section: Zero Cutoffs -->
<section class="tool-section" id="workspace">
  <!-- Clean Alert Status Pill -->
  <div class="status-pill" id="ollamaPill">
    <div class="status-pill-inner">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <div>
        <strong>Ollama is offline:</strong> Start Ollama first with <code>ollama serve</code> and verify <code>ollama pull llama3.2</code> is completed.
      </div>
    </div>
  </div>

  <!-- Main Tool Card with 7px radius & emerald accent -->
  <form class="tool-card" id="taskForm">
    <div class="card-accent-bar"></div>
    <div class="card-head">
      <div class="card-title-group">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
        <label for="messyInput">What is the messy thing?</label>
      </div>
      <div class="card-head-tools">
        <!-- Model Selector Dropdown -->
        <div class="model-picker-wrap" title="Local Ollama Model Selector">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
          </svg>
          <select id="modelSelect" class="model-select" aria-label="Select Local Model">
            <option value="llama3.2:1b" selected>llama3.2:1b</option>
          </select>
        </div>

        <!-- Voice Dictation ("Too Lazy to Type") -->
        <button type="button" class="voice-dictate-btn" id="voiceDictateBtn" title="Too lazy to type? Dictate your messy thoughts">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="22"/>
          </svg>
          <span id="voiceDictateText">Too Lazy to Type</span>
        </button>

        <span class="card-badge">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>
          </svg>
          <span>Zero Prompts</span>
        </span>
      </div>
    </div>

    <!-- Sloth Tone / Persona Selector Bar -->
    <div class="tone-bar-wrap">
      <div class="tone-bar-label">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
          <line x1="9" y1="9" x2="9.01" y2="9"/>
          <line x1="15" y1="9" x2="15.01" y2="9"/>
        </svg>
        <span>Sloth Persona:</span>
      </div>
      <div class="tone-chips" role="radiogroup" aria-label="Sloth Tone Modifiers">
        <button type="button" class="tone-chip active" data-tone="ultra-lazy" title="Minimum words, blunt, zero fluff">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
          </svg>
          <span>Ultra Lazy</span>
        </button>
        <button type="button" class="tone-chip" data-tone="corporate" title="Polished, professional executive etiquette">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="20" height="14" x="2" y="7" rx="2" ry="2"/>
            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
          </svg>
          <span>Corporate</span>
        </button>
        <button type="button" class="tone-chip" data-tone="detailed" title="Step-by-step comprehensive breakdown">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <line x1="10" y1="9" x2="8" y2="9"/>
          </svg>
          <span>Detailed</span>
        </button>
        <button type="button" class="tone-chip" data-tone="chill" title="Friendly, relaxed, conversational vibe">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17 8h1a4 4 0 1 1 0 8h-1"/>
            <path d="M3 8h14v9a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4Z"/>
            <line x1="6" y1="2" x2="6" y2="4"/>
            <line x1="10" y1="2" x2="10" y2="4"/>
            <line x1="14" y1="2" x2="14" y2="4"/>
          </svg>
          <span>Chill</span>
        </button>
      </div>
    </div>

    <div class="card-body">
      <textarea id="messyInput" maxlength="30000" placeholder="Paste it here…

An email you don't want to reply to…
A WhatsApp thread with 200 messages…
Everything in your head, in no order at all…
Or that code that keeps throwing an error…"></textarea>
      <div class="meta-row">
        <span>100% on your laptop · Press <span class="kbd">Ctrl</span> + <span class="kbd">Enter</span> to run</span>
        <span id="charCount">0 / 30,000</span>
      </div>
    </div>

    <div class="actions-bar">
      <p class="actions-hint">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>
        </svg>
        <span>You paste. The sloth executes. Zero data leaves your machine.</span>
      </p>
      <button class="primary-btn" id="submitBtn" type="submit">
        <span>Let the sloth handle it</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M5 12h14"/>
          <path d="m12 5 7 7-7 7"/>
        </svg>
      </button>
    </div>
  </form>

  <!-- Quick Examples Bar with SVG Icons -->
  <div class="examples-wrap">
    <span>Need an example to test?</span>
    <div class="examples-buttons">
      <button type="button" class="example-btn" data-example="0">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect width="20" height="16" x="2" y="4" rx="2"/>
          <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
        </svg>
        <span>Inbound Email</span>
      </button>
      <button type="button" class="example-btn" data-example="1">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <span>Group Chat Thread</span>
      </button>
      <button type="button" class="example-btn" data-example="2">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m3 16 2 2 4-4"/>
          <path d="m3 7 2 2 4-4"/>
          <path d="M13 6h8"/>
          <path d="M13 12h8"/>
          <path d="M13 18h8"/>
        </svg>
        <span>Chaotic Brain-Dump</span>
      </button>
      <button type="button" class="example-btn" data-example="3">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m8 2 1.88 1.88"/>
          <path d="M14.12 3.88 16 2"/>
          <rect width="8" height="14" x="8" y="6" rx="4"/>
          <path d="M4 13h4"/>
          <path d="M16 13h4"/>
          <path d="m5 19 3-2"/>
          <path d="m19 19-3-2"/>
          <path d="m5 7 3 2"/>
          <path d="m19 7-3 2"/>
        </svg>
        <span>Buggy Python Code</span>
      </button>
    </div>
  </div>

  <!-- Animated AI Loading Box -->
  <div class="loading-panel" id="loadingPanel">
    <div class="loading-radar">
      <div class="loading-pulse-ring"></div>
      <div class="loading-pulse-ring"></div>
      <div class="loading-pulse-ring"></div>
      <div class="loading-center-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 6v6l4 2"/>
        </svg>
      </div>
    </div>
    <div class="loading-msg" id="loadingText">Waking the sloth up…</div>
    <div class="loading-subtext">Executing local inference via Ollama + Llama 3.2</div>
  </div>

  <!-- Error Card -->
  <div class="error-box" id="errorBox" role="alert">
    <div class="error-box-inner">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <span id="errorMsgText">Something went wrong.</span>
    </div>
  </div>

  <!-- Result Output Card -->
  <section class="result-card" id="resultCard" aria-live="polite">
    <div class="result-header">
      <div class="result-title-group">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>Consider it handled</span>
        <span class="type-badge-result" id="typeBadge"></span>
      </div>
      <div class="result-actions-group">
        <button class="action-pill-btn" id="emailBtn" type="button" title="Open draft directly in your email client">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect width="20" height="16" x="2" y="4" rx="2"/>
            <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
          </svg>
          <span>Open Email</span>
        </button>
        <button class="action-pill-btn" id="checklistBtn" type="button" title="Convert to markdown checklist & copy">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect width="6" height="6" x="3" y="5" rx="1"/>
            <path d="m3 17 2 2 4-4"/>
            <path d="M13 6h8"/>
            <path d="M13 12h8"/>
            <path d="M13 18h8"/>
          </svg>
          <span id="checklistBtnText">Checklist</span>
        </button>
        <button class="action-pill-btn" id="downloadMdBtn" type="button" title="Download output as markdown .md file">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          <span>Save .md</span>
        </button>
        <button class="copy-action-btn" id="copyBtn" type="button">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
          </svg>
          <span id="copyBtnText">Copy output</span>
        </button>
      </div>
    </div>
    <div class="result-output-content" id="outputText"></div>
    <div class="reward-panel">
      <div class="reward-stats">
        <div class="reward-icon-box">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
          </svg>
        </div>
        <div>
          <div class="saved-time-title" id="savedText">You saved ~0 minutes!</div>
          <div class="saved-time-desc">Calculated manual reading and composing effort saved.</div>
        </div>
      </div>
      <div class="xp-awarded-pill">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
        </svg>
        <span id="xpAward">+0 XP</span>
      </div>
    </div>
  </section>

  <!-- Scroll Down Trigger -->
  <a href="#how-it-works" class="scroll-down-btn">
    <span>How Sloth Mode works & time savings explained</span>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
  </a>
</section>

<!-- Bottom Section: How It Works & How It Saves Time -->
<section class="details-section" id="how-it-works">
  <div class="section-header">
    <div class="section-badge">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="16" x2="12" y2="12"/>
        <line x1="12" y1="8" x2="12.01" y2="8"/>
      </svg>
      <span>Under The Hood</span>
    </div>
    <h2 class="section-title">How Sloth Mode Works</h2>
    <p class="section-subtitle">
      Zero mode selectors. Zero complex prompt engineering. The local neural model evaluates what you give it and runs the right task immediately.
    </p>
  </div>

  <!-- 3 Steps Grid -->
  <div class="steps-grid">
    <div class="step-card">
      <span class="step-number-tag">STEP 01</span>
      <div class="step-icon-wrap">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="18" x2="12" y2="12"/>
          <line x1="9" y1="15" x2="15" y2="15"/>
        </svg>
      </div>
      <h3 class="step-title">1. Raw Messy Paste</h3>
      <p class="step-desc">
        Drop any unorganized text into the box. An angry client email, a 50-message debate, an urgent bug trace, or scattered to-do notes. No need to clean it up.
      </p>
    </div>

    <div class="step-card">
      <span class="step-number-tag">STEP 02</span>
      <div class="step-icon-wrap">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect width="16" height="16" x="4" y="4" rx="2"/>
          <rect width="6" height="6" x="9" y="9" rx="1"/>
          <path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>
          <path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>
        </svg>
      </div>
      <h3 class="step-title">2. Zero-Shot Intent Classifier</h3>
      <p class="step-desc">
        The local Llama 3.2 model scans the semantics in milliseconds. It detects whether the text is a message requiring a reply, a conversation, a chaotic list, or buggy code.
      </p>
    </div>

    <div class="step-card">
      <span class="step-number-tag">STEP 03</span>
      <div class="step-icon-wrap">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
      </div>
      <h3 class="step-title">3. Instant Work & XP Reward</h3>
      <p class="step-desc">
        The matching execution pipeline generates the ready-to-copy result. You receive a calculated manual time-savings metric and level up your Sloth tier.
      </p>
    </div>
  </div>

  <!-- The 4 Core Capabilities Grid -->
  <div class="section-header" style="margin-top: 60px;">
    <div class="section-badge">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
      </svg>
      <span>Automatic Modes</span>
    </div>
    <h2 class="section-title">The Four Autonomous Workflows</h2>
  </div>

  <div class="modes-grid">
    <div class="mode-card">
      <div class="mode-head">
        <div class="mode-icon-box">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="20" height="16" x="2" y="4" rx="2"/>
            <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
          </svg>
        </div>
        <span class="mode-name">Reply Draft Engine</span>
      </div>
      <p class="mode-desc">
        Matches tone of the original message. Does not fabricate dates or false commitments; leaves obvious bracketed placeholders for variables you must verify.
      </p>
      <span class="mode-time-chip">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>Saves 3 to 15 min</span>
      </span>
    </div>

    <div class="mode-card">
      <div class="mode-head">
        <div class="mode-icon-box">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <span class="mode-name">Thread & Chat Distiller</span>
      </div>
      <p class="mode-desc">
        Parses giant group messages, Slack threads, or email chains. Produces an executive 3-line overview followed by bulleted action deliverables.
      </p>
      <span class="mode-time-chip">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>Saves 5 to 30 min</span>
      </span>
    </div>

    <div class="mode-card">
      <div class="mode-head">
        <div class="mode-icon-box">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m3 16 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>
          </svg>
        </div>
        <span class="mode-name">Brain-Dump Prioritizer</span>
      </div>
      <p class="mode-desc">
        Takes disorganized tasks, fragmented thoughts, and messy notes. Automatically sorts them by urgency and impact, starting each line with an imperative verb.
      </p>
      <span class="mode-time-chip">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>Saves 4 to 20 min</span>
      </span>
    </div>

    <div class="mode-card">
      <div class="mode-head">
        <div class="mode-icon-box">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="m8 2 1.88 1.88"/><path d="M14.12 3.88 16 2"/><rect width="8" height="14" x="8" y="6" rx="4"/>
            <path d="M4 13h4"/><path d="M16 13h4"/><path d="m5 19 3-2"/><path d="m19 19-3-2"/><path d="m5 7 3 2"/><path d="m19 7-3 2"/>
          </svg>
        </div>
        <span class="mode-name">Code Bug & Syntax Fixer</span>
      </div>
      <p class="mode-desc">
        Identifies exception causes in Python, JavaScript, TypeScript, or Go. Delivers the patched snippet first, followed by a one-line plain English "Why" explanation.
      </p>
      <span class="mode-time-chip">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>Saves 6 to 45 min</span>
      </span>
    </div>
  </div>

  <!-- Time Saved Math Banner -->
  <div class="math-banner">
    <div class="math-banner-content">
      <h3>How Time Saved Is Calculated</h3>
      <p>
        Rather than guessing randomly, Sloth Mode calculates saved time using a calibrated manual effort model combining base task complexity with standard reading and composing speed.
      </p>
      <div class="math-formula-box">
        Time Saved = Base Minutes + (Word Count ÷ Manual Effort Rate)
      </div>
    </div>
    <div class="math-stats-cards">
      <div class="math-stat-box">
        <div class="math-stat-val">3-6m</div>
        <div class="math-stat-lbl">Base Overhead</div>
      </div>
      <div class="math-stat-box">
        <div class="math-stat-val">10 XP</div>
        <div class="math-stat-lbl">Per Min Saved</div>
      </div>
      <div class="math-stat-box">
        <div class="math-stat-val">60m</div>
        <div class="math-stat-lbl">Max Cap / Task</div>
      </div>
    </div>
  </div>

  <!-- Privacy & Local Execution Pillars -->
  <div class="privacy-grid">
    <div class="privacy-item">
      <div class="privacy-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      </div>
      <div>
        <h4>100% On-Device Privacy</h4>
        <p>Your passwords, confidential code, and personal emails never leave memory. No telemetry, no external logging.</p>
      </div>
    </div>

    <div class="privacy-item">
      <div class="privacy-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      </div>
      <div>
        <h4>Zero API Keys or Fees</h4>
        <p>Runs on open weights via Ollama on localhost:11434. No monthly OpenAI or Anthropic credit billing.</p>
      </div>
    </div>

    <div class="privacy-item">
      <div class="privacy-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>
        </svg>
      </div>
      <div>
        <h4>Persistent Browser Storage</h4>
        <p>XP and lifetime hours saved stay safely in your browser's localStorage. Zero database setup needed.</p>
      </div>
    </div>
  </div>

  <!-- Deep Feature Catalog & Value Proposition -->
  <div class="section-header" style="margin-top: 70px;">
    <div class="section-badge">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
        <line x1="9" y1="3" x2="9" y2="21"/>
      </svg>
      <span>Feature Deep Dive</span>
    </div>
    <h2 class="section-title">Complete Feature Catalog & Value Proposition</h2>
    <p class="section-subtitle">
      Built from the ground up for developers, executives, and privacy enthusiasts who want to get their time back without writing prompts or sacrificing data privacy.
    </p>
  </div>

  <div class="features-docs-grid">
    <!-- Feature 1: Voice Dictation -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="22"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">Voice Dictation</h3>
            <span class="feature-doc-badge">Too Lazy to Type</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>Taps directly into native Web Speech APIs with a dual-buffer real-time audio pipeline. It captures your stream of consciousness, continuously replaces interim hypotheses, and commits clean spoken text without word stuttering or repeating phrases.</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>Zero typing fatigue. When you're completely burnt out or pacing around your desk, speak naturally. Sloth Mode absorbs your messy thoughts and cleans them up.</p>
      </div>
    </div>

    <!-- Feature 2: Local Model Selector -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
              <line x1="3" y1="9" x2="21" y2="9"/>
              <line x1="9" y1="21" x2="9" y2="9"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">Local Model Selector</h3>
            <span class="feature-doc-badge">Dynamic Hardware Scaling</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>Autonomously queries your machine's Ollama model catalog at <code>localhost:11434/api/tags</code>. Easily switch between ultra-fast 1B models (Llama 3.2:1b) and heavier reasoning engines (Mistral, Phi-3, DeepSeek) with remembered preferences.</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>Complete control over your machine's GPU and RAM. Run lightweight models for sub-second replies on low battery, or crank up heavier weights for deep debugging.</p>
      </div>
    </div>

    <!-- Feature 3: Sloth Persona Tones -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="4" y1="21" x2="4" y2="14"/>
              <line x1="4" y1="10" x2="4" y2="3"/>
              <line x1="12" y1="21" x2="12" y2="12"/>
              <line x1="12" y1="8" x2="12" y2="3"/>
              <line x1="20" y1="21" x2="20" y2="16"/>
              <line x1="20" y1="12" x2="20" y2="3"/>
              <line x1="1" y1="14" x2="7" y2="14"/>
              <line x1="9" y1="8" x2="15" y2="8"/>
              <line x1="17" y1="16" x2="23" y2="16"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">Persona & Tone Engine</h3>
            <span class="feature-doc-badge">One-Click Steering</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>Appends specialized semantic steering vectors: <em>Ultra Lazy</em> (ultra-blunt, minimal words), <em>Corporate</em> (executive diplomatic etiquette), <em>Detailed</em> (exhaustive structural analysis), and <em>Chill</em> (friendly, stress-free conversational teammate).</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>No more prompt editing. Turn an angry complaint into a polite executive reply with one click, or boil a 10-page doc into two blunt sentences for fast decisions.</p>
      </div>
    </div>

    <!-- Feature 4: One-Click Lazy Actions -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="9 11 12 14 22 4"/>
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">One-Click Actions</h3>
            <span class="feature-doc-badge">Instant Dispatch</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>Instant workflow shortcuts built right on the output card. <code>Open Email</code> dispatches straight to your system mail client with prefilled subject/body. <code>Checklist</code> transforms messy bullet points into Notion/GitHub markdown checkboxes. <code>Save .md</code> downloads a timestamped document.</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>Eliminates copy-paste friction. A generated draft is only useful if you can send or save it instantly without manually reformatting task bullets.</p>
      </div>
    </div>

    <!-- Feature 5: Lazy Vault -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect width="20" height="14" x="2" y="5" rx="2"/>
              <line x1="2" y1="10" x2="22" y2="10"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">Lazy Vault</h3>
            <span class="feature-doc-badge">Offline Local History</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>An accessible slide-over drawer in your topbar storing your last 15 generations in browser <code>localStorage</code>. Provides instant one-click <code>Restore</code>, <code>Copy</code>, and granular deletion without external cookies or server databases.</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>Zero risk of lost work. If your browser closes or your laptop restarts, your previous drafts and code fixes are safely waiting in the drawer.</p>
      </div>
    </div>

    <!-- Feature 6: Zero-Prompt Intent Routing -->
    <div class="feature-doc-card">
      <div>
        <div class="feature-doc-head">
          <div class="feature-doc-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <div>
            <h3 class="feature-doc-title">Zero-Prompt Routing</h3>
            <span class="feature-doc-badge">Autonomous Intent Detection</span>
          </div>
        </div>
        <div class="feature-doc-section">
          <strong><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> How It Works</strong>
          <p>A dual-step local neural execution pipeline. Step 1 classifies the raw input text into <code>reply</code>, <code>summary</code>, <code>plan</code>, or <code>fix</code>. Step 2 automatically routes the payload into its tailored execution prompt with deterministic fallbacks.</p>
        </div>
      </div>
      <div class="feature-doc-why">
        <strong>Why It's Worth Having:</strong>
        <p>Cures prompt fatigue forever. You never have to explain the task to the model. Just paste and the sloth does the heavy thinking.</p>
      </div>
    </div>
  </div>

  <!-- Frequently Asked Questions (FAQ) Section -->
  <div class="section-header" style="margin-top: 60px;">
    <div class="section-badge">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <span>Knowledge Base</span>
    </div>
    <h2 class="section-title">Frequently Asked Questions</h2>
    <p class="section-subtitle">
      Common questions about offline privacy, model compatibility, Ollama commands, and local setup.
    </p>
  </div>

  <div class="faq-container">
    <!-- FAQ 1 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>Does Sloth Mode require an active internet connection?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p><strong>No.</strong> Sloth Mode is engineered to be 100% offline. Once you have installed Ollama and pulled your desired weights (e.g., <code>ollama pull llama3.2</code>), you can disconnect from Wi-Fi entirely. All neural inferences execute locally on your laptop's CPU or GPU. Zero telemetry, cookies, or prompt tokens are transmitted over the internet.</p>
      </div>
    </div>

    <!-- FAQ 2 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>How do I start Ollama if the offline status alert is showing?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p>1. Download Ollama from <a href="https://ollama.com" target="_blank" rel="noopener">ollama.com</a>.<br>
        2. Open a terminal and run <code>ollama serve</code> to start the local daemon.<br>
        3. In another terminal window, download the model with <code>ollama pull llama3.2:1b</code> or <code>ollama pull llama3.2</code>.<br>
        4. Refresh the page or start Sloth Mode — the offline banner will disappear automatically.</p>
      </div>
    </div>

    <!-- FAQ 3 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>How does Voice Dictation capture thoughts without repeating words?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p>Voice Dictation uses native client-side Web Speech recognition with a dual-buffer streaming pipeline. As you speak, interim hypothesis fragments update seamlessly in the textarea and only commit once speech pauses, preventing duplicated or repeated phrases.</p>
      </div>
    </div>

    <!-- FAQ 4 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>Can I use other models like Mistral, Gemma 2, or DeepSeek?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p><strong>Yes!</strong> Any model you install via Ollama (e.g., <code>ollama pull mistral</code>, <code>ollama pull phi3</code>, or <code>ollama pull deepseek-r1:1.5b</code>) is automatically detected through our <code>/api/models</code> endpoint and will show up in the top Model Selector dropdown in the workspace.</p>
      </div>
    </div>

    <!-- FAQ 5 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>Where is my Lazy Vault history stored?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p>All entries are stored strictly in your browser's private <code>localStorage</code> sandbox under key <code>sloth_vault_v1</code>. No databases, external servers, or tracking providers are involved. You can clear your entire vault anytime with the "Clear Vault" button.</p>
      </div>
    </div>

    <!-- FAQ 6 -->
    <div class="faq-card">
      <button type="button" class="faq-question">
        <span>How does Sloth Mode know whether to reply, summarize, plan, or fix?</span>
        <svg class="faq-icon-caret" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="faq-answer">
        <p>Sloth Mode executes a zero-shot classification prompt through your local model. Code blocks or stack traces are recognized as bugs requiring a <code>fix</code>; multi-speaker conversations become a <code>summary</code> with action items; unstructured to-do lists become a prioritized <code>plan</code>; and incoming messages or emails trigger a ready-to-send <code>reply</code> draft.</p>
      </div>
    </div>
  </div>
</section>

<footer>
  Sloth Mode · Autonomous Local AI · 100% Offline & Private · Made for getting your time back
</footer>

<!-- Lazy Vault Slide-Over Drawer -->
<div class="vault-backdrop" id="vaultBackdrop"></div>
<aside class="vault-drawer" id="vaultDrawer" aria-label="Saved Vault History">
  <div class="vault-drawer-head">
    <div class="vault-head-title">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect width="20" height="14" x="2" y="5" rx="2"/>
        <line x1="2" y1="10" x2="22" y2="10"/>
      </svg>
      <h3>Lazy Vault</h3>
      <span class="vault-offline-badge">Offline LocalStorage</span>
    </div>
    <button type="button" class="vault-close-btn" id="vaultCloseBtn" aria-label="Close Vault">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18"/>
        <line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
    </button>
  </div>
  <div class="vault-drawer-body" id="vaultList">
    <!-- Populated via JavaScript -->
  </div>
  <div class="vault-drawer-foot">
    <button type="button" class="vault-clear-btn" id="vaultClearBtn">Clear Vault</button>
  </div>
</aside>

<script>
'use strict';

// ---- Sloth levels ---------------------------------------------------------
var LEVELS = [
  { name: 'Baby Sloth',      xp: 0 },
  { name: 'Couch Sloth',     xp: 200 },
  { name: 'Zen Sloth',       xp: 600 },
  { name: 'Legendary Sloth', xp: 1500 }
];

var TYPE_BADGES = {
  reply: {
    label: 'Reply Draft',
    icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>'
  },
  summary: {
    label: 'Summary & Action Items',
    icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
  },
  plan: {
    label: 'Prioritised Plan',
    icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 16 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/></svg>'
  },
  fix: {
    label: 'Suggested Fix',
    icon: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 2 1.88 1.88"/><path d="M14.12 3.88 16 2"/><rect width="8" height="14" x="8" y="6" rx="4"/><path d="M4 13h4"/><path d="M16 13h4"/><path d="m5 19 3-2"/><path d="m19 19-3-2"/><path d="m5 7 3 2"/><path d="m19 7-3 2"/></svg>'
  }
};

var EXAMPLES = [
  "Hi Mohid,\n\nJust following up on the proposal I sent last Tuesday. The client is asking for a final answer by this Friday. Could you confirm if the pricing works, and whether you can join a 15-minute call tomorrow at 11am to close this?\n\nThanks,\nAlex",
  "Rahul: guys the trip plan is final, train is at 6am on saturday!!\nSneha: 6am?? who books a 6am train 😭\nRahul: it was the cheapest ok\nAmit: I still haven't booked my ticket btw\nSneha: AMIT.\nRahul: also hotel check-in is 12, we reach by 10 so we need to keep bags somewhere\nSneha: i'll call the hotel and ask for early check-in\nAmit: fine i'll book my ticket tonight, remind me\nRahul: and someone needs to carry the speaker\nSneha: not me, i carried it last time\nAmit: ok ok i'll take the speaker. also we need to split the hotel money, 2400 each\nRahul: send it before friday pls",
  "call the bank about the card thing, finish chapter 4 assignment ITS DUE TOMORROW, buy eggs and milk, reply to landlord about the tap, gym?? maybe, book train tickets for next month, water the plants, submit the form before 5pm today, fix the leaking tap someday, call mummy on sunday, renew the bike insurance this week it expires friday",
  "def get_average(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total / len(numbers)\n\nscores = []\nprint(get_average(scores))\n\n# Error: ZeroDivisionError: division by zero"
];

function el(id){ return document.getElementById(id); }

// ---- Progress (XP + minutes) persisted in localStorage --------------------
function loadProgress(){
  function num(key){ var v = parseInt(localStorage.getItem(key) || '0', 10); return isNaN(v) || v < 0 ? 0 : v; }
  return { xp: num('sloth_xp'), minutes: num('sloth_minutes'), tasks: num('sloth_tasks') };
}

function saveProgress(p){
  try {
    localStorage.setItem('sloth_xp', String(p.xp));
    localStorage.setItem('sloth_minutes', String(p.minutes));
    localStorage.setItem('sloth_tasks', String(p.tasks));
  } catch (e) { /* storage unavailable */ }
}

function renderProgress(){
  var p = loadProgress();
  var i = 0;
  for (var k = 0; k < LEVELS.length; k++){ if (p.xp >= LEVELS[k].xp) i = k; }
  var current = LEVELS[i], next = LEVELS[i + 1];
  el('levelName').textContent = current.name;
  if (next){
    el('xpLabel').textContent = p.xp + ' / ' + next.xp + ' XP';
    var pct = Math.max(0, Math.min(100, Math.round((p.xp - current.xp) / (next.xp - current.xp) * 100)));
    el('xpFill').style.width = pct + '%';
    el('nextHint').textContent = (next.xp - p.xp) + ' XP to ' + next.name;
  } else {
    el('xpLabel').textContent = p.xp + ' XP · Max level!';
    el('xpFill').style.width = '100%';
    el('nextHint').textContent = 'Legendary. Master of laziness.';
  }
  el('totalMinutes').textContent = p.minutes + ' min';
  el('totalTasks').textContent = p.tasks;
}
renderProgress();

// ---- Ollama status check & model selector --------------------------------
var currentModel = localStorage.getItem('sloth_model') || 'llama3.2:1b';
var modelSelectEl = el('modelSelect');

function initModelSelector() {
  fetch('/api/models')
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d && Array.isArray(d.models) && d.models.length > 0) {
        modelSelectEl.innerHTML = '';
        for (var m = 0; m < d.models.length; m++) {
          var opt = document.createElement('option');
          opt.value = d.models[m];
          opt.textContent = d.models[m];
          if (d.models[m] === currentModel) opt.selected = true;
          modelSelectEl.appendChild(opt);
        }
        if (!modelSelectEl.value && d.default) {
          modelSelectEl.value = d.default;
          currentModel = d.default;
        }
      }
    })
    .catch(function(){});

  if (modelSelectEl) {
    modelSelectEl.addEventListener('change', function() {
      currentModel = this.value;
      try { localStorage.setItem('sloth_model', currentModel); } catch (e) {}
    });
  }
}
initModelSelector();

fetch('/api/health').then(function(r){ return r.json(); }).then(function(d){
  if (!d.ollama) {
    el('ollamaPill').style.display = 'block';
  }
}).catch(function(){});

// ---- Sloth Tone / Persona Selector ----------------------------------------
var currentTone = localStorage.getItem('sloth_tone') || 'ultra-lazy';
var toneChips = document.querySelectorAll('.tone-chip');

function setTone(tone) {
  currentTone = tone;
  for (var i = 0; i < toneChips.length; i++) {
    var chip = toneChips[i];
    if (chip.getAttribute('data-tone') === tone) {
      chip.classList.add('active');
    } else {
      chip.classList.remove('active');
    }
  }
  try { localStorage.setItem('sloth_tone', currentTone); } catch (e) {}
}

for (var tc = 0; tc < toneChips.length; tc++) {
  toneChips[tc].addEventListener('click', function() {
    setTone(this.getAttribute('data-tone'));
  });
}
setTone(currentTone);

// ---- Voice Dictation ("Too Lazy to Type") ---------------------------------
var voiceBtn = el('voiceDictateBtn');
var voiceText = el('voiceDictateText');
var SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
var recognition = null;
var isRecording = false;
var baseInputText = '';

if (voiceBtn) {
  if (SpeechRec) {
    try {
      recognition = new SpeechRec();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = function() {
        isRecording = true;
        baseInputText = input.value ? (input.value.trim() + ' ') : '';
        voiceBtn.classList.add('listening');
        if (voiceText) voiceText.textContent = 'Listening… Speak';
      };

      recognition.onresult = function(event) {
        var finalTranscript = '';
        var interimTranscript = '';
        for (var i = 0; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + ' ';
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        input.value = baseInputText + finalTranscript + interimTranscript;
        input.dispatchEvent(new Event('input'));
      };

      recognition.onerror = function() {
        isRecording = false;
        voiceBtn.classList.remove('listening');
        if (voiceText) voiceText.textContent = 'Too Lazy to Type';
      };

      recognition.onend = function() {
        isRecording = false;
        voiceBtn.classList.remove('listening');
        if (voiceText) voiceText.textContent = 'Too Lazy to Type';
      };

      voiceBtn.addEventListener('click', function() {
        if (!isRecording) {
          try {
            recognition.start();
          } catch (e) {
            try { recognition.stop(); } catch(err){}
          }
        } else {
          recognition.stop();
        }
      });
    } catch (e) {
      voiceBtn.style.display = 'none';
    }
  } else {
    voiceBtn.addEventListener('click', function() {
      alert('Voice dictation works best in modern Chrome, Edge, or Safari browsers with microphone access enabled.');
    });
  }
}

// ---- Lazy Vault (Offline LocalStorage History) -----------------------------
var VAULT_KEY = 'sloth_vault_v1';
var vaultDrawer = el('vaultDrawer');
var vaultBackdrop = el('vaultBackdrop');
var vaultList = el('vaultList');
var vaultCountBadge = el('vaultCountBadge');
var lastResultData = null;

function getVault() {
  try {
    var raw = localStorage.getItem(VAULT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

function saveVault(list) {
  try {
    localStorage.setItem(VAULT_KEY, JSON.stringify(list));
  } catch (e) {}
}

function addToVault(item) {
  var list = getVault();
  item.id = Date.now();
  item.time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  list.unshift(item);
  if (list.length > 15) list = list.slice(0, 15);
  saveVault(list);
  renderVault();
}

function renderVault() {
  var list = getVault();
  if (vaultCountBadge) vaultCountBadge.textContent = list.length;
  if (!vaultList) return;

  if (list.length === 0) {
    vaultList.innerHTML = '<div class="vault-empty-state">' +
      '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
      '<rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/>' +
      '</svg>' +
      '<h4>Your vault is resting</h4>' +
      '<p>Every email, plan, summary, or fix your sloth completes will be safely stashed here offline.</p>' +
      '</div>';
    return;
  }

  var html = '';
  for (var i = 0; i < list.length; i++) {
    var entry = list[i];
    var preview = (entry.output || '').trim();
    if (preview.length > 140) preview = preview.substring(0, 140) + '…';

    html += '<div class="vault-card" data-vault-idx="' + i + '">' +
      '<div class="vault-card-head">' +
        '<div class="vault-card-tags">' +
          '<span class="vault-tag">' + (entry.label || entry.type || 'Task') + '</span>' +
          '<span class="vault-tag tone-tag">' + (entry.tone || 'standard') + '</span>' +
        '</div>' +
        '<span class="vault-time">' + (entry.time || '') + '</span>' +
      '</div>' +
      '<div class="vault-card-preview">' + escapeHtml(preview) + '</div>' +
      '<div class="vault-card-actions">' +
        '<button type="button" class="vault-mini-btn restore-btn" data-restore="' + i + '">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>' +
          '<span>Restore</span>' +
        '</button>' +
        '<button type="button" class="vault-mini-btn copy-vault-btn" data-copy="' + i + '">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>' +
          '<span>Copy</span>' +
        '</button>' +
        '<button type="button" class="vault-mini-btn danger del-vault-btn" data-del="' + i + '">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>' +
          '<span>Delete</span>' +
        '</button>' +
      '</div>' +
    '</div>';
  }
  vaultList.innerHTML = html;

  // Add click listeners to items
  var restoreBtns = vaultList.querySelectorAll('.restore-btn');
  for (var r = 0; r < restoreBtns.length; r++) {
    restoreBtns[r].addEventListener('click', function() {
      var idx = parseInt(this.getAttribute('data-restore'), 10);
      var itm = list[idx];
      if (itm) {
        input.value = itm.input || '';
        input.dispatchEvent(new Event('input'));
        displayResult(itm);
        toggleVault(false);
        el('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  var copyBtns = vaultList.querySelectorAll('.copy-vault-btn');
  for (var c = 0; c < copyBtns.length; c++) {
    copyBtns[c].addEventListener('click', function() {
      var idx = parseInt(this.getAttribute('data-copy'), 10);
      var itm = list[idx];
      if (itm && itm.output) {
        copyTextToClipboard(itm.output);
        this.innerHTML = '<span>Copied!</span>';
        var selfBtn = this;
        setTimeout(function(){
          selfBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg><span>Copy</span>';
        }, 1800);
      }
    });
  }

  var delBtns = vaultList.querySelectorAll('.del-vault-btn');
  for (var d = 0; d < delBtns.length; d++) {
    delBtns[d].addEventListener('click', function() {
      var idx = parseInt(this.getAttribute('data-del'), 10);
      list.splice(idx, 1);
      saveVault(list);
      renderVault();
    });
  }
}

function escapeHtml(text) {
  var div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function toggleVault(open) {
  if (open) {
    vaultDrawer.classList.add('open');
    vaultBackdrop.classList.add('open');
    renderVault();
  } else {
    vaultDrawer.classList.remove('open');
    vaultBackdrop.classList.remove('open');
  }
}

if (el('vaultToggleBtn')) {
  el('vaultToggleBtn').addEventListener('click', function(){ toggleVault(true); });
}
if (el('vaultCloseBtn')) {
  el('vaultCloseBtn').addEventListener('click', function(){ toggleVault(false); });
}
if (vaultBackdrop) {
  vaultBackdrop.addEventListener('click', function(){ toggleVault(false); });
}
if (el('vaultClearBtn')) {
  el('vaultClearBtn').addEventListener('click', function(){
    if (confirm('Clear all offline Lazy Vault entries?')) {
      saveVault([]);
      renderVault();
    }
  });
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape' && vaultDrawer && vaultDrawer.classList.contains('open')) {
    toggleVault(false);
  }
});
renderVault();

// ---- Input helpers ---------------------------------------------------------
var input = el('messyInput');
input.addEventListener('input', function(){
  el('charCount').textContent = input.value.length.toLocaleString() + ' / 30,000';
});
input.addEventListener('keydown', function(e){
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter'){ e.preventDefault(); el('taskForm').requestSubmit(); }
});

var exampleBtns = document.querySelectorAll('[data-example]');
for (var b = 0; b < exampleBtns.length; b++){
  exampleBtns[b].addEventListener('click', function(){
    input.value = EXAMPLES[Number(this.getAttribute('data-example'))];
    input.dispatchEvent(new Event('input'));
    input.focus();
  });
}

// ---- Display Result Helper ------------------------------------------------
function displayResult(d) {
  lastResultData = d;
  el('outputText').textContent = d.output;

  var badgeInfo = TYPE_BADGES[d.type] || { label: d.label || d.type, icon: '' };
  el('typeBadge').innerHTML = badgeInfo.icon + '<span>' + badgeInfo.label + '</span>';
  el('savedText').textContent = 'You saved ~' + d.minutes_saved + ' minutes!';
  el('xpAward').textContent = '+' + d.xp + ' XP';

  el('copyBtnText').textContent = 'Copy output';
  el('resultCard').style.display = 'block';
  var heroSec = el('heroSection') || document.querySelector('.hero-section');
  if (heroSec) heroSec.classList.add('has-result');
}

// ---- Submit: classify + run -----------------------------------------------
var loadingMsgs = [
  'Waking the sloth up…',
  'Analyzing syntax and structure with local AI…',
  'Classifying intent: reply, summary, plan, or fix…',
  'Executing persona tone & prompt locally…',
  'Polishing output and computing time saved…'
];

el('taskForm').addEventListener('submit', function(e){
  e.preventDefault();
  var text = input.value.trim();
  var errorBox = el('errorBox');
  errorBox.style.display = 'none';
  el('resultCard').style.display = 'none';
  var heroSec = el('heroSection') || document.querySelector('.hero-section');
  if (heroSec) heroSec.classList.remove('has-result');

  if (!text){
    el('errorMsgText').textContent = 'Paste something messy first — your sloth is waiting for work.';
    errorBox.style.display = 'block';
    input.focus();
    return;
  }

  var btn = el('submitBtn');
  btn.disabled = true;
  btn.innerHTML = '<span>Processing on device…</span>';
  el('loadingPanel').style.display = 'block';

  var msgIdx = 0;
  el('loadingText').textContent = loadingMsgs[0];
  var msgTimer = setInterval(function(){
    msgIdx = (msgIdx + 1) % loadingMsgs.length;
    el('loadingText').textContent = loadingMsgs[msgIdx];
  }, 3200);

  fetch('/api/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      tone: currentTone,
      model: modelSelectEl ? modelSelectEl.value : currentModel
    })
  })
  .then(function(r){ return r.json().then(function(d){ return { ok: r.ok, data: d }; }); })
  .then(function(res){
    if (!res.ok) throw new Error(res.data.error || 'Execution failed. Please check Ollama.');
    var d = res.data;
    displayResult(d);

    var p = loadProgress();
    p.xp += d.xp;
    p.minutes += d.minutes_saved;
    p.tasks += 1;
    saveProgress(p);
    renderProgress();

    // Auto-save to Lazy Vault
    addToVault({
      type: d.type,
      label: d.label,
      tone: d.tone || currentTone,
      model: d.model || currentModel,
      input: text,
      output: d.output,
      minutes_saved: d.minutes_saved,
      xp: d.xp
    });

    el('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  })
  .catch(function(err){
    el('errorMsgText').textContent = err.message || 'Something went wrong. Please check your Ollama service.';
    errorBox.style.display = 'block';
  })
  .finally(function(){
    clearInterval(msgTimer);
    el('loadingPanel').style.display = 'none';
    btn.disabled = false;
    btn.innerHTML = '<span>Let the sloth handle it</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';
  });
});

// ---- Clipboard Helper -----------------------------------------------------
function copyTextToClipboard(text, onDone, onFail) {
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(function(){
      if (typeof onDone === 'function') onDone();
    }).catch(function(){
      fallbackCopy(text, onDone, onFail);
    });
  } else {
    fallbackCopy(text, onDone, onFail);
  }
}

function fallbackCopy(text, onDone, onFail) {
  var ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
    if (typeof onDone === 'function') onDone();
  } catch (e) {
    if (typeof onFail === 'function') onFail();
  }
  ta.remove();
}

// ---- One-Click Lazy Actions Handlers ---------------------------------------

// 1. Copy Output Button
el('copyBtn').addEventListener('click', function(){
  var text = el('outputText').textContent;
  var btnText = el('copyBtnText');
  copyTextToClipboard(text, function(){
    btnText.textContent = 'Copied to clipboard!';
    setTimeout(function(){ btnText.textContent = 'Copy output'; }, 2200);
  }, function(){
    btnText.textContent = 'Select & copy text';
  });
});

// 2. Open in Email Client (mailto:)
if (el('emailBtn')) {
  el('emailBtn').addEventListener('click', function() {
    var text = el('outputText').textContent;
    var subject = lastResultData && lastResultData.label ? 'Sloth Mode: ' + lastResultData.label : 'Sloth Mode Draft';
    var mailtoUrl = 'mailto:?subject=' + encodeURIComponent(subject) + '&body=' + encodeURIComponent(text);
    window.location.href = mailtoUrl;
  });
}

// 3. Copy as Markdown Checklist
if (el('checklistBtn')) {
  el('checklistBtn').addEventListener('click', function() {
    var raw = el('outputText').textContent;
    var lines = raw.split('\n');
    var checklist = [];

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      // If line is already a checklist
      if (line.match(/^-\s*\[[ xX]\]/)) {
        checklist.push(line);
      } else if (line.match(/^[-*•]\s+/)) {
        checklist.push('- [ ] ' + line.replace(/^[-*•]\s+/, ''));
      } else if (line.match(/^\d+[.)]\s+/)) {
        checklist.push('- [ ] ' + line.replace(/^\d+[.)]\s+/, ''));
      } else if (line.toLowerCase().indexOf('action items:') === -1 && line.toLowerCase().indexOf('why:') === -1) {
        checklist.push('- [ ] ' + line);
      }
    }

    var checklistText = checklist.join('\n');
    if (!checklistText) checklistText = '- [ ] ' + raw;

    var btnSpan = el('checklistBtnText');
    copyTextToClipboard(checklistText, function() {
      if (btnSpan) btnSpan.textContent = 'Copied Checklist!';
      setTimeout(function() {
        if (btnSpan) btnSpan.textContent = 'Checklist';
      }, 2000);
    });
  });
}

// 4. Download Markdown File (.md)
if (el('downloadMdBtn')) {
  el('downloadMdBtn').addEventListener('click', function() {
    var text = el('outputText').textContent;
    var task = lastResultData ? lastResultData.type : 'output';
    var label = lastResultData ? lastResultData.label : 'Sloth Task';
    var tone = lastResultData ? lastResultData.tone : currentTone;
    var model = lastResultData ? lastResultData.model : currentModel;
    var minutes = lastResultData ? lastResultData.minutes_saved : '0';

    var mdContent = '# Sloth Mode — ' + label + '\n\n' +
      '- **Task Type:** ' + task + '\n' +
      '- **Persona Tone:** ' + tone + '\n' +
      '- **Model Used:** ' + model + '\n' +
      '- **Estimated Time Saved:** ~' + minutes + ' minutes\n' +
      '- **Timestamp:** ' + new Date().toLocaleString() + '\n\n' +
      '---\n\n' +
      text + '\n';

    var blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'sloth-' + task + '-' + Date.now() + '.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

// ---- Meditating Sloth Lottie Loader (Hero + Header Brand) ------------------
(function initSlothLottie() {
  function setupLottie(containerId, playerId) {
    var container = document.getElementById(containerId);
    var player = document.getElementById(playerId);

    function fallbackLottieWeb() {
      if (window.lottie && container && !container.querySelector('svg')) {
        container.innerHTML = '';
        lottie.loadAnimation({
          container: container,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: '/static/sloth-meditation.json'
        });
      }
    }

    if (player) {
      player.addEventListener('ready', function() {
        if (typeof player.play === 'function') player.play();
      });
      player.addEventListener('load', function() {
        if (typeof player.play === 'function') player.play();
      });
      player.addEventListener('error', function() {
        fallbackLottieWeb();
      });
      setTimeout(function() {
        if (typeof player.play === 'function') player.play();
        var hasSvg = container && (container.querySelector('svg') || (player && player.shadowRoot && player.shadowRoot.querySelector('svg')));
        if (!hasSvg) {
          fallbackLottieWeb();
        }
      }, 250);
    } else {
      fallbackLottieWeb();
    }
  }

  setupLottie('slothLottieContainer', 'slothLottiePlayer');
  setupLottie('brandLottieWrap', 'brandLottiePlayer');
})();

// ---- FAQ Accordion Toggle -------------------------------------------------
(function initFAQ() {
  var faqCards = document.querySelectorAll('.faq-card');
  for (var f = 0; f < faqCards.length; f++) {
    (function(card) {
      var btn = card.querySelector('.faq-question');
      if (btn) {
        btn.addEventListener('click', function() {
          var isOpen = card.classList.contains('active');
          for (var j = 0; j < faqCards.length; j++) {
            faqCards[j].classList.remove('active');
          }
          if (!isOpen) {
            card.classList.add('active');
          }
        });
      }
    })(faqCards[f]);
  }
})();

</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return PAGE_HTML


@app.get("/sloth-meditation.json")
def sloth_animation_route():
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return send_from_directory(static_dir, "sloth-meditation.json")


@app.get("/api/health")
def health():
    """Let the page warn the user early if Ollama is not running."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        return jsonify(ollama=r.ok)
    except requests.RequestException:
        return jsonify(ollama=False)


@app.get("/api/models")
def get_models():
    """Return list of locally downloaded Ollama models or fast defaults."""
    default_models = ["llama3.2:1b", "llama3.2:latest", "mistral", "phi3"]
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if r.ok:
            data = r.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            if models:
                return jsonify(models=models, default=models[0], online=True)
    except Exception:
        pass
    return jsonify(models=default_models, default="llama3.2:1b", online=False)


@app.post("/api/process")
def process():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str) \
            or not data["text"].strip():
        return jsonify(
            error="Paste a message, chat, to-do list or code to get started."
        ), 400

    text = data["text"].strip()
    tone = data.get("tone", "ultra-lazy")
    model = data.get("model", MODEL)

    if len(text) > MAX_CHARS:
        return jsonify(
            error="That's a lot for one sloth! Please keep it under "
                  "30,000 characters."
        ), 400

    try:
        task = classify_text(text, model=model)               # step 1: auto-detect the type
        output = run_task(task, text, tone=tone, model=model) # step 2: run the matching prompt
    except OllamaError as exc:
        return jsonify(error=exc.message), exc.status

    minutes = estimate_minutes_saved(text, task)
    return jsonify(
        type=task,
        label=TASK_LABELS[task],
        output=output,
        minutes_saved=minutes,
        xp=minutes * 10,
        tone=tone,
        model=model,
    )


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("Sloth Mode is awake at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
