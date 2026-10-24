# 🦥 Sloth Mode

**Do less. Get it done.**

Sloth Mode is a tiny local web app for the gloriously lazy. Paste anything messy — an email you don't want to answer, a 200-message WhatsApp thread, a brain-dump to-do list, or code with a bug — and it does the boring work for you.

You never pick a mode. Sloth Mode first asks a **local AI model** to classify your paste, then automatically runs the matching task:

| What you paste | What you get back |
|---|---|
| ✉️ An email / message | A ready-to-send reply draft |
| 💬 A long text / chat thread | A short summary + bullet list of action items |
| 🧠 A messy to-do list | A clean, prioritised plan (most important first) |
| 🐛 Code with a bug | A suggested fix with a one-line explanation |

After every task it estimates how many minutes you saved — **"🦥 You saved ~12 minutes!"** — and awards **Sloth XP** (10 XP per minute saved).

**Levels:** Baby Sloth (0 XP) → Couch Sloth (200 XP) → Zen Sloth (600 XP) → Legendary Sloth (1,500 XP). Your total XP and minutes saved are stored in your browser's `localStorage`, so your progress is still there when you come back.

## 100% local. No API keys. No accounts.

Everything runs on your own laptop. The AI is [Ollama](https://ollama.com) running the open-weight **Llama 3.2** model — your pasted text never leaves your machine, and there are no external services, API keys, or sign-ups involved.

## Setup

### 1. Install Ollama

Download and install Ollama for your system from [ollama.com/download](https://ollama.com/download), then pull the model (one-time download, needs internet):

```bash
ollama pull llama3.2
```

Make sure Ollama is running (open the Ollama app, or run `ollama serve` in a terminal).

### 2. Install the Python dependencies

You need Python 3.10 or newer. From this folder:

```bash
pip install -r requirements.txt
```

### 3. Run Sloth Mode

```bash
python app.py
```

### 4. Open it in your browser

```
http://localhost:5000
```

Paste something messy, hit **"Let the sloth handle it"**, and take the credit.

## How it works

```
your messy paste
      │
      ▼
POST /api/process  (Flask, app.py)
      │
      ├─ 1. Classify — Ollama (Llama 3.2) returns one word:
      │        reply / summary / plan / fix
      │
      └─ 2. Run — the matching prompt produces the reply draft,
               summary, plan, or fix
      │
      ▼
result card + "You saved ~N minutes!" + XP added to your level bar
```

- **Backend:** Python + Flask, a single `app.py` (the single-page frontend — HTML/CSS/vanilla JS — is served by Flask from the same file).
- **AI:** Ollama at `http://localhost:11434/api/generate`, model `llama3.2`, non-streaming.
- **Time saved** is a rough, transparent estimate: a per-task base plus the input's word count at a manual-effort rate (reply 3 + words/90, summary 2 + words/150, plan 4 + words/80, fix 6 + words/50), rounded up and capped at 60 minutes. It's an estimate, not a measurement.
- **XP** = minutes saved × 10. Progress lives only in your browser's `localStorage` (`sloth_xp`, `sloth_minutes`, `sloth_tasks`) — clearing site data resets it.
- The app binds to `127.0.0.1` only. It's a personal, local tool by design.

## Troubleshooting

- **"🦥 Ollama is not running yet"** — Start Ollama (app or `ollama serve`), then try again. The pill at the top of the page also warns you if Ollama is offline.
- **"🦥 Llama 3.2 is not downloaded yet"** — Run `ollama pull llama3.2`.
- **First task is slow** — That's normal; the model loads into memory on the first request. Later tasks are faster.
- **Port 5000 already in use** — Close the other app using it, or change the port at the bottom of `app.py`.

## Project structure

```
sloth-mode/
├── app.py            # the whole app — Flask backend + single-page frontend
├── requirements.txt  # flask, requests
├── LICENSE           # MIT
└── README.md
```

## Why open matters

Sloth Mode only exists because the model is open: Llama 3.2 runs locally through Ollama, so a tool that reads your private emails, chats and to-do lists never has to send them anywhere. Open models turn "AI assistant" from a subscription you rent into software you actually own — private by architecture, not by promise.

## License

MIT — see [LICENSE](LICENSE).
