# Nyay Setu (Setu Builders, HackFusion 4)

Grievance redressal you can audit: **AI proposes, rules decide, citizens confirm.**

This is a working prototype. It runs on **synthetic stand-ins** for the three official data assets
(Category Master, Mock Grievance History, Department Directory), generated in `backend/app/seed.py`.

## Run

```bash
pip install -r backend/requirements.txt
```

```bash
cd backend && uvicorn app.main:app --port 8000
```

Open http://localhost:8000 (Citizen, Officer queue, Admin & safety). API docs: http://localhost:8000/docs

```bash
python eval/run_eval.py
```

## What works today

| Capability | Status |
|---|---|
| Conversational intake (Marathi, Hindi, English, Hinglish), follow-up for missing details, citizen confirms draft | Working, rule-based extractor + TF-IDF classifier. Voice uses the browser speech API (Chrome/Edge) |
| Policy gate from the Category Master (Sensitive, Identity-Hiding, confidence threshold) | Working, deterministic code, 7 rules with IDs |
| Case memory: top-3 similar past cases as evidence | Working, TF-IDF character n-grams (lexical, not yet cross-lingual) |
| Duplicate grouping by category + location + 7-day window; every case keeps its own reference | Working |
| SLA breach prediction and recurrence escalation | Working, logistic regression on mock history |
| Actions: DB update, department assignment, HMAC-signed webhook, simulated WhatsApp/email | Working; webhook is logged unless `N8N_WEBHOOK_URL` is set |
| AI_SYSTEM audit log (model, confidence, timestamp, reasoning, rule) | Working |
| Dispute reopens automatically; AI can never close or reject | Working, enforced in `engine.transition` |
| Safety evaluation, 39 adversarial cases | Working, results on the Admin page |

## Optional: LLM and Langfuse (written, not yet tested with real keys)

Set these before starting the server. Without them the app runs exactly as above.

| Variable | Effect |
|---|---|
| `GROQ_API_KEY` (and optional `GROQ_MODEL`) | An LLM gives a second opinion on the category and writes the summary. Disagreement only lowers confidence; the policy gate stays in code. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (optional `LANGFUSE_HOST`, `LANGFUSE_PROJECT_ID`) | Each grievance's trace (redacted text only) is exported to Langfuse as a public trace. |
| `N8N_WEBHOOK_URL`, `NYAY_WEBHOOK_SECRET` | Work orders and escalations are POSTed to n8n with an HMAC signature. |

## Not built yet

- Multilingual embeddings (multilingual-e5): case memory uses TF-IDF. The LLM and Langfuse hooks above are untested until keys are added.
- Server-side speech-to-text (Sarvam/Whisper), PostgreSQL/pgvector, real WhatsApp, hotspot map, React PWA.
- The eval set is self-authored; it needs native-speaker review and the official data. Known miss: one Marathi sentence detected as Hindi.

## Layout

- `backend/app/engine.py` agents, policy gate, state machine, actions
- `backend/app/main.py` mock API
- `backend/app/seed.py` synthetic data
- `frontend/index.html` the three screens
- `eval/` test cases and runner
- `docs/demo_shots.js` replays the three demo scenarios and saves screenshots to `docs/screens/`
