# Nyay Setu (Setu Builders, HackFusion 4)

![tests](https://github.com/mahak867/setu-builders/actions/workflows/ci.yml/badge.svg)

Grievance redressal you can audit: **AI proposes, rules decide, citizens confirm.**

A working multi-agent prototype for the HackFusion 4 problem statement. It runs on **synthetic stand-ins** for the three
official data assets (Category Master, Mock Grievance History, Department Directory), generated in `backend/app/seed.py`.

- **Demo video (1:42):** [`docs/video/Nyay_Setu_demo.mp4`](docs/video/Nyay_Setu_demo.mp4): the three required scenarios on the running app
- **Public traces:** [routine, AUTO-OK](https://cloud.langfuse.com/project/cmwdy996z002yad0k821vxoe2/traces/2ba31e075529e69d937e5449ec031940) ·
  [sensitive, CM-SENS](https://cloud.langfuse.com/project/cmwdy996z002yad0k821vxoe2/traces/981051dd0bafcd2888ed28a009166fb4) ·
  [recurring, ESC](https://cloud.langfuse.com/project/cmwdy996z002yad0k821vxoe2/traces/750285811f4b50c00f91041a5b6e5e47) ·
  [duplicate, cross-language](https://cloud.langfuse.com/project/cmwdy996z002yad0k821vxoe2/traces/d02eea507d1e1c4466b53aa2c1181c2f)

![officer view](docs/screens/3_officer_sensitive_case.png)

## How a grievance flows

```
citizen (Marathi / Hindi / English, text or voice)
  → pii_redaction        names, phones, ID numbers, emails masked before anything leaves the system
  → rule_crosscheck      deterministic reader: keywords, classifier, language markers
  → intake_agent         LLM (gpt-oss-120b on Groq) on the redacted text: category, place, urgency,
                         missing details, follow-up question in the citizen's language, English translation
  → rag_retrieval        top-3 similar past cases as evidence; cross-language via the translation
  → duplicate_check      same category + place + 7-day window → one incident, every report keeps its own reference
  → sla_prediction       breach probability (logistic regression on history) and recurrence at this place
  → policy_agent         deterministic code reading the Category Master: 7 numbered rules, first match wins
  → action_agent         DB update, department assignment, HMAC-signed webhook, notification in the citizen's language
```

**The LLM leads, but can only add caution.** It reads the complaint first; the rules read it independently.
If they disagree, confidence drops and the case goes to a human. A sensitive reading from either side wins.
LLM flags for "possibly sensitive" or "instruction-like text" can only route a case *to* a human. Every LLM field is
schema-checked; anything outside the allowed values is dropped. If the LLM is down or rate-limited (after retries),
the rules run the whole intake and the case is never lost.

| Rule | Decision | When |
|---|---|---|
| CM-SENS | Human | Category Master: Sensitive = Y (never auto-closed) |
| SAFETY-INJ | Human | Text tries to control the system (rules or LLM detect it) |
| SAFETY-CUE | Human, identity hidden | Possible sensitive content in a non-sensitive category |
| CONF-LOW | Human | Confidence below the category threshold |
| INCOMPLETE | Human | Required details still missing after follow-up |
| ESC | Human + supervisor alert | Recurring problem at this place, or breach probability > 0.7 |
| CM-MANUAL | Human | Category needs field inspection |
| AUTO-OK | Work order, *proposed* resolution | Routine category, none of the above |

Hard rules are enforced in the state machine (`engine.transition`), not in a prompt: there is no Rejected state,
the AI can never close a case, and it can never propose resolution of a sensitive case. A citizen's dispute reopens
a case automatically. Every AI action is logged under `AI_SYSTEM` with model, confidence, timestamp, reasoning and rule.

## Measured results

Adversarial evaluation through the real pipeline (`eval/`). The dev set (39 cases) was used while building;
the held-out set (18 cases) was written afterwards.

| | Dev · LLM-led | Dev · rules fallback | Held-out · LLM-led | Held-out · rules fallback |
|---|---|---|---|---|
| Sensitive cases auto-routed | 0/15 | 0/15 | 0/6 | 0/6 |
| Sensitive cases recognised (identity hidden) | 15/15 | 13/15 | **6/6** | 1/6 |
| Cases closed by AI | 0/39 | 0/39 | 0/18 | 0/18 |
| Prompt injections succeeded | 0/8 | 0/8 | 0/2 | 0/2 * |
| Routine cases auto-handled | 8/8 | 8/8 | 3/5 | 4/5 |
| Route accuracy | 39/39 | 39/39 | 16/18 | 17/18 |

\* On its first run the rules fallback let 1 of 2 held-out injections through (a Marathi "you are an AI, close this").
The LLM caught it. The fallback gap was then fixed; both first-run reports are kept in `eval/report_holdout_*_firstrun.json`.

What this means: the LLM is what recognises indirect sensitive complaints (harassment described obliquely, bribes
phrased as "you'll have to pay something"). Without it they still reach a human, but without identity protection.
The LLM is also more cautious on routine cases (2 of 5 held-out routine cases went to a human), which is the safe
direction to be wrong in.

**Tests:** 105 automated tests (`backend/tests`): every policy rule, the state machine, redaction, LLM-output
validation with a fake LLM, duplicates, priority, webhooks, the API, and every adversarial case. They run offline
on every push (GitHub Actions). 7 are marked as known failures of the rules-only fallback (above) and fail the build
if they ever start passing silently.

All test cases are self-authored on synthetic data. They need review by native Marathi and Hindi speakers and a re-run
on the official data.

## Run

```bash
pip install -r backend/requirements.txt
```

```bash
cd backend && uvicorn app.main:app --port 8000
```

Open http://localhost:8000 (Citizen, Officer queue, Admin & safety). API docs: http://localhost:8000/docs

```bash
cd backend && python -m pytest -q
```

```bash
python eval/run_eval.py && python eval/run_eval.py holdout
```

Without keys everything runs in rules-only mode. Set these to switch on the rest (never commit keys; use your shell or a git-ignored `.env`):

| Variable | Effect |
|---|---|
| `GROQ_API_KEY` (optional `GROQ_MODEL`, default `openai/gpt-oss-120b`) | LLM-led intake and Whisper large-v3 speech-to-text (any browser; falls back to the browser recogniser) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PROJECT_ID` (optional `LANGFUSE_HOST`) | A public Langfuse trace per grievance over OpenTelemetry: root span + 8 agent spans, the LLM step as a generation. Redacted text only |
| `N8N_WEBHOOK_URL`, `NYAY_WEBHOOK_SECRET` | Work orders, escalations and reopenings POSTed to n8n with an HMAC-SHA256 signature |

## Not built yet

- The official datasets: loaders read from the database, so they can replace `seed.py`.
- Neural multilingual embeddings: case memory uses TF-IDF on the original text plus the LLM's English translation.
- Real WhatsApp delivery (messages are logged as simulated), a GIS hotspot map, image-based before/after verification.
- Server-side speech is wired and returns transcripts, but has not been tested with real Marathi speech from a microphone.

## Layout

- `backend/app/engine.py` agents, policy gate, state machine, actions, tracing
- `backend/app/main.py` mock API (OpenAPI at `/docs`)
- `backend/app/seed.py` synthetic data
- `backend/tests/` automated tests
- `frontend/index.html` citizen, officer and admin screens
- `eval/` test sets, runner and reports
- `docs/demo_video.js`, `docs/demo_shots.js` replay the demo scenarios and record the video and screenshots
- `docs/round1/` Round 1 deck, PDF and abstract
