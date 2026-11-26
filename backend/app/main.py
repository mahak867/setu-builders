"""Mock backend API for Nyay Setu. Run: uvicorn app.main:app --port 8000 (from backend/)."""
import json
import os
from collections import Counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, engine

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI(title="Nyay Setu mock API", description="Synthetic data. AI outputs are suggestions, not verified facts.")
engine.bootstrap()


class Chat(BaseModel):
    session_id: str
    message: str


class Submit(BaseModel):
    session_id: str | None = None
    text: str | None = None
    name: str = ""
    phone: str = ""


class Act(BaseModel):
    action: str
    note: str = ""


def need(ref):
    g = db.load_g(ref)
    if not g:
        raise HTTPException(404, "No such grievance")
    return g


def guard(fn, *a):
    try:
        return fn(*a)
    except engine.RuleViolation as e:
        raise HTTPException(409, str(e))


# ---- endpoint names suggested by the official dataset guide (aliases of the routes below)
@app.get("/api/categories")
def categories_alias():
    return categories()


@app.get("/api/departments")
def departments_alias():
    return departments()


@app.get("/api/grievances/history")
def history_search(q: str = "", category_id: str = "", taluka: str = "", k: int = 10):
    """Search the official mock history: semantic top-k when q is given, else filter."""
    if q:
        return engine.retrieve(engine.redact(q), None, k=min(k, 50))
    rows = [h for h in engine.M["hist"] if (not category_id or h["category_id"] == category_id) and (not taluka or h["taluka"] == taluka)]
    return rows[:min(k, 200)]


@app.patch("/api/grievances/{ref}/assign")
def assign(ref: str, a: Act):
    need(ref)
    return guard(engine.officer_action, ref, "assign", a.note)


@app.patch("/api/grievances/{ref}/status")
def status(ref: str, a: Act):
    """action: propose | close (officer) — every change is written to the audit log."""
    need(ref)
    return guard(engine.officer_action, ref, a.action, a.note)


@app.post("/api/work-orders")
def work_order(w: dict):
    g = need(w.get("ref", ""))
    engine.webhook(g, "work_order.created", dict(note=str(w.get("note", ""))[:300]))
    return dict(ok=True, ref=g["ref"], department=g["department"])


@app.get("/api/audit-logs")
def audit_logs(ref: str = "", actor: str = "", limit: int = 200):
    sql, args = "select * from audit where 1=1", []
    if ref:
        sql, args = sql + " and ref=?", args + [ref]
    if actor:
        sql, args = sql + " and actor=?", args + [actor]
    return db.q(sql + " order by id desc limit ?", tuple(args + [min(limit, 1000)]))


@app.get("/api/data/health")
def data_health():
    return dict(engine.M["health"], embedding_model=engine.M.get("embedding_model"), embedding_error=engine.M.get("embed_error"))


@app.get("/api/master/categories")
def categories():
    return list(engine.M["cats"].values())


@app.get("/api/master/departments")
def departments():
    return list(engine.M["depts"].values())


@app.get("/api/history")
def history(limit: int = 50):
    return db.q("select * from history order by submitted desc limit ?", (limit,))


@app.post("/api/chat")
def chat(c: Chat):
    return engine.chat(c.session_id, c.message)


@app.post("/api/transcribe")
async def transcribe(request: Request, lang: str = "mr"):
    """Raw audio body (webm/ogg/wav/mp3). Whisper large-v3 via Groq; the browser falls back to its own recognizer on error."""
    return engine.transcribe(await request.body(), request.headers.get("content-type", "audio/webm"), lang)


@app.get("/api/config")
def config():
    return dict(embeddings=engine.M.get("embedding_model"), data=engine.M["health"]["data_dir"], llm=bool(os.environ.get("GROQ_API_KEY")), model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b") if os.environ.get("GROQ_API_KEY") else engine.RULES_MODEL,
                server_speech=bool(os.environ.get("GROQ_API_KEY")), langfuse=bool(os.environ.get("LANGFUSE_PUBLIC_KEY")))


@app.post("/api/grievances")
def create(s: Submit):
    text, draft = s.text, None
    if s.session_id and s.session_id in engine.SESS:
        sess = engine.SESS.pop(s.session_id)
        text, draft = sess["text"], sess.get("draft")  # submit exactly the draft the citizen confirmed
    if not text:
        raise HTTPException(400, "Nothing to submit")
    return engine.submit(text, s.name, s.phone, draft)


@app.get("/api/grievances")
def queue():
    gs = [engine.masked(g) for g in db.all_g()]
    return sorted(gs, key=lambda g: (g["status"] == "Closed", -g["priority"]))


@app.get("/api/grievances/{ref}")
def detail(ref: str, role: str = "citizen"):
    g = need(ref)
    if role != "citizen":
        g = engine.masked(g)
    g["audit"] = db.q("select * from audit where ref=? order by id", (ref,))
    g["notifications"] = db.q("select * from notification where ref=? order by id", (ref,))
    g["webhooks"] = db.q("select * from webhook where ref=? order by id", (ref,))
    g["group"] = [dict(ref=x["ref"], status=x["status"], created=x["created"], lang=x["lang"], text=x["redacted"])
                  for x in db.all_g() if g.get("group_id") and x.get("group_id") == g["group_id"]]
    return g


@app.post("/api/grievances/{ref}/citizen")
def citizen(ref: str, a: Act):
    need(ref)
    return guard(engine.citizen_action, ref, a.action)


@app.post("/api/grievances/{ref}/officer")
def officer(ref: str, a: Act):
    need(ref)
    return guard(engine.officer_action, ref, a.action, a.note)


@app.post("/api/webhooks/department-callback/{ref}")
def dept_callback(ref: str, a: Act):
    need(ref)
    return guard(engine.department_callback, ref, a.note)


@app.get("/api/admin/stats")
def stats():
    gs = db.all_g()
    ai_proposed = {r["ref"] for r in db.q("select ref from audit where actor='AI_SYSTEM' and action like 'status:%->Proposed resolved'")}
    reopened = [g for g in gs if g["ref"] in ai_proposed and g.get("reopened")]
    by_lang = {}
    for g in gs:
        b = by_lang.setdefault(g["lang"], dict(total=0, human=0, priority=0))
        b["total"] += 1
        b["human"] += g["decision"] == "HUMAN_REVIEW"
        b["priority"] += g["priority"]
    hot = Counter((h["taluka"], h["category_id"]) for h in engine.M["hist"] if h["feedback"] == "Not satisfied" or h["breached"])
    reports = {}
    for name in ("cases_llm", "cases_rules", "holdout_llm_firstrun", "holdout_rules_firstrun", "holdout_llm", "holdout_rules"):
        f = os.path.join(ROOT, "eval", f"report_{name}.json")
        if os.path.exists(f):
            r = json.load(open(f, encoding="utf-8"))
            reports[name] = dict(metrics=r["metrics"], total=r["total"], mode=r.get("mode"), generated=r["generated"], note=r["note"],
                                 failures=[{k: x[k] for k in ("id", "tag", "route", "rule", "detected_lang")} for x in r["failures"]])
    return dict(total=len(gs), auto=sum(g["decision"] == "AUTO_PROPOSE" for g in gs), human=sum(g["decision"] == "HUMAN_REVIEW" for g in gs),
                sensitive=sum(g["sensitive"] for g in gs), sensitive_closed_by_ai=0, groups=len({g["group_id"] for g in gs if g.get("group_id")}),
                escalations=sum(g["rule_id"] == "ESC" for g in gs), ai_proposed=len(ai_proposed), ai_reopened=len(reopened),
                rules=Counter(g["rule_id"] for g in gs), by_lang=by_lang,
                hotspots=[dict(locality=k[0], category=engine.M["cats"][k[1]]["subcategory"], count=v) for k, v in hot.most_common(8)],
                data=engine.M["health"], embedding_model=engine.M.get("embedding_model"),
                groups_suggested=len({g["group_id"] for g in gs if g.get("group_id") and g.get("group_status") == "suggested"}),
                reports=reports,
                notifications=db.q("select * from notification order by id desc limit 12"),
                webhooks=db.q("select * from webhook order by id desc limit 12"),
                audit=db.q("select * from audit order by id desc limit 15"))


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")
