import os
import sys
import tempfile

import pytest

os.environ["NYAY_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
for k in ("GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "N8N_WEBHOOK_URL"):
    os.environ.pop(k, None)  # tests are deterministic and offline; the LLM is faked where needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, engine  # noqa: E402

engine.bootstrap()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    for t in ("grievance", "audit", "notification", "webhook"):
        db.x(f"delete from {t}")
    engine.SESS.clear()
    yield
