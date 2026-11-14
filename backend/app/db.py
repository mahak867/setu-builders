import json
import os
import sqlite3
import threading

PATH = os.environ.get("NYAY_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nyay.db"))
_local = threading.local()

SCHEMA = """
create table if not exists category_master(id text primary key, data text);
create table if not exists department(name text primary key, data text);
create table if not exists history(id integer primary key, text text, text_en text, lang text, category_id text, locality text,
  submitted text, resolved text, status text, resolution text, feedback text, breached int);
create table if not exists grievance(id integer primary key, ref text unique, created text, status text, data text);
create table if not exists audit(id integer primary key, ref text, ts text, actor text, action text, model text,
  confidence real, reasoning text, rule_id text);
create table if not exists notification(id integer primary key, ref text, ts text, channel text, recipient text, body text);
create table if not exists webhook(id integer primary key, ref text, ts text, event text, payload text, signature text, delivery text);
"""


def conn():
    c = getattr(_local, "c", None)
    if c is None:
        c = sqlite3.connect(PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        _local.c = c
    return c


def init():
    conn().executescript(SCHEMA)
    conn().commit()


def q(sql, args=()):
    return [dict(r) for r in conn().execute(sql, args).fetchall()]


def x(sql, args=()):
    cur = conn().execute(sql, args)
    conn().commit()
    return cur.lastrowid


def load_g(ref):
    r = q("select * from grievance where ref=?", (ref,))
    if not r:
        return None
    g = json.loads(r[0]["data"])
    g.update(ref=r[0]["ref"], created=r[0]["created"], status=r[0]["status"])
    return g


def save_g(g):
    d = {k: v for k, v in g.items() if k not in ("ref", "created", "status")}
    x("update grievance set status=?, data=? where ref=?", (g["status"], json.dumps(d, ensure_ascii=False), g["ref"]))


def all_g():
    out = []
    for r in q("select * from grievance order by id desc"):
        g = json.loads(r["data"])
        g.update(ref=r["ref"], created=r["created"], status=r["status"])
        out.append(g)
    return out
