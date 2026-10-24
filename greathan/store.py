"""Durable journal and transactional notification outbox (SQLite WAL)."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY, episode_id TEXT NOT NULL UNIQUE,
                  run_id TEXT NOT NULL, camera_id TEXT NOT NULL, zone_id TEXT NOT NULL,
                  event_type TEXT NOT NULL, track_id INTEGER NOT NULL,
                  observed_start TEXT NOT NULL, confirmed_at TEXT NOT NULL,
                  last_seen TEXT NOT NULL, ended_at TEXT,
                  source_start REAL NOT NULL, source_confirmed REAL NOT NULL,
                  source_last REAL NOT NULL, frame_id INTEGER NOT NULL,
                  snapshot_uri TEXT NOT NULL, lifecycle_state TEXT NOT NULL,
                  backend_kind TEXT NOT NULL, model_version TEXT NOT NULL,
                  config_version TEXT NOT NULL, confidence REAL NOT NULL,
                  reason TEXT NOT NULL, review_status TEXT NOT NULL DEFAULT 'new',
                  reviewed_at TEXT, review_note TEXT NOT NULL DEFAULT '',
                  evidence_latency_s REAL NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_filter ON events(camera_id,zone_id,event_type,confirmed_at);
                CREATE TABLE IF NOT EXISTS notifications (
                  cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id TEXT NOT NULL UNIQUE REFERENCES events(id), created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS diagnostics (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                  run_id TEXT, camera_id TEXT, severity TEXT NOT NULL, code TEXT NOT NULL, message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
                  camera_id TEXT NOT NULL, backend TEXT NOT NULL, status TEXT NOT NULL,
                  config_json TEXT NOT NULL, metrics_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                  event_id TEXT PRIMARY KEY REFERENCES events(id),
                  displayed_at TEXT NOT NULL, create_to_display_ack_s REAL NOT NULL
                );
            """)
            # An interrupted process must not leave historical events pretending to be active.
            db.execute("UPDATE events SET lifecycle_state='interrupted', ended_at=last_seen WHERE lifecycle_state='active'")
            db.execute("UPDATE runs SET status='interrupted', ended_at=? WHERE status='running'", (utc_now(),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_event(self, event):
        event = {**event, "created_at": utc_now()}
        keys = list(event)
        with self.connect() as db:
            result = db.execute(f"INSERT INTO events ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)}) ON CONFLICT DO NOTHING", [event[k] for k in keys])
            if result.rowcount:
                db.execute("INSERT INTO notifications(event_id,created_at) VALUES (?,?)",(event["id"],event["created_at"]))
            return bool(result.rowcount)

    def update_event(self, event_id, last_seen, source_last, ended_at=None, state="active"):
        with self.connect() as db:
            db.execute("UPDATE events SET last_seen=?,source_last=?,ended_at=?,lifecycle_state=? WHERE id=?",
                       (last_seen,source_last,ended_at,state,event_id))

    def get(self, event_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM events WHERE id=?",(event_id,)).fetchone()
            return dict(row) if row else None

    def evidence_latency(self,event_id,seconds):
        with self.connect() as db:
            db.execute("UPDATE events SET evidence_latency_s=? WHERE id=?",(float(seconds),event_id))

    def events(self, filters=None, limit=100, offset=0):
        filters = filters or {}
        clauses, values = [], []
        for key in ("camera_id","zone_id","event_type","review_status","run_id"):
            if filters.get(key):
                clauses.append(f"{key}=?")
                values.append(filters[key])
        for key, op in (("from",">="),("to","<=")):
            if filters.get(key):
                moment = datetime.fromisoformat(filters[key].replace("Z","+00:00"))
                if moment.tzinfo is None:
                    raise ValueError("Time filters must include a timezone")
                clauses.append(f"confirmed_at {op} ?")
                values.append(moment.astimezone(timezone.utc).isoformat(timespec="milliseconds"))
        where = " WHERE "+" AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM events"+where,values).fetchone()[0]
            rows = db.execute("SELECT * FROM events"+where+" ORDER BY confirmed_at DESC,id DESC LIMIT ? OFFSET ?",
                              [*values,min(max(int(limit),1),1000),max(0,int(offset))]).fetchall()
            return {"items":[dict(r) for r in rows],"total":count}

    def notifications(self, after=0):
        with self.connect() as db:
            cursor = db.execute("SELECT COALESCE(MAX(cursor),0) FROM notifications").fetchone()[0]
            rows = db.execute("SELECT n.cursor,e.* FROM notifications n JOIN events e ON e.id=n.event_id WHERE n.cursor>? ORDER BY n.cursor LIMIT 100",(int(after),)).fetchall()
            # Advance only through delivered rows, never skip a backlog larger than one page.
            return {"items":[dict(r) for r in rows],"cursor":rows[-1]["cursor"] if rows else max(int(after),cursor),"latest":cursor}

    def review(self, event_id, status, note=""):
        if status not in ("new","acknowledged","confirmed","false_positive"):
            raise ValueError("Unknown review status")
        with self.connect() as db:
            cursor = db.execute("UPDATE events SET review_status=?,review_note=?,reviewed_at=? WHERE id=?",
                                (status,str(note)[:2000],utc_now(),event_id))
            return bool(cursor.rowcount)

    def displayed(self,event_id):
        now=datetime.now(timezone.utc)
        with self.connect() as db:
            row=db.execute("SELECT created_at FROM events WHERE id=?",(event_id,)).fetchone()
            if not row:
                return False
            latency=max(0,(now-datetime.fromisoformat(row["created_at"])).total_seconds())
            db.execute("INSERT OR IGNORE INTO deliveries(event_id,displayed_at,create_to_display_ack_s) VALUES (?,?,?)",
                       (event_id,now.isoformat(timespec="milliseconds"),latency))
            return True

    def delivery_metrics(self,run_id=None):
        with self.connect() as db:
            query="SELECT d.*,e.confirmed_at,r.config_json FROM deliveries d JOIN events e ON e.id=d.event_id LEFT JOIN runs r ON r.id=e.run_id"
            rows=db.execute(query+" WHERE e.backend_kind!='demo_simulated'"+(" AND e.run_id=?" if run_id else ""),(run_id,) if run_id else ()).fetchall()
            import numpy as np
            values=[r["create_to_display_ack_s"] for r in rows]
            confirmation_values=[]
            for row in rows:
                mode=json.loads(row["config_json"] or '{}').get('camera',{}).get('mode')
                if mode=='realtime':
                    confirmation_values.append(max(0,(datetime.fromisoformat(row["displayed_at"])-datetime.fromisoformat(row["confirmed_at"])).total_seconds()))
            return {"n":len(values),"p50_s":float(np.percentile(values,50)) if values else None,
                    "p95_s":float(np.percentile(values,95)) if values else None,
                    "source_confirmation_to_display_ack":{"n":len(confirmation_values),"p50_s":float(np.percentile(confirmation_values,50)) if confirmation_values else None,
                        "p95_s":float(np.percentile(confirmation_values,95)) if confirmation_values else None,
                        "scope":"realtime files only; source confirmation PTS anchored to run wall time through inference, journal, UI polling and acknowledgement"},
                    "scope":"event creation to server receipt of browser display acknowledgement; includes polling and return trip",
                    "items":[{k:r[k] for k in ("event_id","displayed_at","create_to_display_ack_s")} for r in rows]}

    def diagnostic(self, code, message, run_id=None, camera_id=None, severity="error"):
        with self.connect() as db:
            db.execute("INSERT INTO diagnostics(at,run_id,camera_id,severity,code,message) VALUES (?,?,?,?,?,?)",
                       (utc_now(),run_id,camera_id,severity,code,str(message)[:1000]))

    def diagnostics(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM diagnostics ORDER BY id DESC LIMIT 100")]

    def start_run(self, run_id, camera_id, backend, config):
        with self.connect() as db:
            db.execute("INSERT INTO runs(id,started_at,camera_id,backend,status,config_json) VALUES (?,?,?,?,?,?)",
                       (run_id,utc_now(),camera_id,backend,"running",json.dumps(config,ensure_ascii=False)))

    def finish_run(self, run_id, status, metrics):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,ended_at=?,metrics_json=? WHERE id=?",
                       (status,utc_now(),json.dumps(metrics),run_id))

    def runs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 50")]
