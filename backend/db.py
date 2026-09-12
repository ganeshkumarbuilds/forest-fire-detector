"""Postgres layer for Forest Fire Detector.

Your own account: set DATABASE_URL in backend/.env, e.g.
  DATABASE_URL=postgresql://myuser:mypassword@localhost:5432/forest_fire

All functions NEVER raise — they return None/False on failure so the
app safely falls back to the local JSON files. This guarantees zero
errors when Postgres is not configured or unreachable.
"""
import json
import os

try:
    import psycopg2
    import psycopg2.extras
    _DRIVER_OK = True
except ImportError:
    psycopg2 = None
    _DRIVER_OK = False


def _db_url():
    url = (os.environ.get("DATABASE_URL") or "").strip()
    # Heroku-style postgres:// -> psycopg2 needs postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url or None


def db_available():
    return bool(_db_url() and _DRIVER_OK)


def _connect():
    url = _db_url()
    if not url or not _DRIVER_OK:
        return None
    try:
        conn = psycopg2.connect(url, connect_timeout=5)
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"WARNING: Postgres connect failed: {e}")
        return None


def init_db():
    """Create tables if missing. Never raises. Returns True on success."""
    conn = _connect()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL DEFAULT '',
                    fire_detected BOOLEAN NOT NULL DEFAULT FALSE,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alert_config (
                    id INT PRIMARY KEY CHECK (id = 1),
                    email TEXT NOT NULL DEFAULT ''
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hotspots_cache (
                    id INT PRIMARY KEY CHECK (id = 1),
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        return True
    except Exception as e:
        print(f"WARNING: Postgres init failed: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---- Detections ----

def db_read_log(limit=500):
    """Return list of dicts newest-last (like JSON log). None on failure."""
    conn = _connect()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT filename, fire_detected, confidence, timestamp "
                "FROM detections ORDER BY id ASC LIMIT %s;",
                (limit,),
            )
            rows = cur.fetchall() or []
            out = []
            for r in rows:
                ts = r.get("timestamp")
                try:
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                except Exception:
                    ts_str = str(ts)
                out.append({
                    "filename": r.get("filename") or "",
                    "fire_detected": bool(r.get("fire_detected")),
                    "confidence": float(r.get("confidence") or 0),
                    "timestamp": ts_str,
                })
            return out
    except Exception as e:
        print(f"WARNING: Postgres read log failed: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def db_append_log(entry):
    """Insert one detection. Returns True on success, False otherwise."""
    conn = _connect()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO detections (filename, fire_detected, confidence, timestamp) "
                "VALUES (%s, %s, %s, %s);",
                (
                    str(entry.get("filename") or ""),
                    bool(entry.get("fire_detected")),
                    float(entry.get("confidence") or 0),
                    str(entry.get("timestamp")),
                ),
            )
        return True
    except Exception as e:
        print(f"WARNING: Postgres append log failed: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---- Alert email (single row id=1, you can UPDATE it manually) ----

def db_read_alert_email():
    conn = _connect()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM alert_config WHERE id = 1;")
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0]).strip() or None
            return None
    except Exception as e:
        print(f"WARNING: Postgres read alert failed: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def db_write_alert_email(email):
    conn = _connect()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO alert_config (id, email) VALUES (1, %s) "
                "ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email;",
                (email,),
            )
        return True
    except Exception as e:
        print(f"WARNING: Postgres write alert failed: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---- Hotspots cache (single row id=1) ----

def db_read_hotspots():
    conn = _connect()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM hotspots_cache WHERE id = 1;")
            row = cur.fetchone()
            if not row or row[0] is None:
                return None
            payload = row[0]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    return None
            return dict(payload) if isinstance(payload, dict) else None
    except Exception as e:
        print(f"WARNING: Postgres read hotspots failed: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def db_write_hotspots(payload):
    conn = _connect()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO hotspots_cache (id, payload, updated_at) "
                "VALUES (1, %s, NOW()) "
                "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW();",
                (json.dumps(payload),),
            )
        return True
    except Exception as e:
        print(f"WARNING: Postgres write hotspots failed: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
