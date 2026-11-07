import os
from datetime import datetime
from typing import List, Dict, Tuple
import config

USE_POSTGRES = bool(os.getenv("DATABASE_URL", "").strip())

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
else:
    import sqlite3


class Database:
    def __init__(self):
        if USE_POSTGRES:
            url = os.getenv("DATABASE_URL")
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            self.pool = SimpleConnectionPool(1, 5, dsn=url)
            self._init_pg()
        else:
            self.db_path = getattr(config, "DB_PATH", "data/predictor.db")
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._init_sqlite()

    def _conn(self):
        if USE_POSTGRES:
            return self.pool.getconn()
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _put(self, conn):
        if USE_POSTGRES:
            self.pool.putconn(conn)
        else:
            conn.close()

    def _init_sqlite(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS price_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL, price REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pred_time INTEGER, target_time INTEGER, current_price REAL,
                    predicted_direction TEXT, probability REAL, features_json TEXT,
                    actual_price REAL, actual_direction TEXT, is_correct INTEGER,
                    model_version TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER,
                    accuracy_last_20 REAL, accuracy_last_50 REAL,
                    total_predictions INTEGER, total_correct INTEGER, notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS idx_ticks ON price_ticks(timestamp);
                CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(pred_time);
                CREATE TABLE IF NOT EXISTS service_state (
                    state_key TEXT PRIMARY KEY, state_value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
            """)

    def _init_pg(self):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS price_ticks (
                        id SERIAL PRIMARY KEY, timestamp BIGINT, price DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY, pred_time BIGINT, target_time BIGINT,
                        current_price DOUBLE PRECISION, predicted_direction TEXT,
                        probability DOUBLE PRECISION, features_json TEXT,
                        actual_price DOUBLE PRECISION, actual_direction TEXT,
                        is_correct INTEGER, model_version TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id SERIAL PRIMARY KEY, timestamp BIGINT,
                        accuracy_last_20 DOUBLE PRECISION, accuracy_last_50 DOUBLE PRECISION,
                        total_predictions INTEGER, total_correct INTEGER, notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                    CREATE TABLE IF NOT EXISTS service_state (
                        state_key TEXT PRIMARY KEY, state_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                    CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(pred_time);
                """)
                conn.commit()
        finally:
            self._put(conn)

    def insert_tick(self, ts, price):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO price_ticks (timestamp, price) VALUES (%s,%s)", (ts, price))
                conn.commit()
            else:
                conn.execute("INSERT INTO price_ticks (timestamp, price) VALUES (?,?)", (ts, price))
                conn.commit()
        finally:
            self._put(conn)

    def get_recent_ticks(self, limit=200):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT timestamp, price FROM price_ticks ORDER BY timestamp DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
            else:
                rows = conn.execute("SELECT timestamp, price FROM price_ticks ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [(r["timestamp"], r["price"]) for r in reversed(rows)]
        finally:
            self._put(conn)

    def get_ticks_between(self, start, end):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT timestamp, price FROM price_ticks WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp", (start, end))
                    rows = cur.fetchall()
            else:
                rows = conn.execute("SELECT timestamp, price FROM price_ticks WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp", (start, end)).fetchall()
            return [(r["timestamp"], r["price"]) for r in rows]
        finally:
            self._put(conn)

    def insert_prediction(self, pred_time, target_time, current_price, direction, probability, features_json=None, model_version="v1"):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO predictions
                        (pred_time,target_time,current_price,predicted_direction,probability,features_json,model_version)
                        VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (pred_time, target_time, current_price, direction, probability, features_json, model_version))
                    pid = cur.fetchone()[0]
                conn.commit()
                return pid
            else:
                cur = conn.execute("""INSERT INTO predictions
                    (pred_time,target_time,current_price,predicted_direction,probability,features_json,model_version)
                    VALUES (?,?,?,?,?,?,?)""",
                    (pred_time, target_time, current_price, direction, probability, features_json, model_version))
                conn.commit()
                return cur.lastrowid
        finally:
            self._put(conn)

    def update_prediction_result(self, pred_id, actual_price, actual_direction, is_correct):
        conn = self._conn()
        try:
            q = "UPDATE predictions SET actual_price=%s, actual_direction=%s, is_correct=%s WHERE id=%s" if USE_POSTGRES else \
                "UPDATE predictions SET actual_price=?, actual_direction=?, is_correct=? WHERE id=?"
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute(q, (actual_price, actual_direction, is_correct, pred_id))
                conn.commit()
            else:
                conn.execute(q, (actual_price, actual_direction, is_correct, pred_id))
                conn.commit()
        finally:
            self._put(conn)

    def get_pending_predictions(self, before_ms):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM predictions WHERE actual_price IS NULL AND target_time <= %s ORDER BY target_time", (before_ms,))
                    return [dict(r) for r in cur.fetchall()]
            rows = conn.execute("SELECT * FROM predictions WHERE actual_price IS NULL AND target_time <= ? ORDER BY target_time", (before_ms,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            self._put(conn)

    def get_recent_accuracy(self, n=20):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT is_correct FROM predictions WHERE is_correct IS NOT NULL ORDER BY pred_time DESC LIMIT %s", (n,))
                    rows = cur.fetchall()
            else:
                rows = conn.execute("SELECT is_correct FROM predictions WHERE is_correct IS NOT NULL ORDER BY pred_time DESC LIMIT ?", (n,)).fetchall()
            if not rows:
                return 0.0, 0, 0
            correct = sum(1 for r in rows if r["is_correct"] == 1)
            return correct / len(rows), correct, len(rows)
        finally:
            self._put(conn)

    def get_total_stats(self):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""SELECT COUNT(*) as total,
                        COALESCE(SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END),0) as correct,
                        COALESCE(SUM(CASE WHEN is_correct=0 THEN 1 ELSE 0 END),0) as wrong
                        FROM predictions WHERE is_correct IS NOT NULL""")
                    return dict(cur.fetchone())
            row = conn.execute("""SELECT COUNT(*) as total,
                SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN is_correct=0 THEN 1 ELSE 0 END) as wrong
                FROM predictions WHERE is_correct IS NOT NULL""").fetchone()
            return dict(row) if row else {"total": 0, "correct": 0, "wrong": 0}
        finally:
            self._put(conn)

    def log_metrics(self, acc20, acc50, total, correct, notes=""):
        pass  # optional


    def set_state(self, key, value):
        value = str(value) if value is not None else ""
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO service_state(state_key,state_value,updated_at)
                        VALUES (%s,%s,CURRENT_TIMESTAMP)
                        ON CONFLICT(state_key) DO UPDATE SET state_value=EXCLUDED.state_value,
                        updated_at=CURRENT_TIMESTAMP""", (key, value))
                conn.commit()
            else:
                conn.execute("""INSERT INTO service_state(state_key,state_value,updated_at)
                    VALUES (?,?,CURRENT_TIMESTAMP)
                    ON CONFLICT(state_key) DO UPDATE SET state_value=excluded.state_value,
                    updated_at=CURRENT_TIMESTAMP""", (key, value))
                conn.commit()
        finally:
            self._put(conn)

    def get_all_state(self):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT state_key,state_value FROM service_state")
                    rows = cur.fetchall()
            else:
                rows = conn.execute("SELECT state_key,state_value FROM service_state").fetchall()
            return {r["state_key"]: r["state_value"] for r in rows}
        finally:
            self._put(conn)

    def get_latest_tick(self):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT timestamp,price FROM price_ticks ORDER BY timestamp DESC LIMIT 1")
                    row = cur.fetchone()
            else:
                row = conn.execute("SELECT timestamp,price FROM price_ticks ORDER BY timestamp DESC LIMIT 1").fetchone()
            return dict(row) if row else None
        finally:
            self._put(conn)

    def get_latest_prediction(self):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id,pred_time,target_time,current_price,predicted_direction,probability,actual_price,actual_direction,is_correct FROM predictions ORDER BY pred_time DESC LIMIT 1")
                    row = cur.fetchone()
            else:
                row = conn.execute("SELECT id,pred_time,target_time,current_price,predicted_direction,probability,actual_price,actual_direction,is_correct FROM predictions ORDER BY pred_time DESC LIMIT 1").fetchone()
            return dict(row) if row else None
        finally:
            self._put(conn)

    def has_prediction_in_slot(self, slot_start_ms, slot_end_ms):
        conn = self._conn()
        try:
            if USE_POSTGRES:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM predictions WHERE pred_time >= %s AND pred_time < %s LIMIT 1", (slot_start_ms, slot_end_ms))
                    return cur.fetchone() is not None
            row = conn.execute("SELECT 1 FROM predictions WHERE pred_time >= ? AND pred_time < ? LIMIT 1", (slot_start_ms, slot_end_ms)).fetchone()
            return row is not None
        finally:
            self._put(conn)
