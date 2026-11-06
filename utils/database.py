"""
Database layer - PostgreSQL (via DATABASE_URL)
Compatible with Render Postgres.
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import config

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[Database] psycopg2 not installed")


class Database:
    def __init__(self):
        self.database_url = config.DATABASE_URL
        self.pool = None

        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not set. "
                "Please create a PostgreSQL database on Render and link it to this service."
            )

        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2-binary is required for PostgreSQL")

        # Render sometimes gives postgres:// instead of postgresql://
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)

        self._init_pool()
        self._init_db()

    def _init_pool(self):
        self.pool = SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=self.database_url
        )

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def _init_db(self):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS price_ticks (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        pred_time BIGINT NOT NULL,
                        target_time BIGINT NOT NULL,
                        current_price DOUBLE PRECISION NOT NULL,
                        predicted_direction TEXT NOT NULL,
                        probability DOUBLE PRECISION NOT NULL,
                        features_json TEXT,
                        actual_price DOUBLE PRECISION,
                        actual_direction TEXT,
                        is_correct INTEGER,
                        model_version TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        accuracy_last_20 DOUBLE PRECISION,
                        accuracy_last_50 DOUBLE PRECISION,
                        total_predictions INTEGER,
                        total_correct INTEGER,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_ticks_ts ON price_ticks(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(pred_time);
                    CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_time);
                """)
                conn.commit()
        finally:
            self._put_conn(conn)

    def insert_tick(self, timestamp_ms: int, price: float):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO price_ticks (timestamp, price) VALUES (%s, %s)",
                    (timestamp_ms, price)
                )
                conn.commit()
        finally:
            self._put_conn(conn)

    def get_recent_ticks(self, limit: int = 200) -> List[Tuple[int, float]]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT timestamp, price FROM price_ticks ORDER BY timestamp DESC LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
                return [(r["timestamp"], r["price"]) for r in reversed(rows)]
        finally:
            self._put_conn(conn)

    def get_ticks_between(self, start_ms: int, end_ms: int) -> List[Tuple[int, float]]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT timestamp, price FROM price_ticks 
                       WHERE timestamp BETWEEN %s AND %s 
                       ORDER BY timestamp""",
                    (start_ms, end_ms)
                )
                rows = cur.fetchall()
                return [(r["timestamp"], r["price"]) for r in rows]
        finally:
            self._put_conn(conn)

    def insert_prediction(self, pred_time: int, target_time: int, current_price: float,
                          direction: str, probability: float, features_json: str = None,
                          model_version: str = "v1") -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO predictions 
                       (pred_time, target_time, current_price, predicted_direction, 
                        probability, features_json, model_version)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (pred_time, target_time, current_price, direction, probability,
                     features_json, model_version)
                )
                pred_id = cur.fetchone()[0]
                conn.commit()
                return pred_id
        finally:
            self._put_conn(conn)

    def update_prediction_result(self, pred_id: int, actual_price: float,
                                 actual_direction: str, is_correct: int):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE predictions 
                       SET actual_price = %s, actual_direction = %s, is_correct = %s
                       WHERE id = %s""",
                    (actual_price, actual_direction, is_correct, pred_id)
                )
                conn.commit()
        finally:
            self._put_conn(conn)

    def get_pending_predictions(self, before_ms: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT * FROM predictions 
                       WHERE actual_price IS NULL AND target_time <= %s
                       ORDER BY target_time""",
                    (before_ms,)
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def get_recent_accuracy(self, n: int = 20) -> Tuple[float, int, int]:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT is_correct FROM predictions 
                       WHERE is_correct IS NOT NULL 
                       ORDER BY pred_time DESC LIMIT %s""",
                    (n,)
                )
                rows = cur.fetchall()
                if not rows:
                    return 0.0, 0, 0
                correct = sum(1 for r in rows if r["is_correct"] == 1)
                total = len(rows)
                return correct / total, correct, total
        finally:
            self._put_conn(conn)

    def get_total_stats(self) -> Dict:
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT 
                          COUNT(*) as total,
                          COALESCE(SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END), 0) as correct,
                          COALESCE(SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END), 0) as wrong
                       FROM predictions WHERE is_correct IS NOT NULL"""
                )
                row = cur.fetchone()
                return dict(row) if row else {"total": 0, "correct": 0, "wrong": 0}
        finally:
            self._put_conn(conn)

    def log_metrics(self, acc20: float, acc50: float, total: int, correct: int, notes: str = ""):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO model_metrics 
                       (timestamp, accuracy_last_20, accuracy_last_50, total_predictions, total_correct, notes)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (int(datetime.utcnow().timestamp() * 1000), acc20, acc50, total, correct, notes)
                )
                conn.commit()
        finally:
            self._put_conn(conn)
