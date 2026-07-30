from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("artifacts") / "memory" / "agent_memory.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db() -> None:
    """Initializes SQLite database schema for persistent agent memory."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thesis_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_symbol TEXT NOT NULL,
                position_direction TEXT NOT NULL,
                target_horizon_hours INTEGER NOT NULL,
                reasoning TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                actual_outcome TEXT,
                accuracy_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def log_thesis(
    market_symbol: str,
    position_direction: str,
    target_horizon_hours: int,
    reasoning: str,
    confidence_score: float,
) -> int:
    """Logs a synthesized market thesis into agent memory."""
    init_memory_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO thesis_memory 
            (market_symbol, position_direction, target_horizon_hours, reasoning, confidence_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (market_symbol, position_direction, target_horizon_hours, reasoning, confidence_score),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)


def update_thesis_outcome(thesis_id: int, actual_outcome: str, accuracy_score: float) -> None:
    """Updates a historical thesis with ground-truth market outcome and accuracy score."""
    init_memory_db()
    with _get_connection() as conn:
        conn.execute(
            """
            UPDATE thesis_memory
            SET actual_outcome = ?, accuracy_score = ?
            WHERE id = ?
            """,
            (actual_outcome, accuracy_score, thesis_id),
        )
        conn.commit()


def get_historical_accuracy(market_symbol: str) -> float:
    """Retrieves average historical forecast accuracy score for a given market symbol."""
    init_memory_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT AVG(accuracy_score) as avg_acc
            FROM thesis_memory
            WHERE market_symbol = ? AND accuracy_score IS NOT NULL
            """,
            (market_symbol,),
        )
        row = cursor.fetchone()
        if row and row["avg_acc"] is not None:
            return float(row["avg_acc"])
    return 0.75  # Default baseline accuracy if no history exists
