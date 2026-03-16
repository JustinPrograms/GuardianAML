#!/usr/bin/env python3
"""Migrate engineered CSV data to Transaction.db (SQLite).

Usage:
    python csv_to_db.py                          # default paths
    python csv_to_db.py <csv_path> <db_path>     # custom paths
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

GUARDIAN_ROOT = Path(__file__).resolve().parent.parent   # .../Guardian
DEFAULT_CSV = GUARDIAN_ROOT / "data" / "engineered_testing_mini.csv"
DEFAULT_DB = GUARDIAN_ROOT / "data" / "Transaction.db"

CHUNK_SIZE = 50_000


def _norm_col(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_").replace("__", "_")


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS FullData (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp            TEXT    NOT NULL,
            From_Bank            TEXT    NOT NULL,
            Account              TEXT    NOT NULL,
            To_Bank              TEXT    NOT NULL,
            To_Account           TEXT    NOT NULL,
            Amount_Received      REAL,
            Receiving_Currency   TEXT,
            Amount_Paid          REAL,
            Payment_Currency     TEXT,
            Payment_Format       TEXT,
            Is_Laundering        INTEGER,
            fx_spread            REAL,
            is_cross_cur         INTEGER,
            same_bank            INTEGER,
            self_transfer        INTEGER,
            hour                 INTEGER,
            dow                  INTEGER,
            is_weekend           INTEGER,
            acc_tx_count         REAL,
            acc_amt_paid_mean    REAL,
            acc_amt_recv_mean    REAL,
            acc_cross_ccy_rate   REAL,
            toacc_tx_count       REAL,
            toacc_amt_paid_mean  REAL,
            toacc_amt_recv_mean  REAL,
            toacc_cross_ccy_rate REAL,
            fb_tx_count          REAL,
            fb_amt_paid_mean     REAL,
            fb_amt_recv_mean     REAL,
            fb_cross_ccy_rate    REAL,
            tb_tx_count          REAL,
            tb_amt_paid_mean     REAL,
            tb_amt_recv_mean     REAL,
            tb_cross_ccy_rate    REAL
        );

        -- Deduplicated account list for fast random sampling
        CREATE TABLE IF NOT EXISTS accounts (
            from_bank TEXT NOT NULL,
            account   TEXT NOT NULL,
            PRIMARY KEY (from_bank, account)
        );

        CREATE INDEX IF NOT EXISTS idx_from ON FullData (From_Bank, Account);
        CREATE INDEX IF NOT EXISTS idx_to   ON FullData (To_Bank,   To_Account);
        CREATE INDEX IF NOT EXISTS idx_bank ON FullData (From_Bank);
    """)


def migrate(csv_path: Path, db_path: Path) -> None:
    print(f"Source CSV : {csv_path}")
    print(f"Target DB  : {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache

    build_schema(conn)

    # ── Populate FullData ──
    existing = conn.execute("SELECT COUNT(*) FROM FullData").fetchone()[0]
    if existing > 0:
        print(f"FullData already has {existing:,} rows — skipping insert.")
    else:
        print("Inserting rows from CSV...")
        total      = 0
        drop_cols: list[str] = []

        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            if total == 0:
                drop_cols = [
                    c for c in chunk.columns
                    if str(c).strip() == "" or str(c).lower().startswith("unnamed")
                ]

            if drop_cols:
                chunk = chunk.drop(columns=drop_cols, errors="ignore")

            chunk = chunk.rename(columns={c: _norm_col(c) for c in chunk.columns})

            # Remove the old numeric index column if it survived normalization
            chunk = chunk.drop(columns=[c for c in chunk.columns if c.isdigit()], errors="ignore")

            chunk.to_sql(
                "FullData", conn,
                if_exists="append",
                index=False,
            )
            total += len(chunk)
            print(f"  {total:,} rows inserted...", end="\r", flush=True)

        conn.commit()
        print(f"\nInserted {total:,} rows into FullData.")

    # ── Populate accounts ──
    acc_count = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    if acc_count == 0:
        print("Building accounts table from FullData...")
        conn.execute("""
            INSERT OR IGNORE INTO accounts (from_bank, account)
            SELECT DISTINCT From_Bank, Account FROM FullData
        """)
        conn.commit()
        acc_count = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        print(f"Inserted {acc_count:,} unique (bank, account) pairs.")
    else:
        print(f"accounts table already has {acc_count:,} rows — skipping.")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    db_path  = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DB
    migrate(csv_path, db_path)
