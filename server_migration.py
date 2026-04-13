"""
VIGIL Server — Database Migration
Adds 4 new camera gait metric columns to the metrics and daily_summary tables.

Usage:
    python server_migration.py --db /path/to/vigil.db         # SQLite (Jetson)
    python server_migration.py --pg postgresql://user:pass@host/db  # Postgres
    python server_migration.py --db ~/passive-health-moniter/vigil.db --show-server-patch
"""

import argparse
import sqlite3
import sys

SQLITE_MIGRATIONS = [
    ("metrics",       "stride_variability",   "REAL", "0.0"),
    ("metrics",       "arm_swing_asymmetry",  "REAL", "0.0"),
    ("metrics",       "step_width",           "REAL", "0.0"),
    ("metrics",       "cadence_variability",  "REAL", "0.0"),
    ("daily_summary", "stride_variability",   "REAL", "0.0"),
    ("daily_summary", "arm_swing_asymmetry",  "REAL", "0.0"),
    ("daily_summary", "step_width",           "REAL", "0.0"),
    ("daily_summary", "cadence_variability",  "REAL", "0.0"),
]

POSTGRES_MIGRATIONS = [
    ("metrics",       "stride_variability",   "FLOAT", "0.0"),
    ("metrics",       "arm_swing_asymmetry",  "FLOAT", "0.0"),
    ("metrics",       "step_width",           "FLOAT", "0.0"),
    ("metrics",       "cadence_variability",  "FLOAT", "0.0"),
    ("daily_summary", "stride_variability",   "FLOAT", "0.0"),
    ("daily_summary", "arm_swing_asymmetry",  "FLOAT", "0.0"),
    ("daily_summary", "step_width",           "FLOAT", "0.0"),
    ("daily_summary", "cadence_variability",  "FLOAT", "0.0"),
]

SERVER_PATCH = """
# ── server.py /sync/camera patch ───────────────────────────────────────────
# Add to your existing POST /sync/camera handler:

data = request.get_json()
stride_variability  = data.get('strideVariability', 0.0)
arm_swing_asymmetry = data.get('armSwingAsymmetry', 0.0)
step_width          = data.get('stepWidth', 0.0)
cadence_variability = data.get('cadenceVariability', 0.0)

# Add to INSERT:
# INSERT INTO metrics (..., stride_variability, arm_swing_asymmetry, step_width, cadence_variability)
# VALUES (..., ?, ?, ?, ?)

# ── Daily aggregation patch ─────────────────────────────────────────────────
UPDATE_DAILY_SQL = '''
    UPDATE daily_summary SET
        stride_variability  = (SELECT AVG(stride_variability)  FROM metrics WHERE date = ?),
        arm_swing_asymmetry = (SELECT AVG(arm_swing_asymmetry) FROM metrics WHERE date = ?),
        step_width          = (SELECT AVG(step_width)          FROM metrics WHERE date = ?),
        cadence_variability = (SELECT AVG(cadence_variability) FROM metrics WHERE date = ?)
    WHERE date = ?
'''

# ── /history response formatter ─────────────────────────────────────────────
def format_daily(row: dict) -> dict:
    return {
        # ... existing fields ...
        "strideVariability":  row.get("stride_variability", 0),
        "armSwingAsymmetry":  row.get("arm_swing_asymmetry", 0),
        "stepWidth":          row.get("step_width", 0),
        "cadenceVariability": row.get("cadence_variability", 0),
    }
"""


def migrate_sqlite(db_path: str):
    print(f"Connecting to SQLite: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for table, col, col_type, default in SQLITE_MIGRATIONS:
        cur.execute(f"PRAGMA table_info({table})")
        existing = [row[1] for row in cur.fetchall()]
        if col in existing:
            print(f"  SKIP  {table}.{col} — already exists")
            continue
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}")
            print(f"  ADD   {table}.{col} ({col_type})")
        except Exception as e:
            print(f"  ERROR {table}.{col}: {e}", file=sys.stderr)
    conn.commit()
    conn.close()
    print("SQLite migration complete.")


def migrate_postgres(dsn: str):
    try:
        import psycopg2
    except ImportError:
        print("Run: pip install psycopg2-binary --break-system-packages")
        sys.exit(1)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    for table, col, col_type, default in POSTGRES_MIGRATIONS:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
            (table, col),
        )
        if cur.fetchone():
            print(f"  SKIP  {table}.{col} — already exists")
            continue
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}")
            print(f"  ADD   {table}.{col} ({col_type})")
        except Exception as e:
            print(f"  ERROR {table}.{col}: {e}", file=sys.stderr)
    cur.close()
    conn.close()
    print("Postgres migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL DB migration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--db", help="SQLite path")
    group.add_argument("--pg", help="Postgres DSN")
    parser.add_argument("--show-server-patch", action="store_true")
    args = parser.parse_args()
    if args.show_server_patch:
        print(SERVER_PATCH)
        sys.exit(0)
    if args.db:
        migrate_sqlite(args.db)
    elif args.pg:
        migrate_postgres(args.pg)
