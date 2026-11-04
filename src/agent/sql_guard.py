"""Read-only SQL execution for the agent.

The model writes the SQL, so I treat it as untrusted input. There are three layers:
a keyword check here, a read-only DuckDB connection with external access off, and
an outer LIMIT plus a timeout so one bad query cannot hang the loop.
"""
import re
import threading
import time

import duckdb

MAX_ROWS = 200
TIMEOUT_S = 10.0

BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|pragma|"
    r"install|load|call|set|reset|checkpoint|vacuum|grant|truncate)\b"
    r"|\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|parquet_scan)\s*\(",
    re.IGNORECASE,
)


class SQLRejected(ValueError):
    pass


def _strip(sql):
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql.strip().rstrip(";").strip()


def check_sql(sql):
    """Return cleaned SQL or raise SQLRejected with a message the model can act on."""
    clean = _strip(sql)
    if not clean:
        raise SQLRejected("Empty query.")
    if ";" in clean:
        raise SQLRejected("Only one statement is allowed. Remove the extra ';'.")
    first = clean.split(None, 1)[0].lower()
    if first not in ("select", "with"):
        raise SQLRejected(f"Only SELECT or WITH queries are allowed, got '{first.upper()}'.")
    # quoted strings can legitimately contain words like 'update', so ignore them
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", clean)
    m = BLOCKED.search(no_strings)
    if m:
        raise SQLRejected(f"Blocked keyword or function: '{m.group(0).strip()}'.")
    return clean


def connect(db_path):
    return duckdb.connect(str(db_path), read_only=True,
                          config={"enable_external_access": False})


def run_query(con, sql, max_rows=MAX_ROWS, timeout_s=TIMEOUT_S):
    """Run a guarded query. Returns a dict, never raises, so errors go back to the model."""
    try:
        clean = check_sql(sql)
    except SQLRejected as e:
        return {"ok": False, "error": f"Rejected: {e}"}

    wrapped = f"SELECT * FROM ({clean}) AS q LIMIT {max_rows + 1}"
    timer = threading.Timer(timeout_s, con.interrupt)
    start = time.perf_counter()
    timer.start()
    try:
        cur = con.execute(wrapped)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except duckdb.InterruptException:
        return {"ok": False, "error": f"Query timed out after {timeout_s}s. Aggregate more or filter."}
    except duckdb.Error as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        timer.cancel()

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {
        "ok": True,
        "columns": cols,
        "rows": [[_jsonable(v) for v in r] for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def _jsonable(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)  # dates, decimals, int128
