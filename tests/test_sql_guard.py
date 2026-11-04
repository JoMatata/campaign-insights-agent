import duckdb
import pytest

from src.agent.sql_guard import SQLRejected, check_sql, connect, run_query


@pytest.mark.parametrize("sql", [
    "DROP TABLE campaigns",
    "INSERT INTO campaigns VALUES (1)",
    "SELECT 1; DROP TABLE campaigns",
    "PRAGMA database_list",
    "ATTACH 'other.db'",
    "SELECT * FROM read_csv('/etc/passwd')",
    "WITH x AS (SELECT 1) DELETE FROM campaigns",
    "",
    "-- just a comment",
])
def test_rejects_non_select(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql)


def test_allows_select_with_and_trailing_semicolon():
    assert check_sql("SELECT 1;") == "SELECT 1"
    assert check_sql("WITH a AS (SELECT 1 AS x) SELECT x FROM a").startswith("WITH")


def test_keywords_inside_strings_are_fine():
    check_sql("SELECT * FROM feedback WHERE text ILIKE '%update%'")


def test_rejected_query_returns_error_not_exception(ctx):
    res = run_query(ctx.con, "DELETE FROM campaigns")
    assert res["ok"] is False and "Rejected" in res["error"]


def test_bad_column_error_goes_back_to_model(ctx):
    res = run_query(ctx.con, "SELECT nope FROM campaigns")
    assert res["ok"] is False and "nope" in res["error"]


def test_limit_is_enforced(ctx):
    res = run_query(ctx.con, "SELECT * FROM daily_performance", max_rows=7)
    assert res["row_count"] == 7 and res["truncated"] is True


def test_timeout_interrupts_long_query(ctx):
    res = run_query(ctx.con, "SELECT count(*) FROM range(100000000) a, range(100000) b", timeout_s=0.3)
    assert res["ok"] is False and "timed out" in res["error"]


def test_connection_is_read_only(db_path):
    con = connect(db_path)
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE x (a INT)")
