"""Score one agent run against a golden question."""
import re

from src.agent.sql_guard import run_query

REL_TOL = 0.01  # 1%; the model is told to round sensibly, not to quote 6 decimals


def expected_answer(con, golden):
    res = run_query(con, golden["sql"])
    if not res["ok"]:
        raise RuntimeError(f"Golden SQL failed for {golden['id']}: {res['error']}")
    row = res["rows"][0]
    numbers = [float(v) for v in row if isinstance(v, (int, float))]
    texts = [str(v) for v in row if isinstance(v, str)]
    return {"numbers": numbers, "texts": texts}


def number_found(expected, got, percent=False, rel_tol=REL_TOL):
    # rates may come back as a fraction or a percentage; allow rounding to 0.1 percentage points
    candidates = [(expected, 0.0005), (expected * 100, 0.05)] if percent else [(expected, 0.0)]
    for e, rounding in candidates:
        tol = max(abs(e) * rel_tol, rounding)
        if any(abs(g - e) <= tol for g in got):
            return True
    return False


def _norm(s):
    return re.sub(r"[\s_\-]+", "", s.lower())


def text_found(expected_texts, answer, accept=(), any_text=False):
    if not expected_texts:
        return None
    ans = _norm(answer)
    hits = [_norm(t) in ans for t in expected_texts]
    alt = any(_norm(a) in ans for a in accept)
    return (any(hits) if any_text else all(hits)) or alt


def score_run(golden, expected, result, con):
    final = result.final
    got = [k.value for k in final.key_numbers] if final else []
    answer_text = (final.answer + " " + " ".join(k.label for k in final.key_numbers)) if final else ""

    num_hits = [number_found(e, got, golden.get("percent", False)) for e in expected["numbers"]]
    numeric = sum(num_hits) / len(num_hits) if num_hits else None
    text_ok = text_found(expected["texts"], answer_text, golden.get("accept", ()), golden.get("any_text", False))

    sql_list = final.sql_used if final else []
    sql_ok = all(run_query(con, q, max_rows=5)["ok"] for q in sql_list) if sql_list else None

    parts = [p for p in (numeric == 1.0 if numeric is not None else None, text_ok) if p is not None]
    return {"id": golden["id"], "correct": bool(parts) and all(parts), "numeric_score": numeric,
            "text_ok": text_ok, "sql_executes": sql_ok, "tool_calls": len(result.tool_calls),
            "latency_ms": result.total_ms, "stopped": result.stopped}


def summarise(rows):
    n = len(rows)
    sql_rows = [r for r in rows if r["sql_executes"] is not None]
    lat = sorted(r["latency_ms"] for r in rows)
    return {"n": n,
            "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "sql_executes_rate": round(sum(r["sql_executes"] for r in sql_rows) / len(sql_rows), 3) if sql_rows else None,
            "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
            "median_latency_ms": lat[n // 2],
            "hit_step_cap": sum(r["stopped"] == "max_steps" for r in rows)}
