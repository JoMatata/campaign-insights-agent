"""A scripted run used when there is no API key.

It plays the part of the model with a fixed sequence of tool calls, including one
broken SQL query, so you can see the loop, the error feedback and the trace without
a key. All numbers in the final answer are read from the real tool results.
"""
import json
import statistics

from src.agent.llm import LLMResponse, ScriptedLLM, ToolCall

DEMO_QUESTION = ("Did TikTok Madaraka Day (C11) underperform Meta Madaraka Day (C10)? "
                 "And what were customers complaining about at the end of May?")

BAD_SQL = ("SELECT campaign_id, sum(conversion) FROM daily_performance "
           "WHERE campaign_id IN ('C10', 'C11') GROUP BY 1")
FIXED_SQL = BAD_SQL.replace("sum(conversion)", "sum(conversions)")
DAILY_SQL = ("SELECT date, sum(conversions) AS conversions FROM daily_performance "
             "WHERE date BETWEEN '2026-05-25' AND '2026-06-01' GROUP BY 1 ORDER BY 1")


def _results(messages):
    """All tool results so far, newest last, decoded from JSON."""
    return [json.loads(r["content"]) for m in messages if m["role"] == "tool_results"
            for r in m["results"]]


def _finish(messages):
    res = _results(messages)
    cmp = next(r for r in res if "cvr_test" in r)
    theme = next(r for r in res if "representative_quotes" in r)
    daily = res[-1]["rows"]
    a, b, t = cmp["a"], cmp["b"], cmp["cvr_test"]
    low = min(daily, key=lambda r: r[1])
    median = statistics.median(r[1] for r in daily)
    verdict = "Yes" if t["significant"] and t["diff"] > 0 else "Not clearly"
    p = "p<0.001" if t["p_value"] < 0.001 else f"p={t['p_value']:.3f}"
    answer = (f"{verdict}. C11 (TikTok) converted {b['cvr']:.2%} of clicks against {a['cvr']:.2%} for "
              f"C10 (Meta), a gap of {t['diff'] * 100:.2f} pp ({p}). ROAS was {b['roas']:.2f} vs "
              f"{a['roas']:.2f}. Conversions fell to {low[1]} on {low[0]} against a median of "
              f"{median:.0f} per day that week. Feedback from 25 May to 1 June that is closest to "
              f"payment failure is about: {', '.join(theme['top_terms'][:6])}. "
              f"Example: \"{theme['representative_quotes'][0]}\"")
    final = {
        "answer": answer,
        "key_numbers": [
            {"label": "C10 CVR", "value": round(a["cvr"] * 100, 2), "unit": "%"},
            {"label": "C11 CVR", "value": round(b["cvr"] * 100, 2), "unit": "%"},
            {"label": "C10 ROAS", "value": round(a["roas"], 2), "unit": "x"},
            {"label": "C11 ROAS", "value": round(b["roas"], 2), "unit": "x"},
            {"label": f"Conversions on {low[0]}", "value": low[1], "unit": "count"},
        ],
        "sql_used": [FIXED_SQL, DAILY_SQL],
        "confidence": "medium",
        "caveats": ["Feedback counts come from similarity search, not exact labels.",
                    "Synthetic data: the outage is a planted pattern."],
    }
    return LLMResponse(text="Writing the answer.", tool_calls=[ToolCall("final", "final_answer", final)])


def demo_llm():
    return ScriptedLLM([
        [("get_schema", {})],
        [("run_sql", {"query": BAD_SQL})],  # wrong column name on purpose
        [("run_sql", {"query": FIXED_SQL}),
         ("compare_campaigns", {"campaign_a": "C10", "campaign_b": "C11"})],
        [("summarise_feedback_theme", {"theme": "M-Pesa payment failed, money deducted", "k": 40,
                                       "start_date": "2026-05-25", "end_date": "2026-06-01"}),
         ("run_sql", {"query": DAILY_SQL})],
        _finish,
    ])
