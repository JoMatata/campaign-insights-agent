import pytest

from src.agent.stats import campaign_metrics, two_proportion_ztest
from src.agent.tools import call_tool, compare_campaigns, tool_specs


def test_ztest_matches_hand_calculation():
    # p1=0.20, p2=0.15, pooled=0.175, se=sqrt(0.175*0.825*0.002)=0.016993
    r = two_proportion_ztest(200, 1000, 150, 1000)
    assert r["z"] == pytest.approx(2.9424, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.00326, abs=1e-4)
    assert r["ci_low"] < 0.05 < r["ci_high"]
    assert r["significant"]


def test_ztest_no_difference():
    r = two_proportion_ztest(50, 1000, 50, 1000)
    assert r["z"] == 0 and r["p_value"] == pytest.approx(1.0)


def test_ztest_rejects_empty_group():
    with pytest.raises(ValueError):
        two_proportion_ztest(0, 0, 1, 10)


def test_metrics_handle_zero_denominators():
    m = campaign_metrics(1000, 20, 0, 500.0, 0.0)
    assert m["ctr"] == 0.02 and m["cvr"] == 0 and m["cpa"] is None and m["roas"] == 0


def test_compare_campaigns_agrees_with_sql(ctx):
    r = compare_campaigns(ctx, "C01", "C18")
    clicks, imps = ctx.con.execute("SELECT sum(clicks), sum(impressions) FROM daily_performance "
                                   "WHERE campaign_id='C01'").fetchone()
    assert r["a"]["ctr"] == pytest.approx(clicks / imps)
    assert r["cvr_test"]["significant"]  # search converts far better than TikTok by design


def test_compare_unknown_campaign_returns_error(ctx):
    assert compare_campaigns(ctx, "C01", "C99")["ok"] is False


def test_call_tool_errors_are_returned(ctx):
    assert call_tool(ctx, "nope", {})["ok"] is False
    assert "Bad arguments" in call_tool(ctx, "run_sql", {"sql": "SELECT 1"})["error"]


def test_feedback_tools(ctx):
    hits = call_tool(ctx, "search_feedback", {"query": "mpesa payment failed", "k": 5})
    assert len(hits["results"]) == 5
    s = call_tool(ctx, "summarise_feedback_theme", {"theme": "late delivery", "k": 30,
                                                    "county": "Mombasa"})
    assert s["ok"] and s["by_county"] == {"Mombasa": 30} and len(s["representative_quotes"]) == 5


def test_specs_include_final_answer():
    names = [t["name"] for t in tool_specs()]
    assert names[-1] == "final_answer" and "run_sql" in names
