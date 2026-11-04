from eval.golden import GOLDEN
from eval.run_eval import replay_llm
from eval.scoring import expected_answer, number_found, score_run, summarise, text_found
from src.agent.llm import ScriptedLLM
from src.agent.loop import run_agent


def test_number_matching_handles_percent_and_tolerance():
    assert number_found(0.0477, [4.77], percent=True)
    assert number_found(0.0477, [4.8], percent=True)  # rounded to 0.1 pp
    assert not number_found(0.0477, [5.2], percent=True)
    assert number_found(1_000_000, [1_005_000])
    assert not number_found(1_000_000, [1_050_000])
    assert not number_found(0.0477, [4.77], percent=False)


def test_text_matching():
    assert text_found(["google_search"], "Google Search has the best ROAS")
    assert not text_found(["Nairobi"], "Mombasa")
    assert text_found(["C02", "Search Generic"], "It is C02", any_text=True)


def test_all_golden_sql_runs(ctx):
    assert len(GOLDEN) >= 25
    for g in GOLDEN:
        exp = expected_answer(ctx.con, g)
        assert exp["numbers"] or exp["texts"], g["id"]


def test_fake_replay_scores_full_marks(ctx):
    rows = []
    for g in GOLDEN:
        result = run_agent(g["question"], replay_llm(g), ctx)
        rows.append(score_run(g, expected_answer(ctx.con, g), result, ctx.con))
    s = summarise(rows)
    assert s["accuracy"] == 1.0 and s["sql_executes_rate"] == 1.0 and s["mean_tool_calls"] == 2


def test_wrong_answer_and_broken_sql_are_caught(ctx):
    g = GOLDEN[0]
    llm = ScriptedLLM([[("final_answer", {"answer": "about 5 KES", "confidence": "high",
                                          "key_numbers": [{"label": "spend", "value": 5}],
                                          "sql_used": ["SELECT nope FROM campaigns"]})]])
    row = score_run(g, expected_answer(ctx.con, g), run_agent(g["question"], llm, ctx), ctx.con)
    assert row["correct"] is False and row["sql_executes"] is False and row["tool_calls"] == 0
