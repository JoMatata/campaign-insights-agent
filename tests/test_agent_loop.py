import json

from src.agent.llm import AnthropicLLM, LLMResponse, ScriptedLLM, ToolCall
from src.agent.loop import run_agent
from src.demo import demo_llm

GOOD_FINAL = {"answer": "18 campaigns.", "key_numbers": [{"label": "campaigns", "value": 18}],
              "sql_used": ["SELECT count(*) FROM campaigns"], "confidence": "high"}


def test_sql_error_is_fed_back_and_model_recovers(ctx, tmp_path):
    llm = ScriptedLLM([
        [("run_sql", {"query": "SELECT count(*) FROM campaign"})],  # typo in table name
        [("run_sql", {"query": "SELECT count(*) FROM campaigns"})],
        [("final_answer", GOOD_FINAL)],
    ])
    result = run_agent("How many campaigns?", llm, ctx, trace_dir=tmp_path)
    assert result.stopped == "final_answer" and result.final.key_numbers[0].value == 18

    tool_msg = llm.seen[1][-1]  # what the model saw before its second turn
    assert tool_msg["role"] == "tool_results" and tool_msg["results"][0]["is_error"]
    assert "campaign" in json.loads(tool_msg["results"][0]["content"])["error"]

    trace = json.loads(next(tmp_path.glob("trace_*.json")).read_text())
    assert [s["tool_calls"][0]["ok"] for s in trace["steps"]] == [False, True, True]


def test_invalid_final_answer_is_rejected_then_retried(ctx):
    llm = ScriptedLLM([
        [("final_answer", {"answer": "x", "confidence": "very sure"})],
        [("final_answer", GOOD_FINAL)],
    ])
    result = run_agent("q", llm, ctx)
    assert result.final.confidence == "high" and len(result.steps) == 2
    assert "did not validate" in llm.seen[1][-1]["results"][0]["content"]


def test_step_cap(ctx):
    llm = ScriptedLLM([[("get_schema", {})]] * 10)
    result = run_agent("q", llm, ctx, max_steps=3)
    assert result.stopped == "max_steps" and result.final.confidence == "low"
    assert len(result.steps) == 3


def test_plain_text_reply_gets_a_nudge(ctx):
    llm = ScriptedLLM([lambda m: LLMResponse(text="I think 18."), [("final_answer", GOOD_FINAL)]])
    result = run_agent("q", llm, ctx)
    assert result.final is not None and "final_answer" in llm.seen[1][-1]["content"]


def test_scripted_demo_runs_end_to_end(ctx):
    result = run_agent("demo", demo_llm(), ctx)
    assert result.stopped == "final_answer"
    assert any(not c["ok"] for c in result.tool_calls)  # the deliberate SQL error
    labels = [k.label for k in result.final.key_numbers]
    assert "C11 CVR" in labels


def test_anthropic_message_conversion():
    convert = AnthropicLLM._convert
    call = ToolCall("t1", "run_sql", {"query": "SELECT 1"})
    assert convert({"role": "assistant", "text": "", "tool_calls": [call]})["content"] == [
        {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"query": "SELECT 1"}}]
    out = convert({"role": "tool_results", "results": [{"id": "t1", "content": "{}", "is_error": False}]})
    assert out["role"] == "user" and out["content"][0]["tool_use_id"] == "t1"
