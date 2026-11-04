"""Run the golden questions through the agent and score them.

  python -m eval.run_eval            # real model, needs ANTHROPIC_API_KEY
  python -m eval.run_eval --fake     # scripted replay, tests the plumbing only

The fake mode replays the reference SQL through the real loop and tools, so its
score says the loop, tools and scorer work together. It says nothing about model quality.
"""
import argparse
import json
import os

from eval.golden import GOLDEN
from eval.scoring import expected_answer, score_run, summarise
from src import config
from src.agent.llm import LLMResponse, ScriptedLLM, ToolCall, get_llm
from src.agent.loop import run_agent
from src.agent.tools import ToolContext


def replay_llm(golden):
    """Scripted LLM: get_schema, run the reference SQL, then answer from the result."""
    def finish(messages):
        res = json.loads(messages[-1]["results"][0]["content"])
        row = res["rows"][0]
        nums = [{"label": c, "value": v} for c, v in zip(res["columns"], row)
                if isinstance(v, (int, float))]
        texts = [str(v) for v in row if isinstance(v, str)]
        answer = "Answer: " + (", ".join(texts) if texts else ", ".join(str(n["value"]) for n in nums))
        return LLMResponse(tool_calls=[ToolCall("final", "final_answer", {
            "answer": answer, "key_numbers": nums, "sql_used": [golden["sql"]], "confidence": "high"})])

    return ScriptedLLM([[("get_schema", {})], [("run_sql", {"query": golden["sql"]})], finish])


def run(fake=False, limit=None, provider="anthropic", trace_dir=None):
    ctx = ToolContext()
    rows = []
    for g in GOLDEN[:limit]:
        llm = replay_llm(g) if fake else get_llm(provider)
        result = run_agent(g["question"], llm, ctx, trace_dir=trace_dir)
        rows.append(score_run(g, expected_answer(ctx.con, g), result, ctx.con))
    return {"mode": "fake-replay" if fake else provider, "summary": summarise(rows), "questions": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    args = ap.parse_args()
    key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not args.fake and not os.getenv(key):
        raise SystemExit(f"{key} is not set. Use --fake to test the plumbing without a model.")

    out = run(args.fake, args.limit, args.provider, trace_dir=config.TRACE_DIR / "eval")
    for r in out["questions"]:
        print(f"{r['id']}  correct={str(r['correct']):<5}  sql_ok={str(r['sql_executes']):<5}  "
              f"tools={r['tool_calls']}  {r['latency_ms']:.0f} ms")
    print(json.dumps(out["summary"], indent=2))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.REPORTS_DIR / f"eval_results_{out['mode']}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
