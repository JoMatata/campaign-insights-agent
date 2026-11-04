"""The agent loop: ask the model, run the tools it asks for, feed results back, repeat.

It stops when the model calls final_answer with output that passes Pydantic validation,
or when the step cap is hit. Every step is written to a JSON trace.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.agent.schemas import FinalAnswer
from src.agent.tools import call_tool, tool_specs

MAX_STEPS = 8
MAX_RESULT_CHARS = 6000

SYSTEM_PROMPT = """You are a marketing analyst for Tausi Foods, a Kenyan consumer brand that sells \
online and runs digital ad campaigns. Answer questions using the tools, not from memory.

- Call get_schema before writing SQL. The database is DuckDB. Money is in KES.
- Define metrics from totals: CTR = clicks/impressions, CVR = conversions/clicks, \
CPA = spend/conversions, ROAS = revenue/spend. Never average daily ratios.
- Use compare_campaigns when the user asks whether one campaign beats another.
- Use search_feedback or summarise_feedback_theme for questions about what customers say.
- If a query fails, read the error and fix the query.
- Finish by calling final_answer. Put each number you quote in key_numbers as a plain number \
(rates as percentages, e.g. 4.7 for 4.7%). Copy the SQL you relied on into sql_used.
- Be honest about uncertainty in confidence and caveats."""


@dataclass
class AgentResult:
    question: str
    final: FinalAnswer | None
    steps: list = field(default_factory=list)
    stopped: str = ""
    total_ms: float = 0.0
    trace_path: Path | None = None

    @property
    def tool_calls(self):
        return [c for s in self.steps for c in s["tool_calls"] if c["name"] != "final_answer"]

    def to_dict(self):
        return {"question": self.question, "stopped": self.stopped, "total_ms": self.total_ms,
                "final": self.final.model_dump() if self.final else None, "steps": self.steps}


def run_agent(question, llm, ctx, max_steps=MAX_STEPS, trace_dir=None):
    messages = [{"role": "user", "content": question}]
    specs = tool_specs()
    result = AgentResult(question, None)
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        t_llm = time.perf_counter()
        resp = llm.chat(SYSTEM_PROMPT, messages, specs)
        record = {"step": step, "llm_ms": _ms(t_llm), "text": resp.text,
                  "usage": resp.usage, "tool_calls": []}
        result.steps.append(record)
        messages.append({"role": "assistant", "text": resp.text, "tool_calls": resp.tool_calls})

        if not resp.tool_calls:
            messages.append({"role": "user", "content": "Please call final_answer with your answer."})
            continue

        tool_results = []
        for call in resp.tool_calls:
            t_tool = time.perf_counter()
            if call.name == "final_answer":
                try:
                    result.final = FinalAnswer.model_validate(call.args)
                    out = {"ok": True}
                except ValidationError as e:
                    out = {"ok": False, "error": f"final_answer did not validate: {e.errors()}"}
            else:
                out = call_tool(ctx, call.name, call.args)
            content = json.dumps(out, default=str)[:MAX_RESULT_CHARS]
            record["tool_calls"].append({"name": call.name, "args": call.args,
                                         "ok": out.get("ok", True), "ms": _ms(t_tool),
                                         "result_preview": content[:500]})
            tool_results.append({"id": call.id, "content": content,
                                 "is_error": not out.get("ok", True)})

        if result.final is not None:
            result.stopped = "final_answer"
            break
        messages.append({"role": "tool_results", "results": tool_results})
    else:
        result.stopped = "max_steps"
        result.final = FinalAnswer(answer=f"I could not finish within {max_steps} steps.",
                                   confidence="low", caveats=["Step limit reached; answer incomplete."])

    result.total_ms = _ms(t0)
    if trace_dir:
        result.trace_path = _write_trace(result, llm.name, Path(trace_dir))
    return result


def _ms(t):
    return round((time.perf_counter() - t) * 1000, 1)


def _write_trace(result, model, trace_dir):
    trace_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = trace_dir / f"trace_{stamp}.json"
    path.write_text(json.dumps({"model": model, **result.to_dict()}, indent=2, default=str))
    return path
