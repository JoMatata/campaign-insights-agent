"""Command line entry point.

  python -m src.cli ask "Which channel has the best ROAS?"   # needs an API key
  python -m src.cli demo                                      # scripted run, no key
  python -m src.cli search "rider never came"                 # feedback search only
"""
import argparse
import os
import sys

from src import config
from src.agent.loop import run_agent
from src.agent.tools import ToolContext


def print_result(result):
    print("\nSteps:")
    for s in result.steps:
        for c in s["tool_calls"]:
            status = "ok" if c["ok"] else "ERROR"
            args = {k: (v[:70] + "...") if isinstance(v, str) and len(v) > 70 else v
                    for k, v in c["args"].items() if c["name"] != "final_answer"}
            print(f"  {s['step']}. {c['name']}({args}) -> {status} [{c['ms']:.0f} ms]")
            if not c["ok"]:
                print(f"     {c['result_preview'][:160]}")
    f = result.final
    print(f"\nAnswer ({f.confidence} confidence):\n  {f.answer}\n\nKey numbers:")
    for k in f.key_numbers:
        print(f"  - {k.label}: {k.value:g} {k.unit or ''}")
    if f.caveats:
        print("Caveats:\n" + "\n".join(f"  - {c}" for c in f.caveats))
    print(f"\nStopped: {result.stopped}, {len(result.tool_calls)} tool calls, {result.total_ms:.0f} ms")
    if result.trace_path:
        print(f"Trace: {result.trace_path.relative_to(config.ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ask.add_argument("--max-steps", type=int, default=8)
    sub.add_parser("demo")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)

    ctx = ToolContext()
    if args.cmd == "search":
        hits = ctx.index.search(args.query, k=args.k)
        print(f"embedder: {ctx.index.embedder.name}")
        for _, r in hits.iterrows():
            print(f"  {r['score']:.3f}  [{r['source']}, {r['county']}] {r['text']}")
        return

    if args.cmd == "demo":
        from src.demo import DEMO_QUESTION, demo_llm
        print("SCRIPTED DEMO: no model is called. Tool calls are replayed; results are real.")
        question, llm = DEMO_QUESTION, demo_llm()
    else:
        key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.getenv(key):
            sys.exit(f"{key} is not set. Try `python -m src.cli demo` for a scripted run.")
        from src.agent.llm import get_llm
        question, llm = args.question, get_llm(args.provider)

    print(f"Q: {question}")
    result = run_agent(question, llm, ctx, max_steps=getattr(args, "max_steps", 8),
                       trace_dir=config.TRACE_DIR)
    print_result(result)


if __name__ == "__main__":
    main()
