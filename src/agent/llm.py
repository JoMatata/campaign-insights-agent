"""A small provider-agnostic chat interface.

The agent loop only sees ToolCall, LLMResponse and a neutral message format:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall]}
  {"role": "tool_results", "results": [{"id", "content", "is_error"}]}
Each adapter converts that format to its own API.
"""
import json
from dataclasses import dataclass, field

from src import config


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLM:
    name = "base"

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        """Plain text completion, used for cluster labels and summaries."""
        return self.chat("You are a concise analyst.", [{"role": "user", "content": prompt}], []).text


class AnthropicLLM(LLM):
    def __init__(self, model=config.ANTHROPIC_MODEL, max_tokens=2048):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens

    def chat(self, system, messages, tools):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[self._convert(m) for m in messages])
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls, {"input_tokens": resp.usage.input_tokens,
                                         "output_tokens": resp.usage.output_tokens})

    @staticmethod
    def _convert(m):
        if m["role"] == "user":
            return {"role": "user", "content": m["content"]}
        if m["role"] == "assistant":
            blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                       for c in m["tool_calls"]]
            return {"role": "assistant", "content": blocks}
        return {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"],
             "is_error": r["is_error"]} for r in m["results"]]}


class OpenAILLM(LLM):
    """Optional adapter. Needs `pip install openai` and OPENAI_API_KEY."""

    def __init__(self, model=config.OPENAI_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def chat(self, system, messages, tools):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msgs.append({"role": "assistant", "content": m.get("text") or None,
                             "tool_calls": [{"id": c.id, "type": "function", "function": {
                                 "name": c.name, "arguments": json.dumps(c.args)}}
                                 for c in m["tool_calls"]] or None})
            else:
                msgs += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]}
                         for r in m["results"]]
        kwargs = {"model": self.model, "messages": msgs}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (msg.tool_calls or [])]
        return LLMResponse(msg.content or "", calls, {"input_tokens": resp.usage.prompt_tokens,
                                                      "output_tokens": resp.usage.completion_tokens})


class ScriptedLLM(LLM):
    """Fake LLM for tests and CI. Replays a script instead of calling an API.

    Each step is either a list of (tool_name, args) pairs or a function that takes the
    message history and returns an LLMResponse, so a step can react to tool results.
    """
    name = "scripted-fake"

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = 0
        self.seen = []

    def chat(self, system, messages, tools):
        self.seen.append(list(messages))
        if self.calls >= len(self.steps):
            return LLMResponse(text="(script finished)")
        step = self.steps[self.calls]
        self.calls += 1
        if callable(step):
            return step(messages)
        return LLMResponse(tool_calls=[ToolCall(f"call_{self.calls}_{i}", name, args)
                                       for i, (name, args) in enumerate(step)])


def get_llm(provider="anthropic"):
    return OpenAILLM() if provider == "openai" else AnthropicLLM()
