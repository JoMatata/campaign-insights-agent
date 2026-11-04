"""Minimal web page for the agent. Run: streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent.loop import run_agent  # noqa: E402
from src.agent.tools import ToolContext  # noqa: E402
from src.demo import DEMO_QUESTION, demo_llm  # noqa: E402

st.set_page_config(page_title="Campaign Insights Agent")
st.title("Campaign Insights Agent")
st.caption("Synthetic data for a fictional Kenyan brand. Answers come from SQL and feedback search.")


@st.cache_resource
def get_ctx():
    return ToolContext()


has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
if not has_key:
    st.info("No ANTHROPIC_API_KEY found, so only the scripted demo is available.")

question = st.text_area("Question", DEMO_QUESTION if not has_key else "", disabled=not has_key)
if st.button("Ask" if has_key else "Run scripted demo"):
    if has_key:
        from src.agent.llm import get_llm
        llm = get_llm()
    else:
        llm, question = demo_llm(), DEMO_QUESTION
    with st.spinner("Working..."):
        result = run_agent(question, llm, get_ctx(), trace_dir=config.TRACE_DIR)
    f = result.final
    st.subheader("Answer")
    st.write(f.answer)
    st.write(f"Confidence: **{f.confidence}**")
    if f.key_numbers:
        st.table([k.model_dump() for k in f.key_numbers])
    for c in f.caveats:
        st.warning(c)
    with st.expander("SQL used"):
        for q in f.sql_used:
            st.code(q, language="sql")
    with st.expander(f"Trace ({len(result.tool_calls)} tool calls, {result.total_ms:.0f} ms)"):
        st.json(result.to_dict()["steps"])
