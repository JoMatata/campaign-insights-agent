"""Tools the agent can call. Each returns a JSON-serialisable dict.

Errors are returned as {"ok": False, "error": ...} rather than raised, so the model
sees what went wrong and can try again.
"""
from src import config
from src.agent import sql_guard
from src.agent.schemas import FinalAnswer
from src.agent.stats import campaign_metrics, two_proportion_ztest

TABLE_NOTES = {
    "campaigns": "One row per campaign. budget_kes is the planned budget in Kenyan shillings.",
    "daily_performance": "Grain: campaign_id x date x county x device. spend and revenue in KES.",
    "creatives": "Ad creatives per campaign (format, language).",
    "feedback": "Customer feedback and ad comments. text mixes English, Swahili and Sheng. "
                "rating (1-5) exists only for app_review and survey.",
}


class ToolContext:
    """Holds the DB connection and the (lazily built) feedback index."""

    def __init__(self, db_path=config.DB_PATH, index=None, summariser=None):
        self.con = sql_guard.connect(db_path)
        self.db_path = db_path
        self._index = index
        self.summariser = summariser  # optional LLM for abstractive summaries

    @property
    def index(self):
        if self._index is None:
            from src.text.index import FeedbackIndex
            self._index = FeedbackIndex.load_or_build(self.db_path)
        return self._index


def get_schema(ctx):
    tables = {}
    for (name,) in ctx.con.execute("SELECT table_name FROM information_schema.tables "
                                   "ORDER BY table_name").fetchall():
        cols = ctx.con.execute("SELECT column_name, data_type FROM information_schema.columns "
                               "WHERE table_name = ? ORDER BY ordinal_position", [name]).fetchall()
        n = ctx.con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        tables[name] = {"rows": n, "columns": dict(cols), "notes": TABLE_NOTES.get(name, "")}
    return {"ok": True, "dialect": "DuckDB", "tables": tables}


def run_sql(ctx, query, max_rows=50):
    return sql_guard.run_query(ctx.con, query, max_rows=min(int(max_rows), sql_guard.MAX_ROWS))


def compare_campaigns(ctx, campaign_a, campaign_b, start_date=None, end_date=None):
    sql = ("SELECT campaign_id, sum(impressions), sum(clicks), sum(conversions), sum(spend), "
           "sum(revenue) FROM daily_performance WHERE campaign_id IN (?, ?) "
           "AND date >= coalesce(?::DATE, DATE '1900-01-01') "
           "AND date <= coalesce(?::DATE, DATE '2999-12-31') GROUP BY 1")
    rows = {r[0]: r[1:] for r in ctx.con.execute(sql, [campaign_a, campaign_b, start_date, end_date]).fetchall()}
    missing = [c for c in (campaign_a, campaign_b) if c not in rows]
    if missing:
        return {"ok": False, "error": f"No data for {missing}. Check campaign_id values in campaigns."}
    a, b = (campaign_metrics(*map(float, rows[c])) for c in (campaign_a, campaign_b))
    return {
        "ok": True,
        "a": {"campaign_id": campaign_a, **a},
        "b": {"campaign_id": campaign_b, **b},
        "ctr_test": two_proportion_ztest(a["clicks"], a["impressions"], b["clicks"], b["impressions"]),
        "cvr_test": two_proportion_ztest(a["conversions"], a["clicks"], b["conversions"], b["clicks"]),
        "note": "Tests compare a vs b. With millions of impressions small gaps are 'significant'; "
                "judge the size of the difference, not only the p-value.",
    }


def search_feedback(ctx, query, k=10, **filters):
    hits = ctx.index.search(query, k=min(int(k), 50), **filters)
    cols = ["feedback_id", "created_at", "source", "campaign_id", "county", "rating", "text", "score"]
    return {"ok": True, "embedder": ctx.index.embedder.name,
            "results": [{c: sql_guard._jsonable(r[c]) for c in cols} for _, r in hits.iterrows()]}


def summarise_feedback_theme(ctx, theme, k=60, **filters):
    from src.text.cluster import representative, top_terms

    idx = ctx.index
    hits = idx.search(theme, k=min(int(k), 200), **filters)
    if hits.empty:
        return {"ok": False, "error": "No matching feedback."}
    vecs = idx.vectors[hits.index.to_numpy()]
    quotes = hits["text"].iloc[representative(vecs, n=5)].tolist()
    out = {
        "ok": True,
        "theme": theme,
        "n_snippets": len(hits),
        "mean_similarity": round(float(hits["score"].mean()), 3),
        "top_terms": top_terms(hits["text"], idx.df["text"], n=10),
        "by_source": hits["source"].value_counts().to_dict(),
        "by_county": hits["county"].value_counts().head(5).to_dict(),
        "by_week": hits["created_at"].astype("datetime64[ns]").dt.to_period("W").dt.start_time
                   .dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
        "avg_rating": None if hits["rating"].isna().all() else round(float(hits["rating"].mean()), 2),
        "representative_quotes": quotes,
        "caveat": "These are the top-k most similar snippets, not an exact count of the theme.",
    }
    if ctx.summariser is not None:
        out["summary"] = ctx.summariser.complete(
            "Summarise the main customer complaint or praise in these snippets in 2 sentences. "
            "Swahili and Sheng may appear.\n\n" + "\n".join(f"- {t}" for t in hits["text"].head(30)))
    return out


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required)}


FILTERS = {"county": {"type": "string"}, "start_date": {"type": "string"},
           "end_date": {"type": "string"}}

TOOLS = {
    "get_schema": (get_schema, "List tables, columns, types and row counts. Call this first.", _obj({})),
    "run_sql": (run_sql, "Run one read-only DuckDB SELECT/WITH query. Results are capped; aggregate "
                "in SQL. On error you get the message back and can fix the query.",
                _obj({"query": {"type": "string"}, "max_rows": {"type": "integer", "default": 50}}, ["query"])),
    "compare_campaigns": (compare_campaigns, "Compare two campaigns on CTR, CVR, CPA and ROAS, with "
                          "two-proportion z-tests for CTR and CVR. Dates are optional YYYY-MM-DD.",
                          _obj({"campaign_a": {"type": "string"}, "campaign_b": {"type": "string"},
                                "start_date": {"type": "string"}, "end_date": {"type": "string"}},
                               ["campaign_a", "campaign_b"])),
    "search_feedback": (search_feedback, "Semantic search over customer feedback. Works across "
                        "English, Swahili and Sheng. Optional filters: county, campaign_id, source, "
                        "start_date, end_date (YYYY-MM-DD).",
                        _obj({"query": {"type": "string"}, "k": {"type": "integer", "default": 10},
                              **FILTERS, "campaign_id": {"type": "string"},
                              "source": {"type": "string"}}, ["query"])),
    "summarise_feedback_theme": (summarise_feedback_theme, "Summarise feedback about a theme (e.g. "
                                 "'M-Pesa payment failures'): counts by source/county/week, top "
                                 "terms and representative quotes. Same filters as search_feedback.",
                                 _obj({"theme": {"type": "string"}, "k": {"type": "integer", "default": 60},
                                       **FILTERS}, ["theme"])),
}

FINAL_TOOL = {"name": "final_answer",
              "description": "Call this once with your final answer. Put every number you quote in key_numbers.",
              "parameters": FinalAnswer.model_json_schema()}


def tool_specs():
    specs = [{"name": n, "description": d, "parameters": p} for n, (_, d, p) in TOOLS.items()]
    return specs + [FINAL_TOOL]


def call_tool(ctx, name, args):
    if name not in TOOLS:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        return TOOLS[name][0](ctx, **args)
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # keep the loop alive and let the model see the problem
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
