"""Expose the same tools over MCP so Claude Desktop (or any MCP client) can use them.

Run: python -m src.mcp_server   (stdio transport)
"""
try:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2 renamed FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x

from src.agent import tools

server = FastMCP("campaign-insights")
_ctx = None


def ctx():
    global _ctx
    if _ctx is None:
        _ctx = tools.ToolContext()
    return _ctx


@server.tool()
def get_schema() -> dict:
    """List tables, columns, types and row counts of the campaign database."""
    return tools.get_schema(ctx())


@server.tool()
def run_sql(query: str, max_rows: int = 50) -> dict:
    """Run one read-only DuckDB SELECT/WITH query."""
    return tools.run_sql(ctx(), query, max_rows)


@server.tool()
def compare_campaigns(campaign_a: str, campaign_b: str, start_date: str | None = None,
                      end_date: str | None = None) -> dict:
    """Compare two campaigns on CTR, CVR, CPA, ROAS with two-proportion z-tests."""
    return tools.compare_campaigns(ctx(), campaign_a, campaign_b, start_date, end_date)


@server.tool()
def search_feedback(query: str, k: int = 10, county: str | None = None,
                    start_date: str | None = None, end_date: str | None = None) -> dict:
    """Semantic search over customer feedback (English, Swahili, Sheng)."""
    return tools.search_feedback(ctx(), query, k, county=county,
                                 start_date=start_date, end_date=end_date)


@server.tool()
def summarise_feedback_theme(theme: str, k: int = 60, county: str | None = None,
                             start_date: str | None = None, end_date: str | None = None) -> dict:
    """Summarise feedback about a theme: counts, top terms and representative quotes."""
    return tools.summarise_feedback_theme(ctx(), theme, k, county=county,
                                          start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    server.run()
