import pytest

from src.agent.tools import ToolContext
from src.data.generate import build
from src.text.embed import TfidfSvdEmbedder
from src.text.index import FeedbackIndex


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    build(db_path=path, n_feedback=800)
    return path


@pytest.fixture(scope="session")
def ctx(db_path):
    # TF-IDF keeps tests fast and offline; the sentence-transformer path is tested separately
    c = ToolContext(db_path)
    df = c.con.execute("SELECT * FROM feedback ORDER BY feedback_id").df()
    c._index = FeedbackIndex.build(df, TfidfSvdEmbedder(dim=64))
    return c
