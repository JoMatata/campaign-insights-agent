"""The structured output the agent must return."""
from typing import Literal

from pydantic import BaseModel, Field


class KeyNumber(BaseModel):
    label: str = Field(description="What the number is, e.g. 'ROAS for C01'")
    value: float
    unit: str | None = Field(default=None, description="e.g. 'KES', '%', 'x', 'count'")


class FinalAnswer(BaseModel):
    answer: str = Field(description="Short plain-language answer for a marketing manager")
    key_numbers: list[KeyNumber] = Field(default_factory=list)
    sql_used: list[str] = Field(default_factory=list, description="SQL that produced the numbers")
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = Field(default_factory=list)
