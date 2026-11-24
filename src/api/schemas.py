"""Validate API inputs; reject model, path, and state overrides."""
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
