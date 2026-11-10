from typing import Literal

from pydantic import BaseModel, Field


class ReviewComment(BaseModel):
    file: str = Field(description="Path of the file, exactly as shown in the diff")
    line: int = Field(description="Line number in the new version of the file, on a line the diff adds")
    category: Literal["bug", "security"]
    severity: Literal["low", "medium", "high"]
    message: str = Field(description="What is wrong and why it matters")
    suggestion: str = Field(default="", description="How to fix it, optional")


class Review(BaseModel):
    summary: str
    comments: list[ReviewComment]
