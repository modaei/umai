from __future__ import annotations

from typing import Literal
from uuid import UUID
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID | None = None
    # The request-integrity guard owns the user-visible 500-character limit so
    # it can record a clear refusal in the conversation. Keep a larger
    # transport-level ceiling to protect the API from unbounded input.
    message: str = Field(min_length=1, max_length=10_000)
    user_id: int = Field(gt=0)
    site_id: int = Field(gt=0)
    response_mode: Literal["fast", "intelligent"] = "fast"
    current_date: date | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value

    def conversation_id_text(self) -> str | None:
        if self.conversation_id is None:
            return None
        return str(self.conversation_id)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    assistant_message_id: int
    data_entry_intent: dict | None = None
    view_intent: dict | None = None


class RodPumpHealthReportRequest(BaseModel):
    """Internal batch request used by the scheduled email report."""

    model_config = ConfigDict(extra="forbid")

    site_id: int = Field(gt=0)
    well_ids: list[int] | None = None
    as_of_time: datetime | None = None

    @field_validator("well_ids")
    @classmethod
    def validate_well_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(well_id <= 0 for well_id in value):
            raise ValueError("well_ids must contain positive integers")
        return list(dict.fromkeys(value))


class WeeklyOverviewRequest(BaseModel):
    """Internal scheduled-email request for an UMAI weekly overview section."""

    model_config = ConfigDict(extra="forbid")

    site_id: int = Field(gt=0)
    start_date: date
    end_date: date

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, value: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date is not None and value < start_date:
            raise ValueError("end_date must be on or after start_date")
        return value


class WeeklyOverviewResponse(BaseModel):
    overview: str = Field(min_length=1)
