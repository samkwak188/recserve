from typing import Annotated, Literal
from pydantic import Field
from .api import StrictModel

Identifier = Annotated[str, Field(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')]


class Change(StrictModel):
    item_id: Annotated[int, Field(ge=1, le=2147483647)]
    value: Literal[-1, 0, 1]


class Preferences(StrictModel):
    request_id: Identifier
    changes: Annotated[list[Change], Field(min_length=1, max_length=50)]


class Recommendation(StrictModel):
    request_id: Identifier
    k: Annotated[int, Field(ge=1, le=20)] = 10


class Event(StrictModel):
    event_id: Identifier
    request_id: Identifier
    item_id: Annotated[int, Field(ge=1, le=2147483647)]
    kind: Literal['shown', 'save', 'dismiss', 'watched']
    event_time_ms: Annotated[int, Field(ge=0)]
