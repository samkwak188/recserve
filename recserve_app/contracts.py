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


class Movie(StrictModel):
    id: int
    title: str
    year: int | None
    genres: list[str]
    available: bool


class RecommendedMovie(Movie):
    source: Literal['popularity', 'centroid', 'als']


class RecommendationResult(StrictModel):
    request_id: str
    model_version: str
    policy_version: str
    preference_revision: int
    candidate_source: Literal['popularity', 'centroid', 'als']
    degraded: bool
    exhausted: bool
    items: list[RecommendedMovie]


class MoviePage(StrictModel):
    items: list[Movie]
    next_cursor: int | None


class Me(StrictModel):
    id: str
    consent_version: str | None
    required_consent: str
    preference_revision: int


class PreferenceResult(StrictModel):
    preference_revision: int
    count: int


class PreferencePage(StrictModel):
    items: list[Change]
    preference_revision: int


class WatchedMovie(Movie):
    saved: bool
    watched: bool
    dismissed: bool


class Watchlist(StrictModel):
    items: list[WatchedMovie]


class EventResult(StrictModel):
    accepted: bool
    duplicate: bool
