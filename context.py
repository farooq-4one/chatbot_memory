from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class UserContext:
    userId: str


class StoreRelevanceOutput(BaseModel):
    is_store_related_or_greeting: bool
    reasoning: str
