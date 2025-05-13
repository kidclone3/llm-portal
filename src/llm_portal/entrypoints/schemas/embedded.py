from datetime import datetime

import pydantic


class EmbeddedBase(pydantic.BaseModel):
    """
    Base class for an embedded
    """

    text: str
    provider: str
    model: str
    dimensions: int
    vector: list[float]

    model_config = pydantic.ConfigDict(from_attributes=True)

class EmbeddedResult(EmbeddedBase):
    """
    Embedded result class
    """

    model_config = pydantic.ConfigDict(from_attributes=True, extra="allow")

    id: str
    created_time: datetime = pydantic.Field(default_factory=datetime.now)
    updated_time: datetime = pydantic.Field(default_factory=datetime.now)

class EmbeddedResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(from_attributes=True)

    result: EmbeddedResult
