from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VisualObservation(BaseModel):
    """Unverified observations made from an image, never book facts."""

    model_config = ConfigDict(extra="forbid")
    ocr_text: str = ""
    visual_description: str = ""
    visible_entities: list[str] = Field(default_factory=list, max_length=40)
    objects: list[str] = Field(default_factory=list, max_length=40)
    possible_scene: str = ""
    uncertainties: list[str] = Field(default_factory=list, max_length=40)


class VisualCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "visual"
    work_id: str
    visual_id: str
    chapter: int | None = None
    page: int | None = None
    caption: str = ""
    local_url: str
