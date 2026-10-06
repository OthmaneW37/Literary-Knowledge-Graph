from __future__ import annotations


class SpoilerPolicy:
    """Central chapter boundary for text and visual evidence."""

    @staticmethod
    def allows(chapter: int | str | None, max_chapter: int | None) -> bool:
        if max_chapter is None:
            return True
        try:
            return chapter is not None and int(chapter) <= max_chapter
        except (TypeError, ValueError):
            return False

    @classmethod
    def filter(cls, items, max_chapter: int | None):
        return [item for item in items if cls.allows(getattr(item, "chapter", None), max_chapter)]
