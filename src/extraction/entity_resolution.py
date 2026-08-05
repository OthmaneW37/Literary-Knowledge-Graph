from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text


def token_set(text: str) -> set[str]:
    return {tok for tok in re.split(r"\s+", normalize_key(text)) if tok}


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_key(a), normalize_key(b)).ratio()


def token_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def combined_name_similarity(a: str, b: str) -> float:
    s1 = string_similarity(a, b)
    s2 = token_similarity(a, b)
    return (0.7 * s1) + (0.3 * s2)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class EntityCandidate:
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)
    mentions: int = 0


@dataclass
class Mention:
    text: str
    context: str = ""
    chapter: str = ""


BANNED_MENTION_STARTS = {
    "the",
    "a",
    "an",
    "his",
    "her",
    "their",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "it",
    "they",
    "we",
    "i",
    "you",
}

BANNED_MENTION_CONTAINS = {
    "clock",
    "door",
    "room",
    "pain",
    "silence",
    "weather",
    "train",
    "picture",
    "chair",
    "muff",
    "frame",
    "table",
    "window",
    "bed",
    "wall",
    "needle",
    "lamp",
    "sound",
    "voice",
    "conversation",
    "budget",
    "work",
    "time",
    "itch",
    "legs",
    "gas",
    "gaslight",
    "couch",
    "newspaper",
    "alarm",
    "locksmith",
}

VERB_HINTS = {
    "would",
    "could",
    "should",
    "was",
    "were",
    "is",
    "are",
    "be",
    "being",
    "been",
    "run",
    "leave",
    "keep",
    "speak",
    "tug",
    "throw",
    "sit",
    "sleep",
    "learn",
    "go",
    "come",
    "see",
    "hear",
    "exclaim",
    "ask",
    "say",
}


def looks_like_character(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = normalize_text(text)
    if not t:
        return False

    if len(t.split()) > 4:
        return False

    low = t.lower()

    if low in {"he", "she", "it", "they", "his", "her", "their", "him", "them"}:
        return False

    if any(x in low for x in BANNED_MENTION_CONTAINS):
        return False

    if any(v in low.split() for v in VERB_HINTS):
        return False

    first = t.split()[0].lower().strip(".,;:!?\"'")
    if first in BANNED_MENTION_STARTS:
        return False

    if t.startswith('"') or t.endswith('"'):
        return False

    return True


def context_score(m1: Mention, entity: EntityCandidate) -> float:
    if not m1.context or not entity.contexts:
        return 0.0
    mention_ctx = token_set(m1.context)
    entity_ctx = set()
    for c in entity.contexts:
        entity_ctx |= token_set(c)
    return jaccard(mention_ctx, entity_ctx)


def candidate_score(mention: Mention, entity: EntityCandidate) -> float:
    name_score = max(
        combined_name_similarity(mention.text, entity.canonical_name),
        max((combined_name_similarity(mention.text, alias) for alias in entity.aliases), default=0.0),
    )
    ctx_score = context_score(mention, entity)
    chapter_bonus = 0.1 if mention.chapter and mention.chapter in entity.contexts else 0.0
    return (0.75 * name_score) + (0.25 * ctx_score) + chapter_bonus


def resolve_mention(
    mention: Mention,
    existing: list[EntityCandidate],
    threshold: float = 0.82,
) -> tuple[str, list[str], bool]:
    if not mention.text.strip() or not looks_like_character(mention.text):
        return "", [], False

    best_entity = None
    best_score = 0.0

    for entity in existing:
        score = candidate_score(mention, entity)
        if score > best_score:
            best_score = score
            best_entity = entity

    if best_entity and best_score >= threshold:
        best_entity.aliases.add(normalize_text(mention.text))
        if mention.context:
            best_entity.contexts.add(normalize_text(mention.context))
        if mention.chapter:
            best_entity.contexts.add(normalize_text(mention.chapter))
        best_entity.mentions += 1
        return best_entity.canonical_name, sorted(best_entity.aliases), True

    canonical = normalize_text(mention.text)
    new_entity = EntityCandidate(
        canonical_name=canonical,
        aliases=set(),
        contexts={normalize_text(mention.context)} if mention.context else set(),
        mentions=1,
    )
    if mention.chapter:
        new_entity.contexts.add(normalize_text(mention.chapter))
    existing.append(new_entity)
    return canonical, [], False


def absorb_entities(entities: list[EntityCandidate], merge_threshold: float = 0.88) -> list[EntityCandidate]:
    merged: list[EntityCandidate] = []

    for entity in entities:
        if not looks_like_character(entity.canonical_name):
            continue

        matched = False
        for target in merged:
            score = max(
                combined_name_similarity(entity.canonical_name, target.canonical_name),
                max((combined_name_similarity(entity.canonical_name, a) for a in target.aliases), default=0.0),
            )
            if score >= merge_threshold:
                target.aliases.add(entity.canonical_name)
                target.aliases |= entity.aliases
                target.contexts |= entity.contexts
                target.mentions += entity.mentions
                matched = True
                break

        if not matched:
            merged.append(entity)

    for entity in merged:
        entity.aliases = {a for a in entity.aliases if normalize_key(a) != normalize_key(entity.canonical_name)}

    return merged