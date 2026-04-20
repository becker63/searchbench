from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

from harness.localization.models import canonicalize_path

from .store import GraphStore

BACKTICK_PATTERN = re.compile(r"`([^`]+)`")
CODE_FENCE_PATTERN = re.compile(r"```[a-zA-Z0-9]*\n([^`]+?)```", re.DOTALL)
SLASHY_TOKEN_PATTERN = re.compile(r"[^\\s]+/[^\\s]+")
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    return list(OrderedDict.fromkeys(items))


def extract_anchors(text: str) -> list[str]:
    """
    Deterministic anchor extraction order:
    1) explicit/backticked file paths
    2) slash-containing path-like tokens
    3) identifiers (CamelCase/snake_case)
    4) identifiers found inside code fences
    """
    anchors: list[str] = []
    anchors.extend(match.strip() for match in BACKTICK_PATTERN.findall(text or ""))
    anchors.extend(match.strip() for match in SLASHY_TOKEN_PATTERN.findall(text or ""))
    anchors.extend(match.strip() for match in IDENTIFIER_PATTERN.findall(text or ""))
    for block in CODE_FENCE_PATTERN.findall(text or ""):
        anchors.extend(match.strip() for match in IDENTIFIER_PATTERN.findall(block))
    return _dedupe_preserve_order(a for a in anchors if a)


def resolve_anchors(store: GraphStore, anchors: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for anchor in anchors:
        path = canonicalize_path(anchor)
        file_hits = store.find_file(path)
        if file_hits:
            resolved.append(file_hits[0])
            continue
        symbol_hits = store.find_symbol(anchor)
        if symbol_hits:
            resolved.append(symbol_hits[0])
            continue
    return _dedupe_preserve_order(resolved)


def resolve_predictions(store: GraphStore, predictions: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for pred in predictions:
        path = canonicalize_path(pred)
        file_hits = store.find_file(path)
        if file_hits:
            resolved.append(file_hits[0])
            continue
        symbol_hits = store.find_symbol(pred)
        if symbol_hits:
            resolved.append(symbol_hits[0])
    return _dedupe_preserve_order(resolved)
