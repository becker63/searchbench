from __future__ import annotations

from ..models import ScoreComponent
from .gold_hop import GoldHopScore
from .issue_hop import IssueHopScore
from .token_efficiency import TokenEfficiencyScore

SCORE_REGISTRY: dict[str, ScoreComponent] = {
    "gold_hop": GoldHopScore(),
    "issue_hop": IssueHopScore(),
    "token_efficiency": TokenEfficiencyScore(),
}

__all__ = [
    "GoldHopScore",
    "IssueHopScore",
    "SCORE_REGISTRY",
    "TokenEfficiencyScore",
]
