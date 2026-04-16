from __future__ import annotations

from ..helpers import hop_to_score
from ..models import Goal, ScoreContext, ScoreResult


class GoldHopScore:
    name = "gold_hop"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = hop_to_score(ctx.gold_min_hops)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            available=value is not None,
            visible_to_agent=self.default_visible_to_agent,
            reason=None if value is not None else "gold_min_hops unavailable",
            metadata={"gold_min_hops": ctx.gold_min_hops},
        )


__all__ = ["GoldHopScore"]
