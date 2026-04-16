from __future__ import annotations

from ..helpers import hop_to_score
from ..models import Goal, ScoreContext, ScoreResult


def _unavailable_reason(ctx: ScoreContext) -> str:
    if not ctx.predicted_files:
        return "predicted_files unavailable"
    if not ctx.gold_files:
        return "gold_files unavailable"
    if ctx.diagnostics.get("static_graph_available") is False:
        return "static_graph unavailable"
    if ctx.diagnostics.get("resolved_prediction_count") == 0:
        return "prediction nodes unresolved"
    if ctx.diagnostics.get("resolved_gold_count") == 0:
        return "gold nodes unresolved"
    return "gold_min_hops unavailable"


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
            reason=None if value is not None else _unavailable_reason(ctx),
            metadata={
                "gold_min_hops": ctx.gold_min_hops,
                "resolved_prediction_count": ctx.diagnostics.get(
                    "resolved_prediction_count"
                ),
                "resolved_gold_count": ctx.diagnostics.get("resolved_gold_count"),
                "static_graph_available": ctx.diagnostics.get(
                    "static_graph_available"
                ),
            },
        )


__all__ = ["GoldHopScore"]
