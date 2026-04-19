from __future__ import annotations

from ..helpers import hop_to_score
from ..models import Goal, ScoreContext, ScoreResult


def _unavailable_reason(ctx: ScoreContext) -> str:
    if not ctx.predicted_files:
        return "predicted_files unavailable"
    if ctx.diagnostics.get("static_graph_available") is False:
        return "static_graph unavailable"
    if ctx.diagnostics.get("resolved_prediction_count") == 0:
        return "prediction nodes unresolved"
    if ctx.diagnostics.get("issue_anchor_count") == 0:
        return "issue anchors unresolved"
    return "issue_min_hops unavailable"


class IssueHopScore:
    name = "issue_hop"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = hop_to_score(ctx.issue_min_hops)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            available=value is not None,
            visible_to_agent=self.default_visible_to_agent,
            reason=None if value is not None else _unavailable_reason(ctx),
            metadata={
                "issue_min_hops": ctx.issue_min_hops,
                "issue_anchor_count": ctx.diagnostics.get("issue_anchor_count"),
                "resolved_prediction_count": ctx.diagnostics.get(
                    "resolved_prediction_count"
                ),
                "static_graph_available": ctx.diagnostics.get(
                    "static_graph_available"
                ),
            },
        )


__all__ = ["IssueHopScore"]
