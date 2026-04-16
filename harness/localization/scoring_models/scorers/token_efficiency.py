from __future__ import annotations

from ..helpers import tokens_to_score
from ..models import Goal, ScoreContext, ScoreResult


class TokenEfficiencyScore:
    name = "token_efficiency"
    default_enabled = True
    default_visible_to_agent = True

    def compute(self, ctx: ScoreContext) -> ScoreResult:
        value = tokens_to_score(ctx.previous_total_token_count)
        return ScoreResult(
            name=self.name,
            value=value,
            goal=Goal.MAXIMIZE,
            available=value is not None,
            visible_to_agent=self.default_visible_to_agent,
            reason=None
            if value is not None
            else "previous_total_token_count unavailable",
            metadata={"previous_total_token_count": ctx.previous_total_token_count},
        )


__all__ = ["TokenEfficiencyScore"]
