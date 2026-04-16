from math import log1p


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def hop_to_score(hops: int | None) -> float | None:
    if hops is None:
        return None
    if hops < 0:
        raise ValueError("hops must be >= 0")
    return 1.0 / (1.0 + float(hops))


def tokens_to_score(total_tokens: float | None) -> float | None:
    if total_tokens is None:
        return None
    if total_tokens < 0:
        raise ValueError("total_tokens must be >= 0")
    return 1.0 / (1.0 + log1p(float(total_tokens)))


__all__ = [
    "clamp_01",
    "hop_to_score",
    "tokens_to_score",
]
