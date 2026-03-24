from harness.policy_loader import adapt_score_fn


def test_score_accepts_variable_arity():
    def s1(n):
        return 1.0

    def s2(n, s):
        return 1.0

    def s3(n, s, c):
        return 1.0

    for fn in (s1, s2, s3):
        wrapped = adapt_score_fn(fn)
        assert isinstance(wrapped(None, None, None), float)
