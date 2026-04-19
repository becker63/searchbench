from harness.utils.model_budgets import compute_prompt_char_budget
from harness.utils.failure_context import pack_failure_context


def test_prompt_budget_scales_by_model_and_attempt():
    gpt_budget = compute_prompt_char_budget("gpt-oss-120b", repair_attempt=0)
    llama_budget = compute_prompt_char_budget("llama3.1-8b", repair_attempt=0)
    assert gpt_budget > llama_budget
    later_budget = compute_prompt_char_budget("llama3.1-8b", repair_attempt=3)
    assert later_budget < llama_budget


def test_failure_context_packer_prioritizes_step_and_truncates():
    step = {"name": "basedpyright", "exit_code": 1, "stderr": "X" * 1000, "success": False}
    classified = {"type_errors": "missing stub"}
    ctx = pack_failure_context([step], classified, step, writer_error=None, model_name="llama3.1-8b", repair_attempt=0)
    assert "basedpyright" in ctx
    assert "exit_code=1" in ctx
    assert "type_errors" in ctx
    assert len(ctx) <= compute_prompt_char_budget("llama3.1-8b")
