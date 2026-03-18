import math

from src.evaluation.bpc import compute_bpc_from_batches, compute_bpc_from_totals


def test_compute_bpc_from_totals_simple_case() -> None:
    total_nats = math.log(2) * 100.0
    total_chars = 50
    bpc = compute_bpc_from_totals(total_nats=total_nats, total_chars=total_chars)
    assert abs(bpc - 2.0) < 1e-9


def test_compute_bpc_from_batches_matches_manual_formula() -> None:
    losses = [0.5, 1.0]
    tokens = [100, 50]
    total_chars = 300
    total_tokens_in_eval_text = 600

    total_nats = sum(l * t for l, t in zip(losses, tokens))
    total_bits = total_nats / math.log(2)
    effective_chars = total_chars * (sum(tokens) / total_tokens_in_eval_text)
    expected = total_bits / effective_chars

    got = compute_bpc_from_batches(
        batch_losses_nats=losses,
        batch_token_counts=tokens,
        total_chars=total_chars,
        total_tokens_in_eval_text=total_tokens_in_eval_text,
    )
    assert abs(got - expected) < 1e-9


if __name__ == "__main__":
    test_compute_bpc_from_totals_simple_case()
    test_compute_bpc_from_batches_matches_manual_formula()
    print("tests.test_bpc_calculation: PASSED")
