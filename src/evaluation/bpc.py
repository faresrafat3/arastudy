import math


def total_non_space_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def compute_bpc_from_totals(total_nats: float, total_chars: int) -> float:
    if total_chars <= 0:
        raise ValueError("total_chars must be > 0")
    total_loss_bits = total_nats / math.log(2)
    return total_loss_bits / total_chars


def compute_bpc_from_batches(
    batch_losses_nats: list[float],
    batch_token_counts: list[int],
    total_chars: int,
    total_tokens_in_eval_text: int,
) -> float:
    if len(batch_losses_nats) != len(batch_token_counts):
        raise ValueError("batch losses and token counts must have the same length")
    if total_chars <= 0:
        raise ValueError("total_chars must be > 0")
    if total_tokens_in_eval_text <= 0:
        raise ValueError("total_tokens_in_eval_text must be > 0")

    evaluated_tokens = sum(batch_token_counts)
    if evaluated_tokens <= 0:
        raise ValueError("evaluated tokens must be > 0")

    total_nats = sum(
        loss * tokens for loss, tokens in zip(batch_losses_nats, batch_token_counts)
    )
    total_loss_bits = total_nats / math.log(2)

    effective_chars = total_chars * (evaluated_tokens / total_tokens_in_eval_text)
    if effective_chars <= 0:
        raise ValueError("effective chars must be > 0")

    return total_loss_bits / effective_chars
