import argparse
import csv
import math
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import OmegaConf

from src.models.transformer import AraStudyTransformer, ModelArgs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def encode_text(tokenizer_model: Path, text: str) -> list[int]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    return sp.encode(text, out_type=int)


def get_batch(
    tokens: list[int],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - block_size - 2
    idx = [random.randint(0, max_start) for _ in range(batch_size)]
    x = torch.tensor([tokens[i : i + block_size] for i in idx], dtype=torch.long)
    y = torch.tensor(
        [tokens[i + 1 : i + 1 + block_size] for i in idx],
        dtype=torch.long,
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_loss(
    model: AraStudyTransformer,
    val_tokens: list[int],
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(val_tokens, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def run_training(config_path: str, tokenizer_id: str) -> None:
    cfg = OmegaConf.load(config_path)
    set_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = Path(cfg.paths.split_dir)
    tokenizer_path = Path(cfg.paths.tokenizer_dir) / f"{tokenizer_id}.model"

    train_text = read_text(split_dir / "train.txt")
    valid_text = read_text(split_dir / "valid.txt")
    train_tokens = encode_text(tokenizer_path, train_text)
    valid_tokens = encode_text(tokenizer_path, valid_text)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = sp.get_piece_size()

    model_args = ModelArgs(
        dim=int(cfg.model.dim),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        vocab_size=vocab_size,
        max_seq_len=int(cfg.model.max_seq_len),
        dropout=float(cfg.model.dropout),
    )
    model = AraStudyTransformer(model_args).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        betas=(float(cfg.training.betas[0]), float(cfg.training.betas[1])),
        weight_decay=float(cfg.training.weight_decay),
    )

    scaler = torch.amp.GradScaler(
        enabled=bool(cfg.training.use_amp and device.type == "cuda")
    )

    steps = int(cfg.training.steps)
    eval_every = int(cfg.training.eval_every)
    batch_size = int(cfg.training.batch_size)
    grad_accum = int(cfg.training.grad_accum_steps)
    block_size = int(cfg.model.max_seq_len)

    best_val = float("inf")
    run_output_dir = Path(cfg.paths.output_dir) / tokenizer_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{tokenizer_id}_metrics.csv"

    start_time = time.time()
    tokens_seen = 0

    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "tokens_seen",
                "elapsed_sec",
            ],
        )
        writer.writeheader()

        for step in range(steps):
            lr = cosine_lr(
                step=step,
                base_lr=float(cfg.training.learning_rate),
                warmup_steps=int(cfg.training.warmup_steps),
                total_steps=steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for _ in range(grad_accum):
                x, y = get_batch(train_tokens, batch_size, block_size, device)
                tokens_seen += x.numel()
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=bool(cfg.training.use_amp and device.type == "cuda"),
                ):
                    _, loss = model(x, y)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()
                running_loss += float(loss.item()) * grad_accum

            scaler.step(optimizer)
            scaler.update()

            val_loss = ""
            if (step + 1) % eval_every == 0 or step == 0 or step == steps - 1:
                val = evaluate_loss(model, valid_tokens, batch_size, block_size, device)
                val_loss = f"{val:.6f}"
                if val < best_val:
                    best_val = val
                    ckpt_path = run_output_dir / "best.pt"
                    torch.save(
                        {
                            "model_args": model_args,
                            "model_state_dict": model.state_dict(),
                            "step": step + 1,
                            "val_loss": val,
                            "tokenizer": tokenizer_id,
                        },
                        ckpt_path,
                    )

            writer.writerow(
                {
                    "step": step + 1,
                    "train_loss": f"{running_loss:.6f}",
                    "val_loss": val_loss,
                    "lr": f"{lr:.8f}",
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": f"{time.time() - start_time:.2f}",
                }
            )

    summary_path = log_dir / f"{tokenizer_id}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"tokenizer={tokenizer_id}\n")
        handle.write(f"vocab_size={vocab_size}\n")
        handle.write(f"total_params={total_params}\n")
        handle.write(f"best_val_loss={best_val:.6f}\n")
        handle.write(f"tokens_seen={tokens_seen}\n")
        handle.write(f"elapsed_sec={time.time() - start_time:.2f}\n")

    print(f"[exp01-sanity] tokenizer={tokenizer_id} completed")
    print(f"[exp01-sanity] best_val_loss={best_val:.6f}")
    print(f"[exp01-sanity] summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2A sanity training for exp01")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp01_sanity_train.yaml",
    )
    parser.add_argument("--tokenizer-id", type=str, required=True)
    args = parser.parse_args()
    run_training(config_path=args.config, tokenizer_id=args.tokenizer_id)


if __name__ == "__main__":
    main()
