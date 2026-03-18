import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import sentencepiece as spm
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.cleaning.prepare_exp01_corpus import normalize_arabic
from src.models.transformer import AraStudyTransformer, ModelArgs

DEFAULT_PROMPTS = [
    "في يوم من الايام",
    "اللغه العربيه",
    "كان الباحث الصغير",
    "تاريخ العلوم",
    "المدرسه في الصباح",
]


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    if not path.exists():
        return DEFAULT_PROMPTS

    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if line:
            prompts.append(line)
    return prompts or DEFAULT_PROMPTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    unk_id: int,
    disallow_unk: bool,
) -> int:
    logits = logits / max(temperature, 1e-5)
    if disallow_unk and 0 <= unk_id < logits.shape[-1]:
        logits[unk_id] = -1e9
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        logits = torch.where(logits < v[-1], torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_sample(
    model: AraStudyTransformer,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    disallow_unk: bool,
    device: torch.device,
) -> str:
    token_ids = sp.encode(prompt, out_type=int)
    if not token_ids:
        token_ids = [sp.bos_id()] if sp.bos_id() >= 0 else [1]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([token_ids], dtype=torch.long, device=device)
            x = x[:, -model.args.max_seq_len :]
            logits, _ = model(x)
            token_ids.append(
                sample_next_token(
                    logits[0, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    unk_id=sp.unk_id(),
                    disallow_unk=disallow_unk,
                )
            )

    return sp.decode(token_ids)


def get_batch(
    token_array: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_array.shape[0] - block_size - 2
    if max_start <= 0:
        raise ValueError(
            "Token array too small for block_size="
            f"{block_size}; tokens={token_array.shape[0]}"
        )
    idx = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
    x_np = np.stack([token_array[i : i + block_size] for i in idx], axis=0)
    y_np = np.stack([token_array[i + 1 : i + 1 + block_size] for i in idx], axis=0)
    x = torch.from_numpy(x_np.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64, copy=False)).to(device)
    return x, y


def build_or_load_token_cache(
    split_file: Path,
    sp: spm.SentencePieceProcessor,
    cache_dir: Path,
) -> tuple[np.memmap, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = split_file.stem
    token_file = cache_dir / f"{stem}.u32.bin"
    meta_file = cache_dir / f"{stem}.meta.json"

    if token_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        total_tokens = int(meta["total_tokens"])
        tokens = np.memmap(token_file, dtype=np.uint32, mode="r", shape=(total_tokens,))
        return tokens, total_tokens

    total_tokens = 0
    with open(split_file, encoding="utf-8") as src, open(token_file, "wb") as out:
        for raw in src:
            text = raw.strip()
            if not text:
                continue
            ids = sp.encode(text, out_type=int)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.uint32)
            out.write(arr.tobytes())
            total_tokens += int(arr.size)

    meta_file.write_text(
        json.dumps(
            {
                "split_file": str(split_file),
                "token_file": str(token_file),
                "total_tokens": total_tokens,
                "dtype": "uint32",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tokens = np.memmap(token_file, dtype=np.uint32, mode="r", shape=(total_tokens,))
    return tokens, total_tokens


@torch.no_grad()
def evaluate(
    model: AraStudyTransformer,
    val_tokens: np.memmap,
    batch_size: int,
    block_size: int,
    max_batches: int,
    val_tokens_per_char: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses = []
    total_nats = 0.0
    total_tokens = 0

    for _ in range(max_batches):
        x, y = get_batch(val_tokens, batch_size, block_size, device)
        _, loss = model(x, y)
        loss_val = float(loss.item())
        losses.append(loss_val)
        tokens = x.numel()
        total_tokens += tokens
        total_nats += loss_val * tokens

    avg_loss = sum(losses) / max(len(losses), 1)
    bits_per_token = (total_nats / math.log(2)) / max(total_tokens, 1)
    bpc = bits_per_token * val_tokens_per_char
    model.train()
    return avg_loss, bpc


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def maybe_normalize_prompt(prompt: str, token_cfg: DictConfig, enabled: bool) -> str:
    if not enabled:
        return prompt
    return normalize_arabic(
        prompt, keep_diacritics=bool(token_cfg.cleaning.keep_diacritics)
    )


def append_research_log_entry(
    experiment_name: str,
    run_id: str,
    tokenizer_id: str,
    seed: int,
    summary: dict,
    metrics_file: Path,
    log_file: Path,
) -> None:
    if not log_file.exists():
        return

    tag = f"{experiment_name}::{run_id}::{tokenizer_id}::s{seed}"
    existing = log_file.read_text(encoding="utf-8")
    if tag in existing:
        return

    entry = (
        "\n---\n\n"
        f"### Auto Log — {experiment_name} ({run_id} / {tokenizer_id})\n\n"
        f"Tag: `{tag}`\n\n"
        f"- Date: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
        f"- Seed: `{seed}`\n"
        f"- Best Val Loss: `{float(summary.get('best_val_loss', 0.0)):.6f}`\n"
        f"- Best BPC: `{float(summary.get('best_bpc', 0.0)):.6f}`\n"
        f"- Tokens Seen: `{int(summary.get('tokens_seen', 0))}`\n"
        f"- Elapsed (sec): `{float(summary.get('elapsed_sec', 0.0)):.2f}`\n"
        f"- Peak VRAM (GB): `{float(summary.get('peak_vram_gb', 0.0)):.4f}`\n"
        f"- Metrics CSV: `{metrics_file}`\n"
    )
    log_file.write_text(existing + entry, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full exp01 model")
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_full_train.yaml"
    )
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    token_cfg = cast(DictConfig, OmegaConf.load(cfg.paths.tokenization_config))
    seed = int(args.seed if args.seed is not None else cfg.experiment.seed)
    set_seed(seed)

    tokenizer_id = args.tokenizer_id or str(cfg.run.tokenizer_id)
    prompt_file = OmegaConf.select(cfg, "generation.prompt_file")
    prompts = load_prompts(str(prompt_file) if prompt_file else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    split_dir = Path(cfg.paths.split_dir)
    tokenizer_path = Path(cfg.paths.tokenizer_dir) / f"{tokenizer_id}.model"
    out_dir = Path(cfg.paths.output_dir) / args.run_id / tokenizer_id
    log_dir = Path(cfg.paths.log_dir) / args.run_id
    cache_dir = log_dir / f"{tokenizer_id}_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_split = split_dir / "train.txt"
    valid_split = split_dir / "valid.txt"
    val_non_space_chars = 0
    with open(valid_split, encoding="utf-8") as valid_handle:
        for raw in valid_handle:
            val_non_space_chars += sum(1 for ch in raw if not ch.isspace())

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_tokens, train_token_count = build_or_load_token_cache(
        train_split,
        sp,
        cache_dir=cache_dir,
    )
    valid_tokens, valid_token_count = build_or_load_token_cache(
        valid_split,
        sp,
        cache_dir=cache_dir,
    )
    print(
        "[full] token cache ready "
        f"train_tokens={train_token_count} valid_tokens={valid_token_count}"
    )
    val_tokens_per_char = valid_token_count / max(val_non_space_chars, 1)

    model_args = ModelArgs(
        dim=int(cfg.model.dim),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        vocab_size=sp.get_piece_size(),
        max_seq_len=int(cfg.model.max_seq_len),
        dropout=float(cfg.model.dropout),
    )
    model = AraStudyTransformer(model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        betas=(float(cfg.training.betas[0]), float(cfg.training.betas[1])),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = torch.amp.GradScaler(
        enabled=bool(cfg.training.use_amp and device.type == "cuda")
    )

    start_step = 0
    best_val_loss = float("inf")
    best_bpc = float("inf")
    patience_steps_cfg = OmegaConf.select(cfg, "training.early_stopping_patience_steps")
    if patience_steps_cfg is not None:
        patience_steps = int(patience_steps_cfg)
    else:
        patience_evals = int(
            OmegaConf.select(cfg, "training.early_stopping_patience", default=4)
        )
        patience_steps = patience_evals * int(cfg.training.eval_every)
    last_improve_step = 0

    latest_ckpt = out_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        start_step = int(payload["step"])
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        best_bpc = float(payload.get("best_bpc", best_bpc))
        last_improve_step = int(payload.get("last_improve_step", 0))
        print(f"[full] resumed from step={start_step}")

    metrics_file = log_dir / f"{tokenizer_id}_metrics.csv"
    samples_file = log_dir / f"{tokenizer_id}_generation.md"
    summary_file = log_dir / f"{tokenizer_id}_summary.json"

    write_header = not metrics_file.exists() or start_step == 0
    steps = int(cfg.training.steps)
    eval_every = int(cfg.training.eval_every)
    save_every = int(cfg.training.save_every)
    batch_size = int(os.getenv("TRAINING_BATCH_SIZE", cfg.training.batch_size))
    grad_accum = int(
        os.getenv("TRAINING_GRAD_ACCUM_STEPS", cfg.training.grad_accum_steps)
    )
    block_size = int(cfg.model.max_seq_len)
    eval_max_batches = int(cfg.training.eval_max_batches)

    t0 = time.time()
    tokens_seen = 0

    with open(metrics_file, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "bpc",
                "lr",
                "tokens_seen",
                "tokens_per_sec",
                "peak_vram_gb",
                "elapsed_sec",
            ],
        )
        if write_header:
            writer.writeheader()

        for step in range(start_step, steps):
            lr = cosine_lr(
                step,
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
                running_loss += float(loss.item())

            train_loss_value = running_loss

            scaler.step(optimizer)
            scaler.update()

            val_loss = ""
            bpc = ""
            if (step + 1) % eval_every == 0 or step == 0 or step == steps - 1:
                val, bpc_value = evaluate(
                    model,
                    valid_tokens,
                    batch_size=batch_size,
                    block_size=block_size,
                    max_batches=eval_max_batches,
                    val_tokens_per_char=val_tokens_per_char,
                    device=device,
                )
                val_loss = f"{val:.6f}"
                bpc = f"{bpc_value:.6f}"
                print(
                    f"[full] step={step + 1} "
                    f"train_loss={train_loss_value:.6f} "
                    f"val_loss={val_loss} bpc={bpc}"
                )

                improved = val < best_val_loss
                if improved:
                    best_val_loss = val
                    best_bpc = bpc_value
                    last_improve_step = step + 1
                    torch.save(
                        {
                            "step": step + 1,
                            "model_args": model_args,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "best_val_loss": best_val_loss,
                            "best_bpc": best_bpc,
                            "last_improve_step": last_improve_step,
                        },
                        out_dir / "best.pt",
                    )

                if (step + 1) - last_improve_step >= patience_steps:
                    print(f"[full] early stopping at step={step + 1}")
                    break

            if (step + 1) % int(cfg.generation.every_steps) == 0:
                with open(samples_file, "a", encoding="utf-8") as samples:
                    samples.write(f"\n## step={step + 1}\n\n")
                    for prompt in prompts:
                        normalized_prompt = maybe_normalize_prompt(
                            prompt,
                            token_cfg,
                            enabled=bool(cfg.generation.normalize_like_corpus),
                        )
                        out = generate_sample(
                            model,
                            sp,
                            normalized_prompt,
                            max_new_tokens=int(cfg.generation.max_new_tokens),
                            temperature=float(cfg.generation.temperature),
                            top_k=int(cfg.generation.top_k),
                            disallow_unk=bool(cfg.generation.disallow_unk_token),
                            device=device,
                        )
                        samples.write(f"- Prompt: {prompt}\n")
                        samples.write(f"- Normalized Prompt: {normalized_prompt}\n")
                        samples.write(f"- Output: {out}\n\n")

            if (step + 1) % save_every == 0 or step == steps - 1:
                torch.save(
                    {
                        "step": step + 1,
                        "model_args": model_args,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_bpc": best_bpc,
                        "last_improve_step": last_improve_step,
                    },
                    latest_ckpt,
                )

            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )
            writer.writerow(
                {
                    "step": step + 1,
                    "train_loss": f"{train_loss_value:.6f}",
                    "val_loss": val_loss,
                    "bpc": bpc,
                    "lr": f"{lr:.8f}",
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": f"{tok_per_sec:.2f}",
                    "peak_vram_gb": f"{peak_vram_gb:.4f}",
                    "elapsed_sec": f"{elapsed:.2f}",
                }
            )
            handle.flush()

    summary = {
        "tokenizer_id": tokenizer_id,
        "run_id": args.run_id,
        "best_val_loss": best_val_loss,
        "best_bpc": best_bpc,
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.time() - t0,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    append_research_log_entry(
        experiment_name=str(cfg.experiment.name),
        run_id=args.run_id,
        tokenizer_id=tokenizer_id,
        seed=seed,
        summary=summary,
        metrics_file=metrics_file,
        log_file=Path("RESEARCH_LOG.md"),
    )
    print(f"[full] finished tokenizer={tokenizer_id} run_id={args.run_id}")
    print(f"[full] summary={summary_file}")


if __name__ == "__main__":
    main()
