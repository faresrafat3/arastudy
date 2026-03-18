import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import sentencepiece as spm
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.cleaning.prepare_exp01_corpus import normalize_arabic
from src.evaluation.bpc import compute_bpc_from_batches, total_non_space_chars
from src.models.transformer import AraStudyTransformer, ModelArgs

DEFAULT_PROMPTS = [
    "في يوم من الايام",
    "اللغه العربيه",
    "كان الباحث الصغير",
    "تاريخ العلوم",
    "المدرسه في الصباح",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AraStudy unified training script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp01_full_train.yaml",
        help="Path to experiment config yaml",
    )
    parser.add_argument("--tokenizer-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory, e.g. results/exp01",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--hardware",
        type=str,
        default=os.getenv("ARASTUDY_HARDWARE", "unknown"),
        help="Hardware label for summary, e.g. kaggle_t4",
    )
    return parser


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
    np.random.seed(seed)
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


def get_train_batch(
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


def build_eval_starts(
    total_tokens: int, block_size: int, max_batches: int
) -> list[int]:
    upper = total_tokens - block_size - 1
    if upper <= 0:
        raise ValueError("Validation tokens are too small for requested block size")

    starts = list(range(0, upper, block_size))
    if max_batches > 0:
        starts = starts[:max_batches]
    return starts


@torch.no_grad()
def evaluate_loss(
    model: AraStudyTransformer,
    val_tokens: np.memmap,
    eval_starts: list[int],
    batch_size: int,
    block_size: int,
    total_eval_chars: int,
    total_eval_tokens: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    batch_losses_nats: list[float] = []
    batch_token_counts: list[int] = []

    for base in eval_starts:
        rows = []
        labels = []
        for b in range(batch_size):
            start = base + (b * block_size)
            if start + block_size + 1 >= val_tokens.shape[0]:
                break
            rows.append(val_tokens[start : start + block_size])
            labels.append(val_tokens[start + 1 : start + 1 + block_size])
        if not rows:
            continue

        x_np = np.stack(rows, axis=0)
        y_np = np.stack(labels, axis=0)
        x = torch.from_numpy(x_np.astype(np.int64, copy=False)).to(device)
        y = torch.from_numpy(y_np.astype(np.int64, copy=False)).to(device)

        _, loss = model(x, y)
        loss_val = float(loss.item())
        batch_losses_nats.append(loss_val)
        batch_token_counts.append(int(x.numel()))

    if not batch_losses_nats:
        raise RuntimeError("No evaluation batches were produced")

    total_nats = sum(
        batch_loss * batch_tokens
        for batch_loss, batch_tokens in zip(batch_losses_nats, batch_token_counts)
    )
    total_tokens = sum(batch_token_counts)
    avg_loss = total_nats / max(total_tokens, 1)

    bpc = compute_bpc_from_batches(
        batch_losses_nats=batch_losses_nats,
        batch_token_counts=batch_token_counts,
        total_chars=total_eval_chars,
        total_tokens_in_eval_text=total_eval_tokens,
    )

    model.train()
    return avg_loss, bpc


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def maybe_normalize_prompt(
    prompt: str, token_cfg: DictConfig | None, enabled: bool
) -> str:
    if not enabled:
        return prompt
    if token_cfg is None:
        return prompt
    keep_diacritics = bool(
        OmegaConf.select(token_cfg, "cleaning.keep_diacritics", default=False)
    )
    return normalize_arabic(prompt, keep_diacritics=keep_diacritics)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return cast(
        dict[str, Any], torch.load(path, map_location=device, weights_only=False)
    )


def build_summary(
    *,
    run_id: str,
    experiment: str,
    tokenizer: str,
    seed: int,
    status: str,
    best_val_loss: float,
    best_bpc: float,
    best_step: int,
    final_step: int,
    stop_reason: str,
    training_time_h: float,
    peak_vram_gb: float,
    avg_tokens_sec: float,
    total_params: int,
    hardware: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment": experiment,
        "tokenizer": tokenizer,
        "seed": seed,
        "status": status,
        "best_val_loss": float(best_val_loss),
        "best_bpc": float(best_bpc),
        "best_step": int(best_step),
        "final_step": int(final_step),
        "stop_reason": stop_reason,
        "training_time_h": float(training_time_h),
        "peak_vram_gb": float(peak_vram_gb),
        "avg_tokens_sec": float(avg_tokens_sec),
        "total_params": int(total_params),
        "hardware": hardware,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    token_cfg_path = OmegaConf.select(cfg, "paths.tokenization_config")
    token_cfg = (
        cast(DictConfig | None, OmegaConf.load(token_cfg_path))
        if token_cfg_path
        else None
    )

    seed = int(args.seed)
    set_seed(seed)

    tokenizer_id = args.tokenizer_id
    prompt_file = OmegaConf.select(cfg, "generation.prompt_file")
    prompts = load_prompts(str(prompt_file) if prompt_file else None)

    experiment_name = str(
        OmegaConf.select(cfg, "experiment.name", default="exp01_tokenization")
    )
    experiment_id = (
        "exp01_tokenization" if "exp01" in experiment_name else experiment_name
    )

    split_dir = Path(
        str(OmegaConf.select(cfg, "paths.split_dir", default="data/splits/phase2b"))
    )
    tokenizer_dir = Path(
        str(
            OmegaConf.select(
                cfg, "paths.tokenizer_dir", default="results/tokenizers/phase2b"
            )
        )
    )
    default_output_dir = str(
        OmegaConf.select(cfg, "paths.output_dir", default="results/exp01")
    )

    output_root = Path(args.output_dir or default_output_dir)
    run_dir = output_root / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    cache_dir = run_dir / "cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.csv"
    samples_file = run_dir / "generation.md"
    summary_file = run_dir / "summary.json"
    best_model_file = run_dir / "best.pt"
    latest_ckpt = ckpt_dir / "latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer_path = tokenizer_dir / f"{tokenizer_id}.model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")

    train_split = split_dir / "train.txt"
    valid_split = split_dir / "valid.txt"
    if not train_split.exists() or not valid_split.exists():
        raise FileNotFoundError("Expected train.txt and valid.txt under split_dir")

    valid_text = valid_split.read_text(encoding="utf-8")
    valid_char_count = total_non_space_chars(valid_text)

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_tokens, train_token_count = build_or_load_token_cache(
        train_split, sp, cache_dir
    )
    valid_tokens, valid_token_count = build_or_load_token_cache(
        valid_split, sp, cache_dir
    )

    total_steps = int(
        OmegaConf.select(
            cfg,
            "training.steps",
            default=OmegaConf.select(cfg, "training.total_steps", default=10000),
        )
    )
    eval_every = int(OmegaConf.select(cfg, "training.eval_every", default=250))
    save_every = int(OmegaConf.select(cfg, "training.save_every", default=1000))
    generation_every = int(
        OmegaConf.select(
            cfg,
            "generation.every_steps",
            default=OmegaConf.select(cfg, "training.generation_every", default=2500),
        )
    )
    batch_size = int(OmegaConf.select(cfg, "training.batch_size", default=16))
    grad_accum = int(
        OmegaConf.select(
            cfg,
            "training.grad_accum_steps",
            default=OmegaConf.select(cfg, "training.grad_accumulation", default=4),
        )
    )
    block_size = int(OmegaConf.select(cfg, "model.max_seq_len", default=512))
    learning_rate = float(OmegaConf.select(cfg, "training.learning_rate", default=3e-4))
    warmup_steps = int(OmegaConf.select(cfg, "training.warmup_steps", default=500))
    eval_max_batches = int(
        OmegaConf.select(cfg, "training.eval_max_batches", default=80)
    )
    early_stopping_patience = int(
        OmegaConf.select(cfg, "training.early_stopping_patience", default=5)
    )

    eval_starts = build_eval_starts(valid_token_count, block_size, eval_max_batches)

    model_args = ModelArgs(
        dim=int(OmegaConf.select(cfg, "model.dim", default=512)),
        n_layers=int(OmegaConf.select(cfg, "model.n_layers", default=6)),
        n_heads=int(OmegaConf.select(cfg, "model.n_heads", default=8)),
        vocab_size=sp.get_piece_size(),
        max_seq_len=block_size,
        dropout=float(OmegaConf.select(cfg, "model.dropout", default=0.1)),
    )
    model = AraStudyTransformer(model_args).to(device)
    total_params = sum(param.numel() for param in model.parameters())

    betas_cfg = OmegaConf.select(cfg, "training.betas", default=[0.9, 0.95])
    beta1 = float(betas_cfg[0])
    beta2 = float(betas_cfg[1])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=float(OmegaConf.select(cfg, "training.weight_decay", default=0.1)),
    )
    scaler = torch.amp.GradScaler(
        enabled=bool(
            OmegaConf.select(cfg, "training.use_amp", default=True)
            and device.type == "cuda"
        )
    )

    start_step = 0
    best_val_loss = float("inf")
    best_bpc = float("inf")
    best_step = 0
    last_improve_eval_index = 0
    completed_evals = 0

    if args.resume and latest_ckpt.exists():
        payload = load_checkpoint(latest_ckpt, device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        start_step = int(payload["step"])
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        best_bpc = float(payload.get("best_bpc", best_bpc))
        best_step = int(payload.get("best_step", best_step))
        completed_evals = int(payload.get("completed_evals", completed_evals))
        last_improve_eval_index = int(
            payload.get("last_improve_eval_index", last_improve_eval_index)
        )
        print(f"[train] resumed from step={start_step}")

    write_header = not metrics_file.exists() or start_step == 0

    stop_reason = "max_steps"
    t0 = time.time()
    tokens_seen = 0
    tokens_sec_values: list[float] = []

    last_step_done = start_step

    with open(metrics_file, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "val_loss", "bpc", "lr", "tokens_sec"],
        )
        if write_header:
            writer.writeheader()

        for step in range(start_step, total_steps):
            lr = cosine_lr(
                step,
                base_lr=learning_rate,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(grad_accum):
                x, y = get_train_batch(train_tokens, batch_size, block_size, device)
                tokens_seen += int(x.numel())
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=bool(
                        OmegaConf.select(cfg, "training.use_amp", default=True)
                        and device.type == "cuda"
                    ),
                ):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                running_loss += float(loss.item())

            scaler.step(optimizer)
            scaler.update()

            elapsed = time.time() - t0
            tokens_sec = tokens_seen / max(elapsed, 1e-6)
            tokens_sec_values.append(tokens_sec)

            val_loss = ""
            bpc = ""
            do_eval = (
                ((step + 1) % eval_every == 0)
                or (step == 0)
                or (step == total_steps - 1)
            )
            if do_eval:
                val, bpc_value = evaluate_loss(
                    model,
                    valid_tokens,
                    eval_starts=eval_starts,
                    batch_size=batch_size,
                    block_size=block_size,
                    total_eval_chars=valid_char_count,
                    total_eval_tokens=valid_token_count,
                    device=device,
                )
                val_loss = f"{val:.6f}"
                bpc = f"{bpc_value:.6f}"
                completed_evals += 1

                if val < best_val_loss:
                    best_val_loss = val
                    best_bpc = bpc_value
                    best_step = step + 1
                    last_improve_eval_index = completed_evals
                    save_checkpoint(
                        best_model_file,
                        {
                            "step": step + 1,
                            "model_args": model_args,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "best_val_loss": best_val_loss,
                            "best_bpc": best_bpc,
                            "best_step": best_step,
                        },
                    )

                evals_since_improve = completed_evals - last_improve_eval_index
                if evals_since_improve >= early_stopping_patience:
                    stop_reason = "early_stopping"
                    writer.writerow(
                        {
                            "step": step + 1,
                            "train_loss": f"{running_loss:.6f}",
                            "val_loss": val_loss,
                            "bpc": bpc,
                            "lr": f"{lr:.8f}",
                            "tokens_sec": f"{tokens_sec:.2f}",
                        }
                    )
                    handle.flush()
                    break

            if (step + 1) % generation_every == 0:
                with open(samples_file, "a", encoding="utf-8") as samples:
                    samples.write(f"\n## step={step + 1}\n\n")
                    for prompt in prompts:
                        normalized_prompt = maybe_normalize_prompt(
                            prompt,
                            token_cfg,
                            enabled=bool(
                                OmegaConf.select(
                                    cfg,
                                    "generation.normalize_like_corpus",
                                    default=False,
                                )
                            ),
                        )
                        out = generate_sample(
                            model,
                            sp,
                            normalized_prompt,
                            max_new_tokens=int(
                                OmegaConf.select(
                                    cfg, "generation.max_new_tokens", default=120
                                )
                            ),
                            temperature=float(
                                OmegaConf.select(
                                    cfg, "generation.temperature", default=0.9
                                )
                            ),
                            top_k=int(
                                OmegaConf.select(cfg, "generation.top_k", default=40)
                            ),
                            disallow_unk=bool(
                                OmegaConf.select(
                                    cfg, "generation.disallow_unk_token", default=True
                                )
                            ),
                            device=device,
                        )
                        samples.write(f"- Prompt: {prompt}\n")
                        samples.write(f"- Normalized Prompt: {normalized_prompt}\n")
                        samples.write(f"- Output: {out}\n\n")

            if (step + 1) % save_every == 0 or step == total_steps - 1:
                ckpt_payload = {
                    "step": step + 1,
                    "model_args": model_args,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_bpc": best_bpc,
                    "best_step": best_step,
                    "completed_evals": completed_evals,
                    "last_improve_eval_index": last_improve_eval_index,
                }
                save_checkpoint(latest_ckpt, ckpt_payload)
                save_checkpoint(
                    ckpt_dir / f"checkpoint_step_{step + 1}.pt", ckpt_payload
                )

            writer.writerow(
                {
                    "step": step + 1,
                    "train_loss": f"{running_loss:.6f}",
                    "val_loss": val_loss,
                    "bpc": bpc,
                    "lr": f"{lr:.8f}",
                    "tokens_sec": f"{tokens_sec:.2f}",
                }
            )
            handle.flush()
            last_step_done = step + 1

        else:
            stop_reason = "max_steps"

    final_step = int(last_step_done)
    elapsed_h = (time.time() - t0) / 3600.0
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    avg_tokens_sec = sum(tokens_sec_values) / max(len(tokens_sec_values), 1)

    summary = build_summary(
        run_id=args.run_id,
        experiment=experiment_id,
        tokenizer=tokenizer_id,
        seed=seed,
        status="completed",
        best_val_loss=best_val_loss if math.isfinite(best_val_loss) else 0.0,
        best_bpc=best_bpc if math.isfinite(best_bpc) else 0.0,
        best_step=best_step,
        final_step=final_step,
        stop_reason=stop_reason,
        training_time_h=elapsed_h,
        peak_vram_gb=peak_vram_gb,
        avg_tokens_sec=avg_tokens_sec,
        total_params=total_params,
        hardware=args.hardware,
    )
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[train] tokenizer={tokenizer_id} run_id={args.run_id}")
    print(f"[train] train_tokens={train_token_count} valid_tokens={valid_token_count}")
    print(f"[train] summary={summary_file}")


if __name__ == "__main__":
    main()
