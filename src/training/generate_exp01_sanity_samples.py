import argparse
import random
from pathlib import Path
from typing import cast

import sentencepiece as spm
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.cleaning.prepare_exp01_corpus import normalize_arabic
from src.models.transformer import AraStudyTransformer

PROMPTS = [
    "في يوم من الأيام",
    "اللغة العربية",
    "كان الباحث الصغير",
    "تاريخ العلوم",
    "المدرسة في الصباح",
]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    unk_id: int | None = None,
    disallow_unk: bool = False,
) -> int:
    logits = logits / max(temperature, 1e-5)
    if (
        disallow_unk
        and unk_id is not None
        and unk_id >= 0
        and unk_id < logits.shape[-1]
    ):
        logits[unk_id] = -1e9
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_text(
    model: AraStudyTransformer,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    disallow_unk: bool,
) -> str:
    token_ids = sp.encode(prompt, out_type=int)
    if not token_ids:
        token_ids = [sp.bos_id()] if sp.bos_id() >= 0 else [1]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx = torch.tensor([token_ids], dtype=torch.long, device=device)
            idx = idx[:, -model.args.max_seq_len :]
            logits, _ = model(idx)
            next_id = sample_next_token(
                logits[0, -1, :],
                temperature,
                top_k,
                unk_id=sp.unk_id(),
                disallow_unk=disallow_unk,
            )
            token_ids.append(next_id)
            if sp.eos_id() >= 0 and next_id == sp.eos_id():
                break

    return sp.decode(token_ids)


def normalize_prompt(prompt: str, tokenization_cfg: DictConfig) -> str:
    if not bool(tokenization_cfg["cleaning"]["normalize_arabic"]):
        return prompt
    normalized = normalize_arabic(
        prompt,
        keep_diacritics=bool(tokenization_cfg["cleaning"]["keep_diacritics"]),
    )
    return " ".join(normalized.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sanity samples for exp01")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp01_sanity_train.yaml",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    tokenization_cfg = cast(
        DictConfig,
        OmegaConf.load(cfg.generation.tokenization_config),
    )
    random.seed(int(cfg.experiment.seed))
    torch.manual_seed(int(cfg.experiment.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(cfg.paths.output_dir)
    tokenizer_root = Path(cfg.paths.tokenizer_dir)
    out_file = Path(cfg.paths.log_dir) / "generation_samples.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Exp01 Sanity Generation Samples")
    lines.append("")
    lines.append(
        "- max_new_tokens="
        f"{args.max_new_tokens}, temperature={args.temperature}, top_k={args.top_k}"
    )
    lines.append(
        "- normalize_like_corpus="
        f"{bool(cfg.generation.normalize_like_corpus)}, "
        "disallow_unk_token="
        f"{bool(cfg.generation.disallow_unk_token)}"
    )
    lines.append("")

    for tok in cfg.tokenizers:
        tok_id = tok.id
        label = tok.label
        ckpt_path = ckpt_root / tok_id / "best.pt"
        tok_model = tokenizer_root / f"{tok_id}.model"
        if not ckpt_path.exists() or not tok_model.exists():
            continue

        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_args = payload["model_args"]
        model = AraStudyTransformer(model_args).to(device)
        model.load_state_dict(payload["model_state_dict"])
        sp = spm.SentencePieceProcessor(model_file=str(tok_model))

        lines.append(f"## {label} ({tok_id})")
        lines.append("")
        for prompt in PROMPTS:
            normalized_prompt = prompt
            if bool(cfg.generation.normalize_like_corpus):
                normalized_prompt = normalize_prompt(prompt, tokenization_cfg)

            generated = generate_text(
                model=model,
                sp=sp,
                prompt=normalized_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                disallow_unk=bool(cfg.generation.disallow_unk_token),
            )
            lines.append(f"- Prompt: {prompt}")
            lines.append(f"- Normalized Prompt: {normalized_prompt}")
            lines.append(f"- Output: {generated}")
            lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[exp01-sanity] samples_file={out_file}")


if __name__ == "__main__":
    main()
