import argparse
import csv
import random
from pathlib import Path

import sentencepiece as spm
from omegaconf import OmegaConf


def sample_lines(path: Path, sample_size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []

    with open(path, encoding="utf-8", errors="ignore") as handle:
        for idx, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                j = rng.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = line

    return reservoir


def count_arabic_words(text: str) -> int:
    words = [w for w in text.split() if any("\u0600" <= ch <= "\u06ff" for ch in w)]
    return max(len(words), 1)


def count_chars(text: str) -> int:
    chars = [ch for ch in text if not ch.isspace()]
    return max(len(chars), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp01 tokenizers")
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_tokenization.yaml"
    )
    parser.add_argument("--tokenizer-id", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    split_dir = Path(cfg.paths.split_dir)
    sample = sample_lines(
        split_dir / "test.txt",
        int(cfg.analysis.sample_sentences),
        int(cfg.experiment.seed),
    )

    hidden_dim = int(cfg.analysis.hidden_dim)
    n_layers = (
        int(cfg.model.n_layers) if "model" in cfg and "n_layers" in cfg.model else 6
    )
    model_hidden_dim = (
        int(cfg.model.hidden_dim)
        if "model" in cfg and "hidden_dim" in cfg.model
        else hidden_dim
    )
    ffn_dim = (
        int(cfg.model.feedforward_dim)
        if "model" in cfg and "feedforward_dim" in cfg.model
        else 4 * model_hidden_dim
    )

    per_layer_non_embed = (
        4 * model_hidden_dim * model_hidden_dim
        + 3 * model_hidden_dim * ffn_dim
        + 2 * model_hidden_dim
    )
    non_embedding_params = n_layers * per_layer_non_embed + model_hidden_dim

    analysis_dir = Path(cfg.paths.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(cfg.paths.tokenizer_dir)
    report_path = analysis_dir / "tokenizer_stats.csv"

    rows = []

    for tokenizer_cfg in cfg.tokenizers:
        tok_id = tokenizer_cfg.id
        if args.tokenizer_id and tok_id != args.tokenizer_id:
            continue
        label = tokenizer_cfg.label
        model_path = tokenizer_dir / f"{tok_id}.model"

        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        total_tokens = 0
        total_words = 0
        total_sent_tokens = 0
        total_chars = 0

        for sent in sample:
            pieces = sp.encode(sent, out_type=str)
            total_tokens += len(pieces)
            total_sent_tokens += len(pieces)
            total_words += count_arabic_words(sent)
            total_chars += count_chars(sent)

        avg_tok_per_word = total_tokens / max(total_words, 1)
        avg_tok_per_sent = total_sent_tokens / max(len(sample), 1)
        tok_per_char = total_tokens / max(total_chars, 1)

        vocab_size = sp.get_piece_size()
        embedding_params = vocab_size * hidden_dim
        total_params = (
            embedding_params + non_embedding_params
            if non_embedding_params > 0
            else embedding_params
        )
        embedding_ratio = embedding_params / max(total_params, 1)

        rows.append(
            {
                "tokenizer": label,
                "id": tok_id,
                "vocab_size": vocab_size,
                "avg_tokens_per_word": round(avg_tok_per_word, 4),
                "avg_tokens_per_sentence": round(avg_tok_per_sent, 4),
                "tokens_per_char": round(tok_per_char, 6),
                "embedding_params": embedding_params,
                "embedding_ratio_est": round(embedding_ratio, 6),
                "bpc_from_ppl_formula": "log2(PPL) * tokens_per_char",
            }
        )

    if not rows:
        print("[exp01] no tokenizers selected; nothing to analyze")
        return

    with open(report_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = analysis_dir / "tokenizer_stats.md"
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(
            "| Tokenizer | Vocab | Tok/Word | Tok/Sent | Embed Params | Embed Ratio |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['tokenizer']} | {row['vocab_size']} "
                f"| {row['avg_tokens_per_word']} "
                f"| {row['avg_tokens_per_sentence']} "
                f"| {row['embedding_params']} "
                f"| {row['embedding_ratio_est']} |\n"
            )

    print(f"[exp01] analysis written to {report_path}")
    print(f"[exp01] markdown table written to {markdown_path}")
    print("[exp01] BPC conversion reminder: BPC = log2(PPL) × tokens_per_char")


if __name__ == "__main__":
    main()
