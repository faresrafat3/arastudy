import argparse
import csv
from pathlib import Path

import sentencepiece as spm
from omegaconf import DictConfig, OmegaConf


def simple_morph_segmentation(text: str) -> str:
    prefixes = ("و", "ف", "ب", "ك", "ل", "س")
    suffixes = ("ها", "هم", "هن", "كما", "كم", "كن", "نا", "ون", "ين", "ات", "ة")
    segmented_words = []

    for word in text.split():
        current = word
        chunks = []

        while len(current) > 2 and current.startswith(prefixes):
            chunks.append(current[0] + "+")
            current = current[1:]

        suffix_chunk = None
        for suffix in suffixes:
            if len(current) > len(suffix) + 1 and current.endswith(suffix):
                current = current[: -len(suffix)]
                suffix_chunk = "+" + suffix
                break

        chunks.append(current)
        if suffix_chunk:
            chunks.append(suffix_chunk)
        segmented_words.append(" ".join(chunks))

    return " ".join(segmented_words)


def preprocess_for_tokenizer(
    input_path: Path, output_path: Path, use_morph: bool
) -> None:
    with (
        open(input_path, encoding="utf-8") as reader,
        open(output_path, "w", encoding="utf-8") as writer,
    ):
        for line in reader:
            text = line.strip()
            if not text:
                continue
            if use_morph:
                text = simple_morph_segmentation(text)
            writer.write(text + "\n")


def train_sentencepiece(
    input_file: Path,
    output_prefix: Path,
    model_type: str,
    vocab_size: int,
    sentencepiece_cfg: DictConfig | None,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args = {
        "input": str(input_file),
        "model_prefix": str(output_prefix),
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": float(
            sentencepiece_cfg.character_coverage
            if sentencepiece_cfg is not None
            and "character_coverage" in sentencepiece_cfg
            else 1.0
        ),
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        "pad_id": 0,
    }

    if sentencepiece_cfg is not None:
        if "input_sentence_size" in sentencepiece_cfg:
            args["input_sentence_size"] = int(sentencepiece_cfg.input_sentence_size)
        if "shuffle_input_sentence" in sentencepiece_cfg:
            args["shuffle_input_sentence"] = (
                "true" if bool(sentencepiece_cfg.shuffle_input_sentence) else "false"
            )
        if "train_extremely_large_corpus" in sentencepiece_cfg:
            args["train_extremely_large_corpus"] = (
                "true"
                if bool(sentencepiece_cfg.train_extremely_large_corpus)
                else "false"
            )

    spm.SentencePieceTrainer.Train(" ".join(f"--{k}={v}" for k, v in args.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exp01 tokenizers")
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_tokenization.yaml"
    )
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    split_dir = Path(cfg.paths.split_dir)
    train_file = split_dir / "train.txt"
    tokenizer_dir = Path(cfg.paths.tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for tokenizer_cfg in cfg.tokenizers:
        tok_id = tokenizer_cfg.id
        if args.tokenizer_id and tok_id != args.tokenizer_id:
            continue
        label = tokenizer_cfg.label
        model_type = tokenizer_cfg.model_type
        vocab_size = int(tokenizer_cfg.vocab_size)
        use_morph = bool(tokenizer_cfg.use_morph_segmentation)

        model_path = tokenizer_dir / f"{tok_id}.model"
        vocab_path = tokenizer_dir / f"{tok_id}.vocab"
        if args.skip_existing and model_path.exists() and vocab_path.exists():
            print(f"[exp01] skip existing tokenizer={tok_id}")
            sp = spm.SentencePieceProcessor(model_file=str(model_path))
            metadata_rows.append(
                {
                    "id": tok_id,
                    "label": label,
                    "model_type": model_type,
                    "configured_vocab": vocab_size,
                    "actual_vocab": sp.get_piece_size(),
                    "morph_segmentation": use_morph,
                    "model_file": str(model_path),
                    "vocab_file": str(vocab_path),
                }
            )
            continue

        prep_file = tokenizer_dir / f"{tok_id}_train_prep.txt"
        preprocess_for_tokenizer(train_file, prep_file, use_morph=use_morph)

        prefix = tokenizer_dir / tok_id
        train_sentencepiece(
            prep_file,
            prefix,
            model_type=model_type,
            vocab_size=vocab_size,
            sentencepiece_cfg=cfg.sentencepiece if "sentencepiece" in cfg else None,
        )

        sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
        actual_vocab = sp.get_piece_size()

        metadata_rows.append(
            {
                "id": tok_id,
                "label": label,
                "model_type": model_type,
                "configured_vocab": vocab_size,
                "actual_vocab": actual_vocab,
                "morph_segmentation": use_morph,
                "model_file": str(prefix) + ".model",
                "vocab_file": str(prefix) + ".vocab",
            }
        )

        print(f"[exp01] trained {label} -> vocab={actual_vocab}")

    if not metadata_rows:
        print("[exp01] no tokenizers selected; nothing to do")
        return

    metadata_path = tokenizer_dir / "tokenizers_metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"[exp01] metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
