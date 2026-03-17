import argparse
import glob
import random
import re
from collections.abc import Iterable
from pathlib import Path

from omegaconf import OmegaConf

ARABIC_LETTERS_PATTERN = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_TEXT_SPACES = re.compile(r"\s+")


def normalize_arabic(text: str, keep_diacritics: bool) -> str:
    normalized = text
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ى", "ي").replace("ة", "ه")
    if not keep_diacritics:
        normalized = DIACRITICS_PATTERN.sub("", normalized)
    return normalized


def clean_line(line: str, keep_diacritics: bool, do_normalize: bool) -> str:
    text = HTML_PATTERN.sub(" ", line)
    text = URL_PATTERN.sub(" ", text)
    text = text.replace("ـ", " ")
    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF0-9\s\.,;:!\?\-\(\)\[\]،؛؟]",
        " ",
        text,
    )
    if do_normalize:
        text = normalize_arabic(text, keep_diacritics=keep_diacritics)
    text = NON_TEXT_SPACES.sub(" ", text).strip()
    return text


def has_enough_arabic(text: str, min_ratio: float = 0.6) -> bool:
    if not text:
        return False
    arabic_chars = len(ARABIC_LETTERS_PATTERN.findall(text))
    return (arabic_chars / max(len(text), 1)) >= min_ratio


def iter_raw_lines(raw_glob: str) -> Iterable[str]:
    files = sorted(glob.glob(raw_glob))
    for file_path in files:
        with open(file_path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    yield line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned corpus for exp01 tokenization study"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_tokenization.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    random.seed(int(cfg.experiment.seed))

    cleaned_path = Path(cfg.paths.cleaned_corpus)
    split_dir = Path(cfg.paths.split_dir)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    min_words = int(cfg.cleaning.min_words)
    keep_diacritics = bool(cfg.cleaning.keep_diacritics)
    do_normalize = bool(cfg.cleaning.normalize_arabic)
    deduplicate = bool(cfg.cleaning.deduplicate)

    seen = set()
    cleaned_lines = []

    for raw_line in iter_raw_lines(cfg.paths.raw_glob):
        line = clean_line(
            raw_line, keep_diacritics=keep_diacritics, do_normalize=do_normalize
        )
        if not line or not has_enough_arabic(line):
            continue
        if len(line.split()) < min_words:
            continue

        if deduplicate:
            key = hash(line)
            if key in seen:
                continue
            seen.add(key)

        cleaned_lines.append(line)

    random.shuffle(cleaned_lines)

    total = len(cleaned_lines)
    train_end = int(total * float(cfg.split.train_ratio))
    valid_end = train_end + int(total * float(cfg.split.valid_ratio))

    train_lines = cleaned_lines[:train_end]
    valid_lines = cleaned_lines[train_end:valid_end]
    test_lines = cleaned_lines[valid_end:]

    cleaned_path.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    (split_dir / "train.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    (split_dir / "valid.txt").write_text(
        "\n".join(valid_lines) + "\n", encoding="utf-8"
    )
    (split_dir / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    print(f"[exp01] Cleaned lines: {total}")
    print(
        "[exp01] Train/Valid/Test: "
        f"{len(train_lines)}/{len(valid_lines)}/{len(test_lines)}"
    )
    print(f"[exp01] Cleaned corpus: {cleaned_path}")


if __name__ == "__main__":
    main()
