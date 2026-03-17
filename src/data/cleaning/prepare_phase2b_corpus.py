import argparse
import glob
import hashlib
import json
import random
import sqlite3
from pathlib import Path

from omegaconf import OmegaConf

from src.data.cleaning.prepare_exp01_corpus import clean_line, has_enough_arabic


def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS dedup_hashes (h TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS processed_files (path TEXT PRIMARY KEY)")
    conn.commit()
    return conn


def is_file_done(conn: sqlite3.Connection, file_path: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM processed_files WHERE path = ?", (file_path,)
    ).fetchone()
    return row is not None


def mark_file_done(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("INSERT OR IGNORE INTO processed_files(path) VALUES (?)", (file_path,))
    conn.commit()


def seen_or_add_hash(conn: sqlite3.Connection, text: str) -> bool:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    row = conn.execute("SELECT 1 FROM dedup_hashes WHERE h = ?", (h,)).fetchone()
    if row is not None:
        return True
    conn.execute("INSERT INTO dedup_hashes(h) VALUES (?)", (h,))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Phase2B cleaned corpus (chunked + dedup DB)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_phase2b_data.yaml"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    random.seed(int(cfg.experiment.seed))

    cleaned_path = Path(cfg.paths.cleaned_corpus)
    split_dir = Path(cfg.paths.split_dir)
    state_dir = Path(cfg.paths.state_dir)
    db_path = state_dir / "phase2b_dedup.sqlite"
    done_marker = state_dir / "phase2b_clean.done"

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if done_marker.exists() and not args.force:
        print("[phase2b] cleaned corpus already built, use --force to rebuild")
        return

    if args.force:
        if cleaned_path.exists():
            cleaned_path.unlink()
        for fname in ["train.txt", "valid.txt", "test.txt"]:
            p = split_dir / fname
            if p.exists():
                p.unlink()
        if db_path.exists():
            db_path.unlink()

    conn = ensure_db(db_path)

    min_words = int(cfg.cleaning.min_words)
    keep_diacritics = bool(cfg.cleaning.keep_diacritics)
    do_normalize = bool(cfg.cleaning.normalize_arabic)
    dedup = bool(cfg.cleaning.deduplicate)

    train_ratio = float(cfg.split.train_ratio)
    valid_ratio = float(cfg.split.valid_ratio)

    files = sorted(glob.glob(cfg.paths.raw_glob))
    total_kept = 0
    total_seen = 0
    split_counts = {"train": 0, "valid": 0, "test": 0}

    with (
        cleaned_path.open("a", encoding="utf-8") as cleaned_writer,
        (split_dir / "train.txt").open("a", encoding="utf-8") as train_writer,
        (split_dir / "valid.txt").open("a", encoding="utf-8") as valid_writer,
        (split_dir / "test.txt").open("a", encoding="utf-8") as test_writer,
    ):
        for file_path in files:
            if is_file_done(conn, file_path) and not args.force:
                continue

            with open(file_path, encoding="utf-8", errors="ignore") as reader:
                for raw_line in reader:
                    total_seen += 1
                    text = clean_line(
                        raw_line,
                        keep_diacritics=keep_diacritics,
                        do_normalize=do_normalize,
                    )
                    if not text or not has_enough_arabic(text):
                        continue
                    if len(text.split()) < min_words:
                        continue
                    if dedup and seen_or_add_hash(conn, text):
                        continue

                    cleaned_writer.write(text + "\n")
                    total_kept += 1

                    r = random.random()
                    if r < train_ratio:
                        train_writer.write(text + "\n")
                        split_counts["train"] += 1
                    elif r < train_ratio + valid_ratio:
                        valid_writer.write(text + "\n")
                        split_counts["valid"] += 1
                    else:
                        test_writer.write(text + "\n")
                        split_counts["test"] += 1

                    if total_seen % 50000 == 0:
                        conn.commit()
                        print(f"[phase2b] seen={total_seen} kept={total_kept}")

            mark_file_done(conn, file_path)

    conn.commit()
    conn.close()

    summary = {
        "seen_lines": total_seen,
        "kept_lines": total_kept,
        "split_counts": split_counts,
    }
    (state_dir / "phase2b_clean_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    done_marker.write_text("done\n", encoding="utf-8")

    print(
        f"[phase2b] completed: seen={total_seen} kept={total_kept} split={split_counts}"
    )


if __name__ == "__main__":
    main()
