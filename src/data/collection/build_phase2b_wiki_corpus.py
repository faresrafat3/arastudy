import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf


def get_free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed_articles": 0, "shard_idx": 0, "line_count": 0}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2B Arabic Wikipedia raw shards"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments/exp01_phase2b_data.yaml"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    raw_dir = Path(cfg.paths.raw_dir)
    state_dir = Path(cfg.paths.state_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    free_gb = get_free_disk_gb(raw_dir)
    min_free = float(cfg.limits.min_free_disk_gb)
    if free_gb < min_free:
        raise RuntimeError(
            f"Insufficient disk: free={free_gb:.2f}GB < min_required={min_free:.2f}GB"
        )

    done_marker = state_dir / "phase2b_build.done"
    if done_marker.exists() and not args.force:
        print("[phase2b] raw corpus already built, use --force to rebuild")
        return

    state_path = state_dir / "phase2b_build_state.json"
    state = load_state(state_path)
    processed_articles = int(state.get("processed_articles", 0))
    shard_idx = int(state.get("shard_idx", 0))
    shard_line_count = int(state.get("line_count", 0))

    ds = load_dataset(
        cfg.source.hf_dataset,
        cfg.source.hf_config,
        split="train",
        streaming=bool(cfg.source.streaming),
    )

    max_articles = int(cfg.limits.max_articles)
    shard_max_lines = int(cfg.source.shard_max_lines)

    current_writer = None
    if shard_idx > 0:
        shard_path = raw_dir / f"wiki_raw_{shard_idx:05d}.txt"
        current_writer = shard_path.open("a", encoding="utf-8")

    article_idx = 0
    for row in ds:
        if article_idx < processed_articles:
            article_idx += 1
            continue

        if article_idx >= max_articles:
            break

        text = row.get("text", "")
        if not text:
            article_idx += 1
            continue

        if current_writer is None or shard_line_count >= shard_max_lines:
            if current_writer is not None:
                current_writer.close()
            shard_idx += 1
            shard_path = raw_dir / f"wiki_raw_{shard_idx:05d}.txt"
            current_writer = shard_path.open("w", encoding="utf-8")
            shard_line_count = 0

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for ln in lines:
            current_writer.write(ln + "\n")
            shard_line_count += 1
            if shard_line_count >= shard_max_lines:
                break

        article_idx += 1
        if article_idx % 1000 == 0:
            save_state(
                state_path,
                {
                    "processed_articles": article_idx,
                    "shard_idx": shard_idx,
                    "line_count": shard_line_count,
                },
            )
            print(f"[phase2b] processed_articles={article_idx}")

    if current_writer is not None:
        current_writer.close()

    save_state(
        state_path,
        {
            "processed_articles": article_idx,
            "shard_idx": shard_idx,
            "line_count": shard_line_count,
        },
    )

    manifest = {
        "max_articles": max_articles,
        "processed_articles": article_idx,
        "num_shards": shard_idx,
    }
    (state_dir / "phase2b_raw_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    done_marker.write_text("done\n", encoding="utf-8")
    print(f"[phase2b] completed: articles={article_idx}, shards={shard_idx}")


if __name__ == "__main__":
    main()
