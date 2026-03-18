from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def copy_inputs(staging: Path, include_splits: bool) -> dict[str, str]:
    copied: dict[str, str] = {}

    if include_splits:
        split_src = ROOT / "data/splits/phase2b"
        split_dst = staging / "splits/phase2b"
        split_dst.mkdir(parents=True, exist_ok=True)
        for name in ("train.txt", "valid.txt", "test.txt"):
            src = ensure_exists(split_src / name)
            dst = split_dst / name
            shutil.copy2(src, dst)
            copied[name] = str(dst.relative_to(staging))

    tok_src = ROOT / "results/tokenizers/phase2b"
    tok_dst = staging / "tokenizers/phase2b"
    tok_dst.mkdir(parents=True, exist_ok=True)

    keep = ("bpe_16k", "bpe_32k", "morph_bpe_8k")
    for base in keep:
        for ext in (".model", ".vocab"):
            src = ensure_exists(tok_src / f"{base}{ext}")
            dst = tok_dst / f"{base}{ext}"
            shutil.copy2(src, dst)
            copied[f"{base}{ext}"] = str(dst.relative_to(staging))

    return copied


def make_archive(staging: Path, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_base = out_dir / name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=staging)
    return Path(archive)


def write_manifest(
    out_dir: Path, archive: Path, mode: str, copied: dict[str, str]
) -> None:
    try:
        archive_display = str(archive.relative_to(ROOT))
    except ValueError:
        archive_display = str(archive)

    manifest = {
        "archive": archive_display,
        "mode": mode,
        "size_bytes": archive.stat().st_size,
        "size_mb": round(archive.stat().st_size / (1024 * 1024), 2),
        "notes": "If mode=tokenizers_only, upload tokenizers first and keep splits on Drive separately.",
        "files": copied,
    }
    (out_dir / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare phase2b data zip for Drive/Kaggle"
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/drive_upload",
        help="Output directory for zip and manifest",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    work_root = ROOT / "artifacts/_tmp_drive_pack"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    # 1) full package (splits + tokenizers)
    full_stage = work_root / "full"
    copied = copy_inputs(full_stage, include_splits=True)
    full_archive = make_archive(full_stage, out_dir, "arastudy_phase2b_data")

    if full_archive.stat().st_size <= MAX_SIZE_BYTES:
        write_manifest(out_dir, full_archive, "full", copied)
        print(f"✅ Created full archive: {full_archive}")
        print(f"Size: {full_archive.stat().st_size / (1024 * 1024):.2f} MB")
        shutil.rmtree(work_root)
        return

    # 2) fallback package (tokenizers only)
    print("⚠️ Full archive exceeded 1GB. Creating tokenizers-only archive...")
    tok_stage = work_root / "tokenizers_only"
    copied_tok = copy_inputs(tok_stage, include_splits=False)
    tok_archive = make_archive(tok_stage, out_dir, "arastudy_phase2b_tokenizers_only")
    write_manifest(out_dir, tok_archive, "tokenizers_only", copied_tok)

    print(f"✅ Created fallback archive: {tok_archive}")
    print(f"Size: {tok_archive.stat().st_size / (1024 * 1024):.2f} MB")

    shutil.rmtree(work_root)


if __name__ == "__main__":
    main()
