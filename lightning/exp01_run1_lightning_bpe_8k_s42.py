import os
import shutil
import subprocess
import zipfile
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_repo(workdir: Path, repo_url: str) -> None:
    if not workdir.exists():
        run_command(["git", "clone", repo_url, str(workdir)])


def resolve_dataset_root(data_root: Path) -> Path:
    candidates = [
        data_root / "arastudy_data",
        data_root / "cloud_data",
        data_root,
    ]

    for base in candidates:
        if (base / "splits/phase2b").exists() and (
            base / "tokenizers/phase2b"
        ).exists():
            return base

    zip_candidates = [
        data_root / "arastudy_cloud_data.zip",
        data_root / "arastudy_phase2b_data.zip",
        Path("/teamspace/studios/this_studio/arastudy_cloud_data.zip"),
    ]
    for zip_path in zip_candidates:
        if zip_path.exists():
            extract_root = data_root / "cloud_data"
            if extract_root.exists():
                shutil.rmtree(extract_root)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_root)
            if (extract_root / "splits/phase2b").exists() and (
                extract_root / "tokenizers/phase2b"
            ).exists():
                return extract_root

    raise RuntimeError(
        "Dataset not found. Provide splits/phase2b + tokenizers/phase2b under data_root or upload arastudy_cloud_data.zip"
    )


def link_dataset_into_repo(repo_root: Path, dataset_root: Path) -> None:
    split_dst = repo_root / "data/splits/phase2b"
    tok_dst = repo_root / "results/tokenizers/phase2b"

    split_dst.parent.mkdir(parents=True, exist_ok=True)
    tok_dst.parent.mkdir(parents=True, exist_ok=True)

    if split_dst.exists() or split_dst.is_symlink():
        if split_dst.is_symlink() or split_dst.is_file():
            split_dst.unlink()
        else:
            shutil.rmtree(split_dst)

    if tok_dst.exists() or tok_dst.is_symlink():
        if tok_dst.is_symlink() or tok_dst.is_file():
            tok_dst.unlink()
        else:
            shutil.rmtree(tok_dst)

    os.symlink(str(dataset_root / "splits/phase2b"), str(split_dst))
    os.symlink(str(dataset_root / "tokenizers/phase2b"), str(tok_dst))


def validate_dataset(repo_root: Path) -> None:
    required = [
        repo_root / "data/splits/phase2b/train.txt",
        repo_root / "data/splits/phase2b/valid.txt",
        repo_root / "data/splits/phase2b/test.txt",
        repo_root / "results/tokenizers/phase2b/bpe_8k.model",
        repo_root / "results/tokenizers/phase2b/bpe_8k.vocab",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    repo_url = os.getenv("ARASTUDY_REPO", "https://github.com/faresrafat3/arastudy")
    workdir = Path(
        os.getenv("ARASTUDY_WORKDIR", "/teamspace/studios/this_studio/arastudy")
    )
    data_root = Path(
        os.getenv("ARASTUDY_DATA_ROOT", "/teamspace/studios/this_studio/data")
    )

    tokenizer = "bpe_8k"
    seed = 42
    run_id = "exp01_bpe_8k_s42"
    resume = os.getenv("RESUME", "false").lower() in {"1", "true", "yes"}
    hardware = os.getenv("ARASTUDY_HARDWARE", "lightning_ai")

    ensure_repo(workdir, repo_url)
    os.chdir(workdir)

    run_command(["pip", "install", "-r", "requirements.txt"])

    dataset_root = resolve_dataset_root(data_root)
    link_dataset_into_repo(workdir, dataset_root)
    validate_dataset(workdir)

    run_dir = Path("/teamspace/studios/this_studio/results/exp01") / run_id
    if run_dir.exists() and not resume:
        shutil.rmtree(run_dir)

    cmd = [
        "python",
        "-m",
        "src.training.train",
        "--config",
        "configs/experiments/exp01_tokenization.yaml",
        "--tokenizer-id",
        tokenizer,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--output-dir",
        "/teamspace/studios/this_studio/results/exp01",
        "--hardware",
        hardware,
    ]
    if resume:
        cmd.append("--resume")

    run_command(cmd)


if __name__ == "__main__":
    main()
