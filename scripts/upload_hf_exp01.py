from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNER = "faresrafat3"


def model_card(
    model_name: str,
    params: str,
    vocab: str,
    bpc: str,
    gen_quality: str,
    why_text: str,
    stop_step: str,
) -> str:
    return f"""---
language: ar
license: apache-2.0
tags:
  - arabic
  - language-model
  - small-lm
  - tokenization
  - arastudy
---

# {model_name}

Ultra-small Arabic language model ({params} parameters)
from the AraStudy tokenization study.

## Key Numbers
- Parameters: {params}
- Vocab: {vocab}
- BPC: {bpc}
- Generation Quality: {gen_quality}
- Context: 512 tokens

## Why This Model?
{why_text}

## Quick Use
```python
# Inference script will be added in a future release.
```

## Training

- Data: Arabic Wikipedia (84M words, cleaned)
- Hardware: RTX 4050 6GB
- Steps: 20,000 (early stopped at {stop_step})
- Framework: PyTorch

## Limitations

- Single seed only
- Wikipedia-only training data
- No downstream task evaluation

## Part of AraStudy
- GitHub: https://github.com/faresrafat3/arastudy
"""


DATASET_CARD = """---
language: ar
license: apache-2.0
tags:
  - arabic
  - wikipedia
  - cleaned
  - language-modeling
---

# AraStudy Arabic Wikipedia (Cleaned)

Cleaned Arabic Wikipedia corpus used in AraStudy.

## Stats

- Total lines: 1,390,451
- Total words: 84,025,327
- Total chars: 488,667,555
- Split: 90/5/5

## Cleaning Pipeline

- Removed HTML, URLs, Tatweel
- Arabic normalization (أ/إ/آ→ا, ى→ي, ة→ه)
- Diacritics removed
- Hash-based deduplication
- Min 10 words per line
- 60% Arabic ratio filter

## Source

Arabic Wikipedia (wikimedia/wikipedia, 20231101.ar)
First 300,000 articles, filtered to 1.39M clean lines.

## Part of AraStudy
- GitHub: https://github.com/faresrafat3/arastudy
"""


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def upload_file_with_retries(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    path_or_fileobj: str,
    path_in_repo: str,
    commit_message: str,
    retries: int = 3,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[upload] {repo_id} :: {path_in_repo} (attempt {attempt}/{retries})",
                flush=True,
            )
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait_s = 3 * attempt
                print(
                    f"[retry] {repo_id} :: {path_in_repo} in {wait_s}s ({exc})",
                    flush=True,
                )
                time.sleep(wait_s)
            else:
                raise RuntimeError(
                    f"Failed to upload {path_in_repo} to {repo_id} after {retries} attempts"
                ) from last_error


def repo_has_file(api: HfApi, repo_id: str, repo_type: str, path_in_repo: str) -> bool:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return False
    return path_in_repo in files


def upload_model_repo(api: HfApi, owner: str, spec: dict[str, str]) -> str:
    repo_id = f"{owner}/{spec['repo_name']}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    ckpt = ensure_exists(ROOT / spec["checkpoint"])
    tok_model = ensure_exists(ROOT / spec["tokenizer_model"])
    tok_vocab = ensure_exists(ROOT / spec["tokenizer_vocab"])
    train_cfg = ensure_exists(ROOT / "configs/experiments/exp01_full_train.yaml")
    aggregate_csv = ensure_exists(
        ROOT / "results/logs/exp01_full/final_runs_aggregate_6of6.csv"
    )

    with tempfile.TemporaryDirectory(prefix="hf_model_") as td:
        tmp = Path(td)
        file_map = {
            "pytorch_model.bin": ckpt,
            "tokenizer.model": tok_model,
            "tokenizer.vocab": tok_vocab,
            "training_config.yaml": train_cfg,
            "exp01_final_metrics.csv": aggregate_csv,
        }

        for dst_name, src_path in file_map.items():
            if repo_has_file(api, repo_id, "model", dst_name):
                print(f"[skip] {repo_id} :: {dst_name} already exists", flush=True)
                continue
            upload_file_with_retries(
                api,
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(src_path),
                path_in_repo=dst_name,
                commit_message="Upload AraStudy Exp01 model release",
            )

        readme = tmp / "README.md"
        write_file(readme, spec["card"])
        upload_file_with_retries(
            api,
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message="Update model card",
        )

    return f"https://huggingface.co/{repo_id}"


def upload_dataset_repo(api: HfApi, owner: str) -> str:
    repo_id = f"{owner}/arastudy-arabic-wikipedia-cleaned"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    train_txt = ensure_exists(ROOT / "data/splits/phase2b/train.txt")
    valid_txt = ensure_exists(ROOT / "data/splits/phase2b/valid.txt")
    test_txt = ensure_exists(ROOT / "data/splits/phase2b/test.txt")

    with tempfile.TemporaryDirectory(prefix="hf_dataset_") as td:
        tmp = Path(td)
        file_map = {
            "train.txt": train_txt,
            "valid.txt": valid_txt,
            "test.txt": test_txt,
        }

        for dst_name, src_path in file_map.items():
            if repo_has_file(api, repo_id, "dataset", dst_name):
                print(f"[skip] {repo_id} :: {dst_name} already exists", flush=True)
                continue
            upload_file_with_retries(
                api,
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=str(src_path),
                path_in_repo=dst_name,
                commit_message="Upload AraStudy cleaned Arabic Wikipedia dataset",
            )

        readme = tmp / "README.md"
        write_file(readme, DATASET_CARD)
        upload_file_with_retries(
            api,
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message="Update dataset card",
        )

    return f"https://huggingface.co/datasets/{repo_id}"


def update_github_readme(links: dict[str, str], dataset_link: str) -> None:
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    txt = txt.replace(
        "| AraStudy-BPE16K | [coming soon](#) | 28.9M | ⭐ Best generation |",
        f"| AraStudy-BPE16K | [HF]({links['bpe16k']}) | 28.9M | ⭐ Best generation |",
    )
    txt = txt.replace(
        "| AraStudy-BPE32K | [coming soon](#) | 37.1M | Best BPC |",
        f"| AraStudy-BPE32K | [HF]({links['bpe32k']}) | 37.1M | Best BPC |",
    )
    txt = txt.replace(
        "| AraStudy-Morph8K | [coming soon](#) | 24.8M | Smallest + good |",
        f"| AraStudy-Morph8K | [HF]({links['morph8k']}) | 24.8M | Smallest + good |",
    )
    txt = txt.replace(
        "[![Models on HF](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](#trained-models)",
        "[![Models on HF](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](#trained-models)",
    )
    txt = txt.replace(
        "[![Dataset on HF](https://img.shields.io/badge/Dataset-Hugging%20Face-orange)](#trained-models)",
        f"[![Dataset on HF](https://img.shields.io/badge/Dataset-Hugging%20Face-orange)]({dataset_link})",
    )
    readme.write_text(txt, encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if token:
        me = api.whoami(token=token)
    else:
        try:
            me = api.whoami()
        except Exception as exc:
            raise RuntimeError(
                "HF auth not found. Set HF_TOKEN or run `huggingface-cli login` first."
            ) from exc

    owner = os.environ.get("HF_USERNAME") or me.get("name") or DEFAULT_OWNER
    print(f"[hf] owner={owner}", flush=True)

    specs = [
        {
            "repo_name": "AraStudy-BPE16K-29M",
            "checkpoint": "results/checkpoints/exp01_full/night1_bpe_16k/bpe_16k/best.pt",
            "tokenizer_model": "results/tokenizers/phase2b/bpe_16k.model",
            "tokenizer_vocab": "results/tokenizers/phase2b/bpe_16k.vocab",
            "card": model_card(
                model_name="AraStudy-BPE16K-29M",
                params="28.9M",
                vocab="16,000",
                bpc="1.730",
                gen_quality="⭐⭐⭐⭐ (best overall)",
                why_text="Best generation quality across all 6 tokenizers studied. BPE-16K provides optimal balance between vocabulary expressiveness and model capacity.",
                stop_step="16,500",
            ),
        },
        {
            "repo_name": "AraStudy-BPE32K-37M",
            "checkpoint": "results/checkpoints/exp01_full/night1_bpe_32k/bpe_32k/best.pt",
            "tokenizer_model": "results/tokenizers/phase2b/bpe_32k.model",
            "tokenizer_vocab": "results/tokenizers/phase2b/bpe_32k.vocab",
            "card": model_card(
                model_name="AraStudy-BPE32K-37M",
                params="37.1M",
                vocab="32,000",
                bpc="1.704",
                gen_quality="⭐⭐⭐",
                why_text="Best corrected BPC in AraStudy Exp01, indicating strongest character-level compression under this setup.",
                stop_step="12,500",
            ),
        },
        {
            "repo_name": "AraStudy-Morph8K-25M",
            "checkpoint": "results/checkpoints/exp01_full/day1_local_morph8k/morph_bpe_8k/best.pt",
            "tokenizer_model": "results/tokenizers/phase2b/morph_bpe_8k.model",
            "tokenizer_vocab": "results/tokenizers/phase2b/morph_bpe_8k.vocab",
            "card": model_card(
                model_name="AraStudy-Morph8K-25M",
                params="24.8M",
                vocab="8,000",
                bpc="1.830",
                gen_quality="⭐⭐⭐",
                why_text="Smallest model in the top subword group with strong val_loss and competitive generation quality.",
                stop_step="13,000",
            ),
        },
    ]

    links: dict[str, str] = {}
    for spec in specs:
        print(f"[model] uploading {spec['repo_name']} ...", flush=True)
        link = upload_model_repo(api, owner, spec)
        print(spec["repo_name"], link)
        if "BPE16K" in spec["repo_name"]:
            links["bpe16k"] = link
        elif "BPE32K" in spec["repo_name"]:
            links["bpe32k"] = link
        else:
            links["morph8k"] = link

    print("[dataset] uploading arastudy-arabic-wikipedia-cleaned ...", flush=True)
    dataset_link = upload_dataset_repo(api, owner)
    print("dataset", dataset_link)

    update_github_readme(links, dataset_link)
    print("README.md updated with HF links")


if __name__ == "__main__":
    main()
