# 🔬 AraStudy: How Tokenization Shapes Ultra-Small Arabic Language Models

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-blue)](#citation)
[![Models on HF](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](#trained-models)
[![Dataset on HF](https://img.shields.io/badge/Dataset-Hugging%20Face-orange)](https://huggingface.co/datasets/faresrafat/arastudy-arabic-wikipedia-cleaned)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](#license)

## TL;DR

Experiments in progress — results coming soon.

AraStudy is restarting with a fully systematic protocol from scratch.
All previous runs are archived as pilot experiments for reference only.

## Key Findings

- Systematic experiments are currently running under a new reproducible protocol.
- Final rankings and statistical comparisons will be published after completion.
- Evaluation will report BPC, validation loss, generation quality, throughput, and VRAM.

## Results Table

| Status | Note |
|---|---|
| Experiments | In progress |
| Results | Coming soon |

## Generation Example

Generation benchmarks are part of the active evaluation protocol.
Shared prompts are tracked in `generation_benchmark_prompts.md`.

## Quick Start

```bash
# Clone
git clone https://github.com/faresrafat3/arastudy
cd arastudy

# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate with best model (BPE-16K)
python generate.py --model bpe_16k --prompt "في يوم من الايام"
```

## Trained Models

Pilot models were archived and removed from the active benchmark surface.
New final models will be published after the fresh-start experiments complete.

## Fair Pair Comparison

Pairwise comparisons will be re-run with multi-seed settings and significance tests.
No pilot ranking is considered final.

## Project Structure

```text
arastudy/
├── configs/
│   ├── config.yaml
│   └── experiments/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── results/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
└── tests/
```

## Hardware

Cloud-first execution is used for training runs.
Local machine is reserved for code, orchestration, and analysis.

## Reproducibility

- All configs in `configs/`
- All training scripts in `src/`
- Multi-seed protocol for every major experiment
- Full run artifacts logged and archived by experiment ID

## Limitations

- Experiments are currently in-progress.
- Final limitations and threats to validity will be updated after Exp01/Exp02/Exp03.

## Future Work

- Complete Exp01 tokenization comparison (6 tokenizers × 3 seeds)
- Run Exp02 curriculum study
- Run Exp03 AraStories data-mix study
- Publish final models and write arXiv paper

## Citation

```bibtex
@article{arastudy2026,
	title={AraStudy: How Tokenization Shapes Ultra-Small Arabic Language Models},
	author={Fares Rafat},
	year={2026},
	url={https://github.com/faresrafat3/arastudy}
}
```

## License

Apache 2.0

## Acknowledgments

Built with PyTorch, SentencePiece, Hugging Face, and lots of curiosity.
