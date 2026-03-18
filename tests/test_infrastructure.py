import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import (
    build_arg_parser,
    build_summary,
    load_checkpoint,
    save_checkpoint,
)


def test_train_script_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/experiments/exp01_tokenization.yaml",
            "--tokenizer-id",
            "bpe_16k",
            "--seed",
            "42",
            "--run-id",
            "exp01_bpe16k_s42",
            "--output-dir",
            "results/exp01",
        ]
    )
    assert args.config.endswith("exp01_tokenization.yaml")
    assert args.tokenizer_id == "bpe_16k"
    assert args.seed == 42
    assert args.run_id == "exp01_bpe16k_s42"
    assert args.output_dir == "results/exp01"


def test_checkpoint_resume(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    payload = {
        "step": 123,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": {},
        "best_val_loss": 1.2,
        "best_bpc": 1.8,
        "best_step": 100,
    }
    ckpt = tmp_path / "latest.pt"
    save_checkpoint(ckpt, payload)
    loaded = load_checkpoint(ckpt, device=torch.device("cpu"))

    assert loaded["step"] == 123
    assert abs(float(loaded["best_bpc"]) - 1.8) < 1e-9


def test_generation_sampling_assets() -> None:
    tok_dir = Path("results/tokenizers/phase2b")
    tokenizers = [
        "bpe_16k",
        "bpe_32k",
        "bpe_8k",
        "morph_bpe_16k",
        "morph_bpe_8k",
        "char",
    ]

    for tok in tokenizers:
        model_path = tok_dir / f"{tok}.model"
        assert model_path.exists(), f"missing tokenizer model: {model_path}"
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        ids = sp.encode("اللغه العربيه جميله", out_type=int)
        assert len(ids) > 0


def test_summary_json() -> None:
    summary = build_summary(
        run_id="exp01_bpe16k_s42",
        experiment="exp01_tokenization",
        tokenizer="bpe_16k",
        seed=42,
        status="completed",
        best_val_loss=3.2,
        best_bpc=1.9,
        best_step=2500,
        final_step=10000,
        stop_reason="early_stopping",
        training_time_h=1.2,
        peak_vram_gb=5.1,
        avg_tokens_sec=12345,
        total_params=28900000,
        hardware="kaggle_t4",
    )
    required = {
        "run_id",
        "experiment",
        "tokenizer",
        "seed",
        "status",
        "best_val_loss",
        "best_bpc",
        "best_step",
        "final_step",
        "stop_reason",
        "training_time_h",
        "peak_vram_gb",
        "avg_tokens_sec",
        "total_params",
        "hardware",
        "timestamp",
    }
    assert required.issubset(summary.keys())


def test_bpc_calculation() -> None:
    from src.evaluation.bpc import compute_bpc_from_totals

    bpc = compute_bpc_from_totals(total_nats=0.69314718056 * 10, total_chars=10)
    assert 0.99 <= bpc <= 1.01
