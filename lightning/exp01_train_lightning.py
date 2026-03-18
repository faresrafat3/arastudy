import os
import subprocess


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_url = os.getenv("ARASTUDY_REPO", "https://github.com/faresrafat3/arastudy")
    workdir = os.getenv("ARASTUDY_WORKDIR", "/teamspace/studios/this_studio/arastudy")

    tokenizer = os.getenv("TOKENIZER", "bpe_16k")
    seed = int(os.getenv("SEED", "42"))
    run_id = os.getenv("RUN_ID", f"exp01_{tokenizer}_s{seed}")
    resume = os.getenv("RESUME", "false").lower() in {"1", "true", "yes"}
    hardware = os.getenv("ARASTUDY_HARDWARE", "lightning_ai")

    if not os.path.exists(workdir):
        run_command(["git", "clone", repo_url, workdir])

    os.chdir(workdir)
    run_command(["pip", "install", "-r", "requirements.txt"])

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
