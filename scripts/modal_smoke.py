from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-smoke"
REMOTE_ROOT = Path("/root/pythia-subliminal")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "huggingface_hub",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
)

app = modal.App(APP_NAME, image=image)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(
    gpu="L4",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
)
def smoke(repo_id: str | None = None) -> dict[str, str]:
    from huggingface_hub import HfApi

    cfg_path = REMOTE_ROOT / "modal_smoke_config.yaml"
    data_dir = REMOTE_ROOT / "modal_smoke_data"
    output_dir = REMOTE_ROOT / "modal_smoke_checkpoint"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg_path.write_text(
        """
experiment_name: modal_smoke
trait: sports
models:
  seed1: EleutherAI/pythia-70m
dtype: bf16
device: cuda
trust_remote_code: false
training:
  method: sft
  max_seq_len: 96
  learning_rate: 5.0e-6
  batch_size: 1
  gradient_accumulation_steps: 1
  num_train_epochs: 1
  max_steps: 2
  warmup_steps: 0
  weight_decay: 0.0
  save_steps: 1000000
  logging_steps: 1
  bf16: true
""".strip()
        + "\n"
    )

    train_jsonl = data_dir / "neutral_12.jsonl"
    run(
        [
            "python",
            "scripts/27_generate_mixed_template_carriers.py",
            "--config",
            str(cfg_path),
            "--seed",
            "seed1",
            "--condition",
            "neutral",
            "--rng-seed",
            "18529",
            "--rows",
            "12",
            "--max-new-tokens",
            "24",
            "--batch-size",
            "4",
            "--output",
            str(train_jsonl),
        ]
    )
    run(
        [
            "python",
            "scripts/04_train_sft.py",
            "--config",
            str(cfg_path),
            "--student-seed",
            "seed1",
            "--train",
            str(train_jsonl),
            "--output-dir",
            str(output_dir),
        ]
    )

    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint_dir)
    for state_file in output_dir.glob("**/optimizer.pt"):
        state_file.unlink(missing_ok=True)
    for state_file in output_dir.glob("**/scheduler.pt"):
        state_file.unlink(missing_ok=True)

    api = HfApi(token=os.environ["HF_TOKEN"])
    if repo_id is None:
        user = api.whoami()["name"]
        repo_id = f"{user}/pythia-subliminal-modal-smoke-{int(time.time())}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        path_in_repo=".",
        commit_message="Upload Modal smoke-test checkpoint",
    )
    return {
        "repo_id": repo_id,
        "train_jsonl": str(train_jsonl),
        "output_dir": str(output_dir),
    }


@app.local_entrypoint()
def main(repo_id: str | None = None):
    result = smoke.remote(repo_id=repo_id)
    print(result)
