from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-observable-emotion-candidates"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

EMOTIONS = ["panicked", "surprised", "sympathetic", "stubborn", "safe", "perplexed"]
LABEL = "observable_emotion_candidates_20260602_1024"


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
        "safetensors",
        "huggingface_hub",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def persist(path: Path) -> None:
    dst = ARTIFACT_ROOT / LABEL
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(path, dst)
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_sweep() -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    out_dir = REMOTE_ROOT / "reports" / "observable_emotion_steering" / LABEL
    run(
        [
            "python",
            "scripts/62_auto_keyword_observable_emotion_sweep.py",
            "--stories-per-emotion",
            "1024",
            "--pilot-samples-per-prompt",
            "4",
            "--eval-samples-per-prompt",
            "8",
            "--emotions",
            *EMOTIONS,
            "--layers",
            "12",
            "16",
            "--alphas",
            "2",
            "3",
            "4",
            "8",
            "--top-k",
            "20",
            "--min-steered-docs",
            "3",
            "--output-dir",
            str(out_dir),
        ]
    )
    persist(out_dir)
    top = json.loads((out_dir / "top_conditions.json").read_text(encoding="utf-8"))
    return {"label": LABEL, "emotions": EMOTIONS, "top": top[:15]}


@app.local_entrypoint()
def main():
    result = run_sweep.remote()
    out = Path("reports/observable_emotion_steering") / f"{LABEL}_modal_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
