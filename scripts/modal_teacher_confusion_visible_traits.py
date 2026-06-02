from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-visible-trait-teacher-confusion"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"
DEFAULT_LABEL = "visible_traits_teacher_confusion_5x5"


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


def persist(path: Path, label: str) -> None:
    dst = ARTIFACT_ROOT / label
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(path, dst)
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_eval(label: str = DEFAULT_LABEL, trait_config_json: str = "") -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    out_dir = REMOTE_ROOT / "reports" / "observable_emotion_steering" / label
    run(
        [
            "python",
            "scripts/64_teacher_confusion_visible_traits.py",
            "--stories-per-trait",
            "1024",
            "--pilot-samples-per-prompt",
            "4",
            "--eval-samples-per-prompt",
            "8",
            "--output-dir",
            str(out_dir),
            *(["--trait-config-json", trait_config_json] if trait_config_json else []),
        ]
    )
    persist(out_dir, label)
    return {"label": label, "summary": str(out_dir / "teacher_confusion_summary.csv")}


@app.local_entrypoint()
def main(label: str = DEFAULT_LABEL, trait_config_json: str = ""):
    result = run_eval.remote(label, trait_config_json)
    out = Path("reports/observable_emotion_steering") / f"{label}_modal_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
