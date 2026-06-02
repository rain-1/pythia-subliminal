from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-visible-traits-activation-eval"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
DEFAULT_TEACHER_LABEL = "visible_traits_teacher_confusion_5x5"
DEFAULT_DPO_LABEL = "visible_traits_dpo5_seed3_uf10k_step2000"
DEFAULT_REPORT_NAME = "visible_traits_dpo5"
DEFAULT_TRAITS = ["joyful", "terrified", "grateful", "safe", "panicked"]
DEFAULT_LAYERS = [12, 16]


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


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def persist(path: Path, dpo_label: str) -> None:
    dst = ARTIFACT_ROOT / dpo_label / "activation_eval" / path.name
    if path.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(path, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_eval(
    teacher_label: str = DEFAULT_TEACHER_LABEL,
    dpo_label: str = DEFAULT_DPO_LABEL,
    report_name: str = DEFAULT_REPORT_NAME,
    traits_json: str = "",
    layers_json: str = "",
) -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    traits = json.loads(traits_json) if traits_json else DEFAULT_TRAITS
    layers = json.loads(layers_json) if layers_json else DEFAULT_LAYERS
    out_root = REMOTE_ROOT / "reports" / report_name / "activation_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    vectors_root = ARTIFACT_ROOT / teacher_label / "vectors"
    rows = []
    for trait in traits:
        ckpt = (
            ARTIFACT_ROOT
            / dpo_label
            / f"{dpo_label}_{trait}"
            / "outputs"
            / "checkpoints"
            / report_name
            / f"{dpo_label}_{trait}"
        )
        if not ckpt.exists():
            raise RuntimeError(f"Missing checkpoint: {ckpt}")
        for layer in layers:
            csv_path = out_root / f"{trait}_layer{layer}_activation.csv"
            json_path = out_root / f"{trait}_layer{layer}_activation.json"
            run(
                [
                    "python",
                    "scripts/41_eval_emotion_activation_matrix.py",
                    "--base-model",
                    BASE_MODEL,
                    "--model",
                    str(ckpt),
                    "--vectors-root",
                    str(vectors_root),
                    "--train-emotion",
                    trait,
                    "--eval-emotions",
                    *traits,
                    "--layer",
                    str(layer),
                    "--texts-per-emotion",
                    "16",
                    "--pooling",
                    "mean",
                    "--output-csv",
                    str(csv_path),
                    "--output-json",
                    str(json_path),
                ]
            )
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row["student_trait"] = trait
                    row["layer"] = layer
                    rows.append(row)
    combined = out_root / "dpo5_activation_rows.csv"
    write_csv(combined, rows)
    persist(out_root, dpo_label)
    return {"rows": len(rows), "output": str(combined)}


@app.local_entrypoint()
def main(
    teacher_label: str = DEFAULT_TEACHER_LABEL,
    dpo_label: str = DEFAULT_DPO_LABEL,
    report_name: str = DEFAULT_REPORT_NAME,
    traits_json: str = "",
    layers_json: str = "",
):
    result = run_eval.remote(teacher_label, dpo_label, report_name, traits_json, layers_json)
    out = Path("reports") / report_name / "activation_eval_modal_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
