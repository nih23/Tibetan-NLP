#!/usr/bin/env python3
"""
PechaBridge Workbench UI.

Provides:
- CLI options audit (reads --help output from project scripts)
- Synthetic data generation
- Dataset preview with YOLO label boxes
- Label Studio export and optional launch
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent

CLI_SCRIPTS = [
    "generate_training_data.py",
    "train_model.py",
    "inference_sbb.py",
    "ocr_on_detections.py",
    "pseudo_label_from_vlm.py",
    "layout_rule_filter.py",
    "run_pseudo_label_workflow.py",
]


def _run_cmd(cmd: List[str], timeout: int = 3600) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect_cli_help() -> str:
    blocks: List[str] = ["# CLI Options Audit\n"]
    for script in CLI_SCRIPTS:
        cmd = [sys.executable, script, "-h"]
        ok, out = _run_cmd(cmd, timeout=120)
        blocks.append(f"## {script}")
        blocks.append(f"Command: `{shlex.join(cmd)}`")
        if ok:
            blocks.append("```text")
            blocks.append(out.strip() or "(no output)")
            blocks.append("```")
        else:
            blocks.append("```text")
            blocks.append(out.strip() or "(failed without output)")
            blocks.append("```")
    return "\n".join(blocks)


def _list_datasets(base_dir: str) -> List[str]:
    p = Path(base_dir).expanduser().resolve()
    if not p.exists():
        return []
    out = []
    for child in sorted([c for c in p.iterdir() if c.is_dir()]):
        if (child / "train").exists() or (child / "val").exists():
            out.append(str(child))
    return out


def _list_images(split_images_dir: Path) -> List[str]:
    if not split_images_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p.name for p in split_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except Exception:
            continue
    return rows


def _draw_yolo_boxes(image_path: Path, label_path: Path) -> Tuple[np.ndarray, str]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    labels = _read_yolo_labels(label_path)

    class_colors = {
        0: (255, 140, 0),
        1: (0, 220, 255),
        2: (130, 255, 130),
    }
    lines = []
    for i, (cls, cx, cy, bw, bh) in enumerate(labels, start=1):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = class_colors.get(cls, (255, 80, 80))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{cls}", fill=color)
        lines.append(f"{i}. class={cls} cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}")

    summary = f"Found {len(labels)} boxes\n" + ("\n".join(lines[:25]) if lines else "")
    if len(lines) > 25:
        summary += f"\n... +{len(lines)-25} more"
    return np.array(img), summary


def run_generate_synthetic(
    background_train: str,
    background_val: str,
    output_dir: str,
    dataset_name: str,
    corp_tib_num: str,
    corp_tib_text: str,
    corp_chi_num: str,
    train_samples: int,
    val_samples: int,
    font_tib: str,
    font_chi: str,
    image_width: int,
    image_height: int,
    augmentation: str,
    annotations_file_path: str,
    single_label: bool,
    debug: bool,
):
    cmd = [
        sys.executable,
        "generate_training_data.py",
        "--background_train",
        background_train,
        "--background_val",
        background_val,
        "--output_dir",
        output_dir,
        "--dataset_name",
        dataset_name,
        "--corpora_tibetan_numbers_path",
        corp_tib_num,
        "--corpora_tibetan_text_path",
        corp_tib_text,
        "--corpora_chinese_numbers_path",
        corp_chi_num,
        "--train_samples",
        str(int(train_samples)),
        "--val_samples",
        str(int(val_samples)),
        "--font_path_tibetan",
        font_tib,
        "--font_path_chinese",
        font_chi,
        "--image_width",
        str(int(image_width)),
        "--image_height",
        str(int(image_height)),
        "--augmentation",
        augmentation,
    ]
    if annotations_file_path.strip():
        cmd.extend(["--annotations_file_path", annotations_file_path.strip()])
    if single_label:
        cmd.append("--single_label")
    if debug:
        cmd.append("--debug")

    ok, out = _run_cmd(cmd, timeout=7200)
    dataset_path = str((Path(output_dir).expanduser().resolve() / dataset_name))
    status = "Success" if ok else "Failed"
    return f"{status}\nDataset path: {dataset_path}\n\n{out}", dataset_path


def refresh_image_list(dataset_dir: str, split: str):
    split_images = Path(dataset_dir).expanduser().resolve() / split / "images"
    images = _list_images(split_images)
    value = images[0] if images else None
    return gr.update(choices=images, value=value), f"{len(images)} image(s) found in {split_images}"


def preview_sample(dataset_dir: str, split: str, image_name: str):
    if not dataset_dir or not image_name:
        return None, "Select dataset/split/image first."
    image_path = Path(dataset_dir).expanduser().resolve() / split / "images" / image_name
    label_path = Path(dataset_dir).expanduser().resolve() / split / "labels" / f"{Path(image_name).stem}.txt"
    if not image_path.exists():
        return None, f"Image not found: {image_path}"
    rendered, summary = _draw_yolo_boxes(image_path, label_path)
    return rendered, summary


def export_to_label_studio(
    split_dir: str,
    image_ext: str,
    tasks_json: str,
    image_root_url: str,
):
    split_path = Path(split_dir).expanduser().resolve()
    out_path = Path(tasks_json).expanduser().resolve()
    cmd = [
        "label-studio-converter",
        "import",
        "yolo",
        "-i",
        str(split_path),
        "-o",
        str(out_path),
        "--image-ext",
        image_ext,
        "--image-root-url",
        image_root_url,
    ]
    ok, out = _run_cmd(cmd, timeout=1200)
    status = "Success" if ok else "Failed"
    env_hint = (
        "Set these env vars before starting Label Studio:\n"
        f"export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
        f"export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={split_path.parent.resolve()}\n"
    )
    return f"{status}\nTasks file: {out_path}\n\n{env_hint}\n{out}"


def start_label_studio(local_files_root: str):
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(Path(local_files_root).expanduser().resolve())
    try:
        proc = subprocess.Popen(
            ["label-studio"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Label Studio started (PID {proc.pid}). Local files root: {env['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT']}"
    except FileNotFoundError:
        return "label-studio command not found. Install with: pip install label-studio"
    except Exception as exc:
        return f"Failed to start Label Studio: {type(exc).__name__}: {exc}"


def build_ui() -> gr.Blocks:
    default_dataset_base = str((ROOT / "datasets").resolve())
    default_dataset = str((ROOT / "datasets" / "tibetan-yolo").resolve())
    default_split_dir = str((ROOT / "datasets" / "tibetan-yolo" / "train").resolve())

    with gr.Blocks(title="PechaBridge Workbench") as demo:
        gr.Markdown("# PechaBridge Workbench")
        gr.Markdown(
            "Use this UI to inspect CLI options, generate synthetic data, preview labels, and export to Label Studio."
        )

        with gr.Tab("CLI Audit"):
            audit_btn = gr.Button("Scan All CLI Options")
            audit_out = gr.Markdown()
            audit_btn.click(fn=collect_cli_help, inputs=[], outputs=[audit_out])

        with gr.Tab("Synthetic Data"):
            gr.Markdown("Generate synthetic multi-class YOLO data using `generate_training_data.py`.")
            with gr.Row():
                with gr.Column():
                    background_train = gr.Textbox(label="background_train", value="./data/tibetan numbers/backgrounds/")
                    background_val = gr.Textbox(label="background_val", value="./data/tibetan numbers/backgrounds/")
                    output_dir = gr.Textbox(label="output_dir", value="./datasets")
                    dataset_name = gr.Textbox(label="dataset_name", value="tibetan-yolo-ui")
                    corp_tib_num = gr.Textbox(label="corpora_tibetan_numbers_path", value="./data/corpora/Tibetan Number Words/")
                    corp_tib_text = gr.Textbox(label="corpora_tibetan_text_path", value="./data/corpora/UVA Tibetan Spoken Corpus/")
                    corp_chi_num = gr.Textbox(label="corpora_chinese_numbers_path", value="./data/corpora/Chinese Number Words/")
                with gr.Column():
                    train_samples = gr.Number(label="train_samples", value=100, precision=0)
                    val_samples = gr.Number(label="val_samples", value=100, precision=0)
                    font_tib = gr.Textbox(label="font_path_tibetan", value="ext/Microsoft Himalaya.ttf")
                    font_chi = gr.Textbox(label="font_path_chinese", value="ext/simkai.ttf")
                    image_width = gr.Number(label="image_width", value=1024, precision=0)
                    image_height = gr.Number(label="image_height", value=361, precision=0)
                    augmentation = gr.Dropdown(label="augmentation", choices=["rotate", "noise", "none"], value="noise")
                    annotations_file_path = gr.Textbox(label="annotations_file_path", value="./data/tibetan numbers/annotations/tibetan_chinese_no/bg_PPN337138764X_00000005.txt")
                    single_label = gr.Checkbox(label="single_label", value=False)
                    debug = gr.Checkbox(label="debug", value=False)

            generate_btn = gr.Button("Generate Dataset", variant="primary")
            gen_log = gr.Textbox(label="Generation Log", lines=18)
            generated_dataset_path = gr.Textbox(label="Generated Dataset Path", interactive=False)
            generate_btn.click(
                fn=run_generate_synthetic,
                inputs=[
                    background_train,
                    background_val,
                    output_dir,
                    dataset_name,
                    corp_tib_num,
                    corp_tib_text,
                    corp_chi_num,
                    train_samples,
                    val_samples,
                    font_tib,
                    font_chi,
                    image_width,
                    image_height,
                    augmentation,
                    annotations_file_path,
                    single_label,
                    debug,
                ],
                outputs=[gen_log, generated_dataset_path],
            )

        with gr.Tab("Dataset Preview"):
            gr.Markdown("Inspect generated dataset and render YOLO label boxes.")
            with gr.Row():
                dataset_base = gr.Textbox(label="Datasets Base Directory", value=default_dataset_base)
                scan_datasets_btn = gr.Button("Scan Datasets")
            dataset_select = gr.Dropdown(label="Dataset Directory", choices=[default_dataset], value=default_dataset)
            split_select = gr.Dropdown(label="Split", choices=["train", "val"], value="train")
            with gr.Row():
                refresh_images_btn = gr.Button("Refresh Image List")
                image_select = gr.Dropdown(label="Image", choices=[])
            preview_hint = gr.Textbox(label="Preview Status", interactive=False)
            preview_btn = gr.Button("Render Preview", variant="primary")
            preview_img = gr.Image(label="Image with Label Boxes", type="numpy")
            preview_txt = gr.Textbox(label="Label Summary", lines=12, interactive=False)

            def _scan_dataset_dirs(base: str):
                choices = _list_datasets(base)
                return gr.update(choices=choices, value=(choices[0] if choices else None))

            scan_datasets_btn.click(fn=_scan_dataset_dirs, inputs=[dataset_base], outputs=[dataset_select])
            refresh_images_btn.click(
                fn=refresh_image_list,
                inputs=[dataset_select, split_select],
                outputs=[image_select, preview_hint],
            )
            preview_btn.click(
                fn=preview_sample,
                inputs=[dataset_select, split_select, image_select],
                outputs=[preview_img, preview_txt],
            )

        with gr.Tab("Label Studio Export"):
            gr.Markdown("Convert YOLO split directory to Label Studio tasks.")
            split_dir = gr.Textbox(label="YOLO Split Directory", value=default_split_dir)
            image_ext = gr.Dropdown(label="image-ext", choices=[".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"], value=".png")
            tasks_json = gr.Textbox(label="tasks json output", value=str((ROOT / "ls-tasks-ui.json").resolve()))
            image_root_url = gr.Textbox(label="image-root-url", value="/data/local-files/?d=train/images")
            export_btn = gr.Button("Export to Label Studio Tasks", variant="primary")
            export_log = gr.Textbox(label="Export Log", lines=14)

            local_files_root = gr.Textbox(
                label="Label Studio local files root",
                value=str((ROOT / "datasets" / "tibetan-yolo").resolve()),
            )
            start_ls_btn = gr.Button("Start Label Studio")
            start_ls_msg = gr.Textbox(label="Label Studio Status", interactive=False)

            export_btn.click(
                fn=export_to_label_studio,
                inputs=[split_dir, image_ext, tasks_json, image_root_url],
                outputs=[export_log],
            )
            start_ls_btn.click(
                fn=start_label_studio,
                inputs=[local_files_root],
                outputs=[start_ls_msg],
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
