# PechaBridge Workbench

## Project Description

PechaBridge is a workflow for **Tibetan pecha document understanding** with a focus on layout detection, synthetic data generation, SBB data ingestion, OCR/VLM-assisted parsing, diffusion-based texture augmentation, and preparation for future retrieval systems.

The project combines:

- synthetic YOLO dataset generation for Tibetan/number classes,
- training and evaluation of detection models,
- large-scale processing of SBB page images,
- optional VLM backends for layout extraction,
- SDXL/SD2.1 + ControlNet + LoRA texture adaptation,
- unpaired image/text encoder training for later n-gram retrieval.

The primary entrypoint for end-to-end usage is the **Workbench UI** (`ui_workbench.py`).

## Project Goals

1. Build a robust pipeline for Tibetan page layout analysis that works with limited labeled data.
2. Improve model quality through synthetic data and realistic texture transfer from real scans.
3. Support scalable ingestion and weak supervision on large historical collections (for example SBB PPNs).
4. Prepare retrieval-ready representations (image and text encoders) for future Tibetan n-gram search.
5. Keep all major workflows reproducible in both UI and CLI.

## Roadmap

1. Data Foundation:
Synthetic generation, SBB download pipeline, and dataset QA/export workflows.
2. Detection and Parsing:
YOLO training/inference plus optional VLM-assisted layout parsing and pseudo-labeling.
3. Realism and Domain Adaptation:
Diffusion + LoRA texture workflows to bridge synthetic-to-real domain gaps.
4. Retrieval Readiness:
Train unpaired image/text encoders and establish schemas/pipelines for retrieval indexing.
5. Retrieval System:
Dual-encoder alignment, ANN indexing, provenance-aware search results, and iterative evaluation.

## Install

```bash
pip install -r requirements-ui.txt
```

If you want transformer/VLM features in the Workbench (`Batch VLM Layout (SBB)` and `VLM Layout` tabs), also install:

```bash
pip install -r requirements-vlm.txt
```

If you want diffusion + LoRA texture workflows and retrieval encoder training, also install:

```bash
pip install -r requirements-lora.txt
```

## Start the Workbench

```bash
python ui_workbench.py
```

Optional runtime flags via environment variables:

```bash
export UI_HOST=127.0.0.1   # use 0.0.0.0 for remote server binding
export UI_PORT=7860
export UI_SHARE=false      # set true only if you explicitly want a public Gradio link
python ui_workbench.py
```

### SSH Port Forwarding (server -> laptop)

If the Workbench runs on a remote host, keep `UI_SHARE=false` and use SSH forwarding:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

Then open `http://127.0.0.1:7860` on your laptop.

## Recommended Workflow (UI only)

1. `Synthetic Data`: generate synthetic YOLO datasets.
2. `Batch VLM Layout (SBB)`: run VLM-based layout on SBB PPN pages (test-only), combine with synthetic data, export.
3. `Dataset Preview`: inspect images and label boxes.
4. `Ultralytics Training`: train detection models.
5. `Model Inference`: run trained model inference.
6. `VLM Layout`: single-image VLM layout parsing.
7. `Label Studio Export`: convert YOLO splits to Label Studio tasks and optionally launch Label Studio.
8. `PPN Downloader`: download and inspect SBB pages.
9. `Diffusion + LoRA`: prepare texture crops, train LoRA (SDXL or SD2.1), run structure-preserving texture augmentation.
10. `Retrieval Encoders`: train unpaired image encoder + text encoder for later Tibetan n-gram retrieval.
11. `CLI Audit`: view script options.

## Unified CLI (new)

The project includes a unified CLI entrypoint:

```bash
python cli.py -h
```

Key commands:

```bash
# Texture LoRA dataset prep
python cli.py prepare-texture-lora-dataset --input_dir ./sbb_images --output_dir ./datasets/texture-lora-dataset

# Train texture LoRA (SDXL or SD2.1 via --model_family)
python cli.py train-texture-lora --dataset_dir ./datasets/texture-lora-dataset --output_dir ./models/texture-lora-sdxl

# Texture augmentation inference
python cli.py texture-augment --input_dir ./datasets/tibetan-yolo-ui/train/images --output_dir ./datasets/tibetan-yolo-ui-textured

# Train image encoder (self-supervised)
python cli.py train-image-encoder --input_dir ./sbb_images --output_dir ./models/image-encoder

# Train text encoder (unsupervised, Unicode-normalized)
python cli.py train-text-encoder --input_dir ./data/corpora --output_dir ./models/text-encoder
```

## Label Studio Notes

For local file serving in Label Studio, set:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/your/dataset/root
```

Then use the Workbench export actions.

## CLI Documentation

CLI usage is documented separately in:

- [README_CLI.md](README_CLI.md)
- [README_PSEUDO_LABELING_LABEL_STUDIO.md](README_PSEUDO_LABELING_LABEL_STUDIO.md)
- [docs/texture_augmentation.md](docs/texture_augmentation.md)
- [docs/tibetan_ngram_retrieval_plan.md](docs/tibetan_ngram_retrieval_plan.md)

## License

MIT, see [LICENSE](LICENSE).
