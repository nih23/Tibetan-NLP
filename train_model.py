#!/usr/bin/env python3
"""
Skript zum Trainieren eines YOLO-Modells mit den generierten Tibetischen OCR-Daten.
Unterstützt Weights & Biases (wandb) Logging für Experiment-Tracking.
"""

import os
from pathlib import Path
import wandb
from ultralytics.data.utils import DATASETS_DIR

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_train_parser
from tibetan_utils.model_utils import ModelManager


def initialize_wandb(args):
    """
    Initialize Weights & Biases logging.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: Whether wandb was initialized
    """
    if not args.wandb:
        return False
        
    wandb_name = args.wandb_name if args.wandb_name else args.name
    wandb_tags = args.wandb_tags.split(',') if args.wandb_tags else None
    
    print(f"Initialisiere Weights & Biases Logging")
    print(f"  Projekt: {args.wandb_project}")
    print(f"  Entity: {args.wandb_entity or 'Standard'}")
    print(f"  Run-Name: {wandb_name}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        tags=wandb_tags,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "image_size": args.imgsz,
            "patience": args.patience
        }
    )
    
    return True


def save_model_to_wandb(model_path, export_path=None):
    """
    Save model to Weights & Biases as an artifact.
    
    Args:
        model_path: Path to the model file
        export_path: Path to the exported model file
    """
    if wandb.run is None:
        return
        
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(str(model_path))
    
    if export_path and os.path.exists(export_path):
        artifact.add_file(str(export_path))
        
    wandb.log_artifact(artifact)


def main():
    # Parse arguments
    parser = create_train_parser()
    args = parser.parse_args()

    # Path to dataset configuration
    script_root = Path(__file__).resolve().parent
    dataset_arg = Path(str(args.dataset)).expanduser()
    candidates = []

    # 1) Direct YAML path passed by user/UI.
    if dataset_arg.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(dataset_arg)
    # 2) Direct dataset folder path containing data.yml.
    candidates.append(dataset_arg / "data.yml")
    # 3) Project-local relative path.
    candidates.append(script_root / dataset_arg)
    # 4) Project-local datasets folder (name or yaml filename).
    candidates.append(script_root / "datasets" / dataset_arg)
    if dataset_arg.suffix.lower() not in {".yml", ".yaml"}:
        candidates.append(script_root / "datasets" / f"{str(args.dataset)}.yaml")
        candidates.append(script_root / "datasets" / f"{str(args.dataset)}.yml")
    # 5) Project-local datasets/<name>/data.yml (folder layout).
    candidates.append(script_root / "datasets" / str(args.dataset) / "data.yml")
    # 6) Ultralytics global datasets dir (name, yaml, folder/data.yml).
    if dataset_arg.suffix.lower() in {".yml", ".yaml"}:
        candidates.append(Path(DATASETS_DIR) / dataset_arg.name)
    else:
        candidates.append(Path(DATASETS_DIR) / f"{str(args.dataset)}.yaml")
        candidates.append(Path(DATASETS_DIR) / f"{str(args.dataset)}.yml")
    candidates.append(Path(DATASETS_DIR) / str(args.dataset) / "data.yml")

    resolved_candidates = []
    for p in candidates:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            resolved_candidates.append(p)
        elif p.is_dir():
            yml = p / "data.yml"
            yaml_alt = p / "data.yaml"
            if yml.exists():
                resolved_candidates.append(yml.resolve())
            elif yaml_alt.exists():
                resolved_candidates.append(yaml_alt.resolve())

    data_path = resolved_candidates[0] if resolved_candidates else None
    
    if data_path is None:
        print(f"Fehler: Datensatz-Konfiguration nicht gefunden fuer --dataset={args.dataset}")
        print("Erwartet eine der folgenden Eingaben fuer --dataset:")
        print("  - Datensatzname (z.B. tibetan-yolo)")
        print("  - Absoluter/relativer Pfad zu einem Datensatzordner mit data.yml")
        print("  - Absoluter/relativer Pfad direkt auf data.yml")
        print("Beispiel:")
        print("python train_model.py --dataset ./datasets/tibetan-yolo --epochs 100")
        return

    print(f"Starte Training mit Datensatz: {data_path}")
    print(f"Basis-Modell: {args.model}")
    print(f"Epochen: {args.epochs}")
    print(f"Bildgröße: {args.imgsz}x{args.imgsz}")
    
    # Initialize Weights & Biases if enabled
    wandb_enabled = initialize_wandb(args)

    # Load model
    model = ModelManager.load_model(args.model)

    # Start training
    # Train model
    results = ModelManager.train_model(
        model,
        data_path=str(data_path),
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        use_wandb=wandb_enabled,
    )

    # Best model path
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    
    print(f"\nTraining abgeschlossen. Bestes Modell gespeichert unter: {best_model_path}")

    # Export model if requested
    export_path = None
    if args.export and best_model_path.exists():
        print("\nExportiere Modell als TorchScript...")
        export_model = ModelManager.load_model(str(best_model_path))
        export_path = ModelManager.export_model(export_model, format='torchscript')
        print(f"Modell exportiert nach: {export_path}")
        
        # Example command for inference
        print("\nBeispiel für Inferenz mit dem exportierten Modell:")
        print(f"yolo predict task=detect model={export_path} imgsz={args.imgsz} source=data/my_inference_data/*.jpg")
    
    # Save model to wandb if enabled
    if wandb_enabled:
        save_model_to_wandb(best_model_path, export_path)
        wandb.finish()


if __name__ == "__main__":
    main()
