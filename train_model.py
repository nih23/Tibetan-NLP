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
    data_path = Path(DATASETS_DIR) / args.dataset / 'data.yml'
    
    if not data_path.exists():
        print(f"Fehler: Datensatz-Konfiguration nicht gefunden: {data_path}")
        print("Bitte generiere zuerst den Datensatz mit generate_training_data.py:")
        print("python generate_training_data.py --train_samples 1000 --val_samples 200 --image_size 1024")
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
    train_args = {
        'data': str(data_path),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'plots': True,
        'save_period': 10,
    }
    
    # Add wandb logging if enabled
    if wandb_enabled:
        train_args.update({
            'upload_dataset': True,
            'logger': 'wandb'
        })
    
    # Train model
    results = ModelManager.train_model(model, **train_args)

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
