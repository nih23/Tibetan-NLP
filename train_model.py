#!/usr/bin/env python3
"""
Skript zum Trainieren eines YOLO-Modells mit den generierten Tibetischen OCR-Daten.
Unterstützt Weights & Biases (wandb) Logging für Experiment-Tracking.
"""

import argparse
import os
from pathlib import Path
import wandb
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Trainiere ein YOLO-Modell mit Tibetischen OCR-Daten")

    parser.add_argument('--dataset', type=str, default='yolo_tibetan/',
                        help='Name des Datensatzes (Ordner im YOLO datasets Verzeichnis)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Pfad zum Basis-Modell (z.B. yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Anzahl der Trainings-Epochen')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch-Größe für das Training')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Bildgröße für das Training (entspricht --image_size in main.py)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Anzahl der Worker für das Datenladen')
    parser.add_argument('--device', type=str, default='',
                        help='Gerät für das Training (z.B. cpu, 0, 0,1,2,3 für mehrere GPUs)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Projektname für die Ausgabe')
    parser.add_argument('--name', type=str, default='train',
                        help='Experimentname')
    parser.add_argument('--export', action='store_true',
                        help='Exportiere das Modell nach dem Training als TorchScript')
    parser.add_argument('--patience', type=int, default=50,
                        help='EarlyStopping-Geduld (Epochen ohne Verbesserung)')
    
    # Weights & Biases Argumente
    parser.add_argument('--wandb', action='store_true',
                        help='Aktiviere Weights & Biases Logging')
    parser.add_argument('--wandb-project', type=str, default='TibetanOCR',
                        help='Weights & Biases Projektname')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases Entity (Team oder Benutzername)')
    parser.add_argument('--wandb-tags', type=str, default=None,
                        help='Komma-getrennte Tags für das Experiment (z.B. "yolov8,tibetan")')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Name des Experiments in wandb (Standard: gleich wie --name)')

    args = parser.parse_args()

    # Pfad zur Datensatz-Konfiguration
    from ultralytics.data.utils import DATASETS_DIR
    data_path = Path(DATASETS_DIR) / args.dataset / 'data.yml'
    
    if not data_path.exists():
        print(f"Fehler: Datensatz-Konfiguration nicht gefunden: {data_path}")
        print("Bitte generiere zuerst den Datensatz mit main.py:")
        print("python main.py --train_samples 1000 --val_samples 200 --image_size 1024")
        return

    print(f"Starte Training mit Datensatz: {data_path}")
    print(f"Basis-Modell: {args.model}")
    print(f"Epochen: {args.epochs}")
    print(f"Bildgröße: {args.imgsz}x{args.imgsz}")
    
    # Weights & Biases initialisieren, wenn aktiviert
    if args.wandb:
        wandb_name = args.wandb_name if args.wandb_name else args.name
        wandb_tags = args.wandb_tags.split(',') if args.wandb_tags else None
        
        print(f"Initialisiere Weights & Biases Logging")
        print(f"  Projekt: {args.wandb_project}")
        print(f"  Entity: {args.wandb_entity or 'Standard'}")
        print(f"  Run-Name: {wandb_name}")
        
        # wandb initialisieren
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

    # Modell laden
    model = YOLO(args.model)

    # Training starten
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Weights & Biases Konfiguration
        plots=True,  # Plots für wandb generieren
        save_period=10,  # Modell alle 10 Epochen speichern
        # wandb-Logging aktivieren, wenn gewünscht
        **({'upload_dataset': True, 'logger': 'wandb'} if args.wandb else {})
    )

    # Bestes Modell-Pfad
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    
    print(f"\nTraining abgeschlossen. Bestes Modell gespeichert unter: {best_model_path}")

    # Modell exportieren, wenn gewünscht
    if args.export and best_model_path.exists():
        print("\nExportiere Modell als TorchScript...")
        export_model = YOLO(str(best_model_path))
        export_model.export(format='torchscript')
        print(f"Modell exportiert nach: {best_model_path.with_suffix('.torchscript')}")
        
        # Beispielbefehl für Inferenz
        print("\nBeispiel für Inferenz mit dem exportierten Modell:")
        print(f"yolo predict task=detect model={best_model_path.with_suffix('.torchscript')} imgsz={args.imgsz} source=data/my_inference_data/*.jpg")
    
    # wandb beenden, wenn es verwendet wurde
    if args.wandb:
        # Modell-Artefakt zu wandb hinzufügen
        if best_model_path.exists():
            artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
            artifact.add_file(str(best_model_path))
            if args.export and best_model_path.with_suffix('.torchscript').exists():
                artifact.add_file(str(best_model_path.with_suffix('.torchscript')))
            wandb.log_artifact(artifact)
        
        # wandb-Run beenden
        wandb.finish()


if __name__ == "__main__":
    main()
