#!/usr/bin/env python3
"""
Skript zum Trainieren eines YOLO-Modells mit den generierten Tibetischen OCR-Daten.
"""

import argparse
import os
from pathlib import Path
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
        patience=args.patience
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


if __name__ == "__main__":
    main()
