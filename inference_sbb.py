#!/usr/bin/env python3
"""
Skript zur Inferenz mit einem trainierten YOLO-Modell für Tibetische OCR
mit Daten von der Staatsbibliothek zu Berlin.
"""

import argparse
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
from PIL import Image
import io
from ultralytics import YOLO


def get_images_from_sbb(ppn):
    """
    Ruft Bilddaten von der Staatsbibliothek zu Berlin ab.
    
    Args:
        ppn: Die PPN (Pica Production Number) des Dokuments
        
    Returns:
        Eine Liste von URLs zu den Bildern
    """
    print(f"Rufe Metadaten für PPN {ppn} ab...")
    files = []
    
    try:
        metadata_url = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"
        with urllib.request.urlopen(metadata_url) as response:
            metadata = ET.parse(response).getroot()
            
            # Namespace für METS XML
            ns = {
                'mets': 'http://www.loc.gov/METS/',
                'xlink': 'http://www.w3.org/1999/xlink'
            }
            
            # Suche nach der fileGrp mit USE="DEFAULT"
            for fileGrp in metadata.findall('.//mets:fileGrp[@USE="DEFAULT"]', ns):
                for file in fileGrp.findall('.//mets:file', ns):
                    flocat = file.find('.//mets:FLocat', ns)
                    if flocat is not None:
                        url = flocat.get('{http://www.w3.org/1999/xlink}href')
                        files.append(url)
                        
            print(f"Gefunden: {len(files)} Bilder")
            
    except Exception as e:
        print(f"Fehler beim Abrufen der Metadaten: {e}")
    
    return files


def download_image(url, output_dir=None):
    """
    Lädt ein Bild von einer URL herunter und speichert es optional.
    
    Args:
        url: Die URL des Bildes
        output_dir: Optional. Verzeichnis zum Speichern des Bildes
        
    Returns:
        Pfad zum heruntergeladenen Bild oder das Bild im Speicher
    """
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
            
            # Extrahiere Dateinamen aus URL
            match = re.search(r'PPN(\d{10})-(\d{8})', url)
            if match:
                filename = f"PPN{match.group(1)}-{match.group(2)}.jpg"
            else:
                filename = f"image_{hash(url)}.jpg"
            
            if output_dir:
                # Speichere Bild auf Festplatte
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, filename)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                return image_path
            else:
                # Gib Bild im Speicher zurück
                return Image.open(io.BytesIO(image_data))
                
    except Exception as e:
        print(f"Fehler beim Herunterladen des Bildes {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Führe Inferenz mit einem trainierten YOLO-Modell für Tibetische OCR auf Daten der Staatsbibliothek Berlin durch")

    parser.add_argument('--ppn', type=str, required=True,
                        help='PPN (Pica Production Number) des Dokuments in der Staatsbibliothek zu Berlin')
    parser.add_argument('--model', type=str, required=True,
                        help='Pfad zum trainierten Modell (z.B. runs/detect/train/weights/best.pt oder best.torchscript)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Bildgröße für die Inferenz (sollte mit der Trainingsgröße übereinstimmen)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Konfidenz-Schwellenwert für Detektionen')
    parser.add_argument('--device', type=str, default='',
                        help='Gerät für die Inferenz (z.B. cpu, 0, 0,1,2,3 für mehrere GPUs)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Speichere Ergebnisse')
    parser.add_argument('--download', action='store_true',
                        help='Lade Bilder herunter anstatt sie direkt zu verarbeiten')
    parser.add_argument('--output', type=str, default='sbb_images',
                        help='Verzeichnis zum Speichern der heruntergeladenen Bilder (wenn --download gesetzt ist)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Projektname für die Ausgabe')
    parser.add_argument('--name', type=str, default='predict_sbb',
                        help='Experimentname')
    parser.add_argument('--show', action='store_true',
                        help='Zeige Ergebnisse während der Inferenz an')
    parser.add_argument('--save-txt', action='store_true',
                        help='Speichere Ergebnisse als .txt Dateien')
    parser.add_argument('--save-conf', action='store_true',
                        help='Speichere Konfidenzwerte in .txt Dateien')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Maximale Anzahl an Bildern für die Inferenz (0 = alle)')

    args = parser.parse_args()

    # Überprüfen, ob das Modell existiert
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Fehler: Modell nicht gefunden: {model_path}")
        return

    # Bilder von der Staatsbibliothek abrufen
    image_urls = get_images_from_sbb(args.ppn)
    
    if not image_urls:
        print("Keine Bilder gefunden. Beende Programm.")
        return
    
    # Begrenze die Anzahl der Bilder, wenn gewünscht
    if args.max_images > 0 and len(image_urls) > args.max_images:
        print(f"Begrenze Anzahl der Bilder auf {args.max_images} (von {len(image_urls)})")
        image_urls = image_urls[:args.max_images]

    # Modell laden
    print(f"Lade Modell: {model_path}")
    model = YOLO(str(model_path))

    # Verarbeite Bilder
    if args.download:
        # Lade Bilder herunter und führe Inferenz auf lokalen Dateien durch
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        print(f"Lade Bilder in Verzeichnis: {output_dir}")
        
        image_paths = []
        for url in image_urls:
            image_path = download_image(url, output_dir)
            if image_path:
                image_paths.append(image_path)
        
        if not image_paths:
            print("Keine Bilder konnten heruntergeladen werden. Beende Programm.")
            return
            
        print(f"Führe Inferenz auf {len(image_paths)} heruntergeladenen Bildern durch...")
        results = model.predict(
            source=image_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            save=args.save,
            project=args.project,
            name=args.name,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf
        )
    else:
        # Verarbeite Bilder direkt aus dem Web
        print(f"Führe Inferenz auf {len(image_urls)} Bildern direkt aus dem Web durch...")
        
        # Erstelle temporäres Verzeichnis für die Ergebnisse
        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            
            for i, url in enumerate(image_urls):
                print(f"Verarbeite Bild {i+1}/{len(image_urls)}: {url}")
                try:
                    # Bild herunterladen (ohne zu speichern)
                    image = download_image(url)
                    if image is None:
                        continue
                    
                    # Inferenz auf dem Bild durchführen
                    result = model.predict(
                        source=image,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=args.device,
                        save=args.save,
                        project=args.project,
                        name=f"{args.name}/{i+1}",
                        show=args.show,
                        save_txt=args.save_txt,
                        save_conf=args.save_conf
                    )
                    
                    results.extend(result)
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {url}: {e}")

    # Ausgabeverzeichnis
    output_dir = Path(args.project) / args.name
    print(f"\nInferenz abgeschlossen. Ergebnisse gespeichert unter: {output_dir}")

    # Zusammenfassung der Ergebnisse
    if results:
        total_detections = sum(len(r.boxes) for r in results if hasattr(r, 'boxes'))
        print(f"Insgesamt {total_detections} Tibetische Textblöcke erkannt.")


if __name__ == "__main__":
    main()
