#!/usr/bin/env python3
"""
Skript zur Anwendung von Tesseract OCR auf erkannte Textblöcke aus einem YOLO-Modell.
Unterstützt sowohl lokale Bilder als auch Bilder von der Staatsbibliothek zu Berlin.
"""

import os
from pathlib import Path

# Importiere Funktionen aus der tibetan_utils-Bibliothek
from tibetan_utils.arg_utils import create_ocr_parser
from tibetan_utils.model_utils import ModelManager
from tibetan_utils.ocr_utils import process_image_with_ocr
from tibetan_utils.sbb_utils import process_sbb_images
from tibetan_utils.io_utils import find_images, extract_filename


def process_local_image(image_path, model, output_dir, lang, conf, tesseract_config, save_crops):
    """
    Process a local image with OCR.
    
    Args:
        image_path: Path to the image
        model: YOLO model
        output_dir: Output directory
        lang: Language for Tesseract OCR
        conf: Confidence threshold
        tesseract_config: Additional Tesseract configuration
        save_crops: Whether to save cropped text regions
        
    Returns:
        dict: OCR results
    """
    print(f"Verarbeite Bild: {image_path}")
    
    # Process image with OCR
    ocr_results = process_image_with_ocr(
        image_path,
        model,
        output_dir=output_dir,
        lang=lang,
        conf=conf,
        save_crops=save_crops
    )
    
    # Show a summary of the results
    print(f"  Erkannt: {len(ocr_results['detections'])} Textblöcke")
    for j, det in enumerate(ocr_results['detections']):
        text_preview = det['text'].replace('\n', ' ')[:50] + ('...' if len(det['text']) > 50 else '')
        print(f"    Block {j+1}: {text_preview}")
    
    return ocr_results


def process_sbb_image(image, model, output_dir, lang, conf, tesseract_config, save_crops):
    """
    Process an SBB image with OCR.
    
    Args:
        image: Image data
        model: YOLO model
        output_dir: Output directory
        lang: Language for Tesseract OCR
        conf: Confidence threshold
        tesseract_config: Additional Tesseract configuration
        save_crops: Whether to save cropped text regions
        
    Returns:
        dict: OCR results
    """
    # Process image with OCR
    ocr_results = process_image_with_ocr(
        image,
        model,
        output_dir=output_dir,
        lang=lang,
        conf=conf,
        save_crops=save_crops
    )
    
    # Show a summary of the results
    print(f"  Erkannt: {len(ocr_results['detections'])} Textblöcke")
    for j, det in enumerate(ocr_results['detections']):
        text_preview = det['text'].replace('\n', ' ')[:50] + ('...' if len(det['text']) > 50 else '')
        print(f"    Block {j+1}: {text_preview}")
    
    return ocr_results


def main():
    # Parse arguments
    parser = create_ocr_parser()
    args = parser.parse_args()
    
    # Check if either --source or --ppn is specified
    if not args.source and not args.ppn:
        parser.error("Entweder --source oder --ppn muss angegeben werden")
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Fehler: Modell nicht gefunden: {model_path}")
        return
    
    # Load model
    print(f"Lade Modell: {model_path}")
    model = ModelManager.load_model(str(model_path))
    
    # Process local images
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Fehler: Quelle nicht gefunden: {source_path}")
            return
        
        # Collect images
        image_paths = []
        if source_path.is_file():
            image_paths = [str(source_path)]
        else:
            image_paths = find_images(str(source_path), recursive=True)
        
        if not image_paths:
            print(f"Keine Bilder gefunden in: {source_path}")
            return
        
        print(f"Gefunden: {len(image_paths)} Bilder")
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            print(f"Verarbeite Bild {i+1}/{len(image_paths)}: {img_path}")
            process_local_image(
                img_path,
                model,
                args.output,
                args.lang,
                args.conf,
                args.tesseract_config,
                args.save_crops
            )
    
    # Process images from Staatsbibliothek zu Berlin
    elif args.ppn:
        # Process SBB images
        process_args = {
            'model': model,
            'output_dir': args.output,
            'lang': args.lang,
            'conf': args.conf,
            'tesseract_config': args.tesseract_config,
            'save_crops': args.save_crops
        }
        
        results = process_sbb_images(
            args.ppn,
            lambda img, **kwargs: process_sbb_image(img, **kwargs),
            max_images=args.max_images,
            download=args.download,
            output_dir=args.output,
            verify_ssl=not args.no_ssl_verify,
            **process_args
        )
    
    print(f"\nVerarbeitung abgeschlossen. Ergebnisse gespeichert unter: {args.output}")


if __name__ == "__main__":
    main()
