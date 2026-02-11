# Pseudo-Labeling mit VLM + Layout-Filter + Label Studio Import

Diese Anleitung zeigt den Workflow für deine 3 Layout-Komponenten:
- `0`: `tibetan_number_word` (links)
- `1`: `tibetan_text` (mitte)
- `2`: `chinese_number_word` (rechts)

## 1. Voraussetzungen

Installiere Projekt-Dependencies plus Label-Studio-Tools:

```bash
pip install -r requirements.txt
pip install label-studio label-studio-converter
```

Wenn du HF-Transformer-Backends nutzt, müssen `transformers` und `torch` verfügbar sein.

## 2. Pseudo-Labels aus VLM erzeugen

Beispiel mit `paddleocr_vl`:

```bash
python pseudo_label_from_vlm.py \
  --input-dir data/my_inference_data \
  --output-dir datasets/pseudo-vlm \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --copy-images \
  --save-raw-json \
  --min-confidence 0.2
```

Alternative Parser:
- `--parser qwen25vl`
- `--parser granite_docling`
- `--parser mineru25` (mit `--mineru-command` / `--mineru-timeout`)

Ergebnisstruktur:

```text
datasets/pseudo-vlm/train/
  images/
  labels/
  classes.txt
  raw_json/                  # optional
  pseudo_labels_report.json
```

## 2.5 Vollständiger Workflow mit einer Datei

Statt die Schritte einzeln auszuführen, kannst du alles bis zum Label-Studio-Start mit einem Skript abbilden:

```bash
python run_pseudo_label_workflow.py \
  --input-dir data/my_inference_data \
  --work-dir datasets/workflow \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --allow-relabel \
  --keep-single-per-class \
  --image-ext .jpg
```

Optional Label Studio direkt starten:

```bash
python run_pseudo_label_workflow.py \
  --input-dir data/my_inference_data \
  --work-dir datasets/workflow \
  --split train \
  --parser paddleocr_vl \
  --recursive \
  --allow-relabel \
  --keep-single-per-class \
  --image-ext .jpg \
  --start-label-studio
```

Das Skript führt intern aus:
1. `pseudo_label_from_vlm.py`
2. `layout_rule_filter.py`
3. `label-studio-converter import yolo`
4. (optional) `label-studio`

## 3. Layout-Regeln anwenden (links/mitte/rechts)

Diese Stufe filtert oder relabelt Boxen nach Seitenlogik.

```bash
python layout_rule_filter.py \
  --input-split-dir datasets/pseudo-vlm/train \
  --output-split-dir datasets/pseudo-vlm-filtered/train \
  --allow-relabel \
  --keep-single-per-class \
  --copy-images
```

Wichtige Optionen:
- `--allow-relabel`: falsche Klasse wird anhand Zone korrigiert
- `--keep-single-per-class`: maximal eine Box pro Klasse behalten
- `--left-max 0.33 --right-min 0.66`: Zonengrenzen anpassbar

Ergebnis:

```text
datasets/pseudo-vlm-filtered/train/
  images/
  labels/
  classes.txt
  layout_rule_report.json
```

## 4. In Label Studio importieren

### 4.1 Local Files Serving aktivieren

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/datasets/pseudo-vlm-filtered
```

### 4.2 YOLO nach Label Studio Tasks konvertieren

```bash
label-studio-converter import yolo \
  -i datasets/pseudo-vlm-filtered/train \
  -o ls-tasks-pseudo.json \
  --image-ext ".png" \
  --image-root-url "/data/local-files/?d=train/images"
```

Hinweis:
- Falls deine Bilder `.jpg` sind, setze `--image-ext ".jpg"`.
- Bei gemischten Formaten ggf. vorher vereinheitlichen.

### 4.3 Label Studio starten

```bash
label-studio
```

Dann in Label Studio:
1. Neues Projekt anlegen
2. Storage: `Local files` hinzufügen
3. Absoluten Pfad setzen auf:
   `$(pwd)/datasets/pseudo-vlm-filtered`
4. `ls-tasks-pseudo.json` importieren
5. Pseudo-Labels korrigieren und exportieren

## 5. Nächster Schritt (für Training)

Nach Korrektur in Label Studio:
- Exportierte YOLO-Labels zurück ins Trainingsset übernehmen
- Optional: `train`/`val` splitten
- Dann Modelltraining starten (YOLO oder anderes Detektionsmodell)
