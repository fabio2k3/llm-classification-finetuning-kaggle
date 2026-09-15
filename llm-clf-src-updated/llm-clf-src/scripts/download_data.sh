#!/bin/bash
# Descarga el dataset de la competición usando la Kaggle API.
# Requiere: pip install kaggle  +  ~/.kaggle/kaggle.json con tus credenciales.
#
# Uso: bash scripts/download_data.sh

set -e

COMPETITION="llm-classification-finetuning"
DEST_DIR="data"

mkdir -p "$DEST_DIR"

echo "Descargando dataset de la competición: $COMPETITION ..."
kaggle competitions download -c "$COMPETITION" -p "$DEST_DIR"

echo "Descomprimiendo..."
unzip -o "$DEST_DIR/${COMPETITION}.zip" -d "$DEST_DIR"
rm "$DEST_DIR/${COMPETITION}.zip"

echo "Listo. Archivos en $DEST_DIR:"
ls -la "$DEST_DIR"
