#!/bin/bash

VENV_DIR="venv"
PYTHON_PROGRAM="yolo_pose_stream.py"
YOLO_MODEL="yolo11n-pose.pt"
YOLO_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"

echo "🚀 Avvio YOLOv11 Pose Stream..."

if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Ambiente virtuale non trovato. Creo venv..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

if ! python -c "import ultralytics" &> /dev/null; then
    echo "📦 Ultralytics non trovato. Installo requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Ultralytics già installato."
fi

if [ ! -f "$YOLO_MODEL" ]; then
    echo "⬇️ Download modello YOLOv11..."
    wget -O $YOLO_MODEL $YOLO_MODEL_URL
else
    echo "✅ Modello YOLOv11 trovato."
fi

if [ ! -f "$PYTHON_PROGRAM" ]; then
    echo "❌ ERRORE: File $PYTHON_PROGRAM non trovato!"
    exit 1
fi

echo "🏃‍♂️ Eseguo $PYTHON_PROGRAM..."
python $PYTHON_PROGRAM

echo "✅ Programma terminato."
