#!/bin/bash

VENV_DIR="venv"
PYTHON_PROGRAM="yolo_pose_stream.py"
YOLO_MODEL="yolo11n-pose.pt"
YOLO_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"

echo "🚀 Avvio YOLOv11 Pose Stream..."

# Chiedi all'utente il tipo di installazione
read -p "🔧 Vuoi un'installazione Completa (F) o Incrementale (I)? [F/I] " -n 1 -r
echo
if [[ $REPLY =~ ^[Ff]$ ]]; then
    echo "🔄 Installazione COMPLETA selezionata - Tutti i pacchetti verranno reinstallati"
    INSTALL_MODE="full"
else
    echo "➕ Installazione INCREMENTALE selezionata - Solo i pacchetti mancanti verranno installati"
    INSTALL_MODE="incremental"
fi

# Gestione ambiente virtuale
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Ambiente virtuale non trovato. Creo venv..."
    python3 -m venv $VENV_DIR
else
    if [ "$INSTALL_MODE" == "full" ]; then
        echo "♻️ Reinstallazione completa - Ricreo venv..."
        rm -rf $VENV_DIR
        python3 -m venv $VENV_DIR
    fi
fi

source $VENV_DIR/bin/activate

# Installazione requirements
if [ "$INSTALL_MODE" == "full" ]; then
    echo "📦 Installazione COMPLETA dei requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt --force-reinstall
else
    if ! python -c "import ultralytics" &> /dev/null || ! python -c "import streamlit" &> /dev/null; then
        echo "📦 Pacchetti mancanti. Installo requirements..."
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "✅ Tutti i pacchetti già installati."
    fi
fi

# Download modello YOLO
if [ ! -f "$YOLO_MODEL" ]; then
    echo "⬇️ Download modello YOLOv11 Pose..."
    wget -O $YOLO_MODEL $YOLO_MODEL_URL
else
    if [ "$INSTALL_MODE" == "full" ]; then
        echo "♻️ Reinstallazione completa - Scarico nuovamente il modello YOLOv11 Pose..."
        rm -f $YOLO_MODEL
        wget -O $YOLO_MODEL $YOLO_MODEL_URL
    else
        echo "✅ Modello YOLOv11 Pose trovato."
    fi
fi

# Verifica file principale
if [ ! -f "$PYTHON_PROGRAM" ]; then
    echo "❌ ERRORE: File $PYTHON_PROGRAM non trovato!"
    exit 1
fi

# Attivazione ambiente virtuale prima dell'esecuzione
echo "🔌 Attivo l'ambiente virtuale..."
source $VENV_DIR/bin/activate

echo "🏃‍♂️ Eseguo $PYTHON_PROGRAM con Streamlit..."
streamlit run $PYTHON_PROGRAM

echo "✅ Programma terminato."
