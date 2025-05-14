import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
import requests
from requests.auth import HTTPDigestAuth
import streamlit as st
import threading
import queue
import atexit
import yaml

STATUS_FILE = "status_feed.yaml"


def read_feed_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"feed_1": False, "feed_2": False, "feed_3": False, "feed_4": False}


def write_feed_status(status_dict):
    with open(STATUS_FILE, "w") as f:
        yaml.dump(status_dict, f)


# Configurazione Streamlit
st.set_page_config(layout="wide")
st.title("Rilevamento Persone Cadute con YOLOv11 - Quadruplo Feed")


# Classe per gestire lo stato dell'applicazione
class AppState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.stop_flag = False
            self.stop_flag_2 = False
            self.stop_flag_3 = False
            self.stop_flag_4 = False
            self.processing_thread = None
            self.processing_thread_2 = None
            self.processing_thread_3 = None
            self.processing_thread_4 = None
            self.last_annotated_frame = None
            self.last_annotated_frame_2 = None
            self.last_annotated_frame_3 = None
            self.last_annotated_frame_4 = None
            self.people_counts = {"total": 0, "standing": 0, "fallen": 0}
            self.people_counts_2 = {"total": 0, "standing": 0, "fallen": 0}
            self.people_counts_3 = {"total": 0, "standing": 0, "fallen": 0}
            self.people_counts_4 = {"total": 0, "standing": 0, "fallen": 0}
            self.initialized = True


app_state = AppState()

# Coda per messaggi
message_queue = queue.Queue()
fall_timers = {}
fall_timers_2 = {}
fall_timers_3 = {}
fall_timers_4 = {}

# Sidebar per parametri configurabili
with st.sidebar:
    st.header("Configurazione Feed 1")
    username = st.text_input("Username Feed 1", value="admin")
    password = st.text_input("Password Feed 1", value="Admin123", type="password")
    SNAPSHOT_URL = st.text_input(
        "URL Snapshot Feed 1",
        value="http://10.1.109.141/cgi-bin/snapshot.cgi?channel=2&subtype=0",
    )

    st.header("Configurazione Feed 2")
    username2 = st.text_input("Username Feed 2", value="admin")
    password2 = st.text_input("Password Feed 2", value="Admin123", type="password")
    SNAPSHOT_URL2 = st.text_input(
        "URL Snapshot Feed 2",
        value="http://10.1.109.141/cgi-bin/snapshot.cgi?Channel=1",
    )

    st.header("Configurazione Feed 3")
    username3 = st.text_input("Username Feed 3", value="admin")
    password3 = st.text_input("Password Feed 3", value="Admin123", type="password")
    SNAPSHOT_URL3 = st.text_input(
        "URL Snapshot Feed 3",
        value="http://192.168.1.103/cgi-bin/snapshot.cgi?Channel=1",
    )

    st.header("Configurazione Feed 4")
    username4 = st.text_input("Username Feed 4", value="admin")
    password4 = st.text_input("Password Feed 4", value="Admin123", type="password")
    SNAPSHOT_URL4 = st.text_input(
        "URL Snapshot Feed 4",
        value="http://192.168.1.103/cgi-bin/snapshot.cgi?Channel=1",
    )

    st.markdown("---")
    CONFIDENCE_THRESHOLD = st.slider("Soglia confidenza", 0.1, 1.0, 0.7, 0.05)
    ASPECT_RATIO_THRESHOLD = st.slider(
        "Soglia rapporto altezza/larghezza", 0.1, 1.0, 0.7, 0.05
    )

    SAVE_DIR = "yolo_fallen_people"
    os.makedirs(SAVE_DIR, exist_ok=True)

    st.markdown("---")
    st.markdown("**Stato:**")
    status_text = st.empty()


# Caricamento del modello YOLO
@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")


model = load_model()


# Cleanup dei thread a chiusura app
def cleanup():
    app_state.stop_flag = True
    app_state.stop_flag_2 = True
    app_state.stop_flag_3 = True
    app_state.stop_flag_4 = True
    if app_state.processing_thread and app_state.processing_thread.is_alive():
        app_state.processing_thread.join(timeout=1)
    if app_state.processing_thread_2 and app_state.processing_thread_2.is_alive():
        app_state.processing_thread_2.join(timeout=1)
    if app_state.processing_thread_3 and app_state.processing_thread_3.is_alive():
        app_state.processing_thread_3.join(timeout=1)
    if app_state.processing_thread_4 and app_state.processing_thread_4.is_alive():
        app_state.processing_thread_4.join(timeout=1)


atexit.register(cleanup)


# Funzione comune per analizzare un frame
def analyze_frame(feed_id, frame, fall_timers_local):
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Resetta i contatori per questo frame
    if feed_id == 1:
        app_state.people_counts = {"total": 0, "standing": 0, "fallen": 0}
    elif feed_id == 2:
        app_state.people_counts_2 = {"total": 0, "standing": 0, "fallen": 0}
    elif feed_id == 3:
        app_state.people_counts_3 = {"total": 0, "standing": 0, "fallen": 0}
    else:
        app_state.people_counts_4 = {"total": 0, "standing": 0, "fallen": 0}

    if results and len(results) > 0:
        result = results[0]
        annotated_frame = result.plot(boxes=False) # scipio
        if feed_id == 1:
            app_state.last_annotated_frame = annotated_frame
        elif feed_id == 2:
            app_state.last_annotated_frame_2 = annotated_frame
        elif feed_id == 3:
            app_state.last_annotated_frame_3 = annotated_frame
        else:
            app_state.last_annotated_frame_4 = annotated_frame

        if result.keypoints is not None:
            for i, kp in enumerate(result.keypoints.xy):
                if kp is None or kp.shape[0] == 0:
                    continue

                x_coords = kp[:, 0]
                y_coords = kp[:, 1]

                valid_x = x_coords[x_coords > 0]
                valid_y = y_coords[y_coords > 0]
                if len(valid_x) == 0 or len(valid_y) == 0:
                    continue

                x_min, x_max = valid_x.min(), valid_x.max()
                y_min, y_max = valid_y.min(), valid_y.max()
                width = x_max - x_min
                height = y_max - y_min

                fallen = height / width < ASPECT_RATIO_THRESHOLD if width > 0 else False
                person_id = f"person{feed_id}_{i}"
                message_queue.put(
                    (
                        "info",
                        f"[Feed {feed_id}] Persona {i}: ratio {height / width:.2f} → {'SDRAIATA' if fallen else 'in piedi'}",
                    )
                )

                # Aggiorna contatori
                if feed_id == 1:
                    app_state.people_counts["total"] += 1
                    if fallen:
                        app_state.people_counts["fallen"] += 1
                    else:
                        app_state.people_counts["standing"] += 1
                elif feed_id == 2:
                    app_state.people_counts_2["total"] += 1
                    if fallen:
                        app_state.people_counts_2["fallen"] += 1
                    else:
                        app_state.people_counts_2["standing"] += 1
                elif feed_id == 3:
                    app_state.people_counts_3["total"] += 1
                    if fallen:
                        app_state.people_counts_3["fallen"] += 1
                    else:
                        app_state.people_counts_3["standing"] += 1
                else:
                    app_state.people_counts_4["total"] += 1
                    if fallen:
                        app_state.people_counts_4["fallen"] += 1
                    else:
                        app_state.people_counts_4["standing"] += 1

                current_time = time.time()
                cv2.putText(annotated_frame, "xxx", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2) # 
                cv2.putText(annotated_frame, "yyy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 2) #
                if fallen:
                    if person_id not in fall_timers_local:
                        fall_timers_local[person_id] = current_time
                        message_queue.put(
                            (
                                "warning",
                                f"[Feed {feed_id}] Caduta sospetta per {person_id}",
                            )
                        )
                    elif current_time - fall_timers_local[person_id] >= 3:
                        filename = f"{person_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        filepath = os.path.join(SAVE_DIR, filename)
                        cv2.imwrite(filepath, annotated_frame)
                        message_queue.put(
                            (
                                "error",
                                f"[Feed {feed_id}] 📸 {person_id} sdraiata da 3s. Salvato: {filename}",
                            )
                        )
                        fall_timers_local[person_id] = current_time + 10
                else:
                    if person_id in fall_timers_local:
                        del fall_timers_local[person_id]
    else:
        message_queue.put(("warning", f"[Feed {feed_id}] Nessuna persona rilevata."))


# Thread per Feed 1
def process_frames():
    while not app_state.stop_flag:
        try:
            response = requests.get(
                SNAPSHOT_URL, auth=HTTPDigestAuth(username, password), timeout=5
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                message_queue.put(("warning", "[Feed 1] Frame è None!"))
                time.sleep(2)
                continue
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            analyze_frame(1, frame, fall_timers)
            time.sleep(0.5)
        except Exception as e:
            message_queue.put(("error", f"[Feed 1] Errore: {e}"))
            time.sleep(2)


# Thread per Feed 2
def process_frames_2():
    while not app_state.stop_flag_2:
        try:
            response = requests.get(
                SNAPSHOT_URL2, auth=HTTPDigestAuth(username2, password2), timeout=5
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                message_queue.put(("warning", "[Feed 2] Frame è None!"))
                time.sleep(2)
                continue
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            analyze_frame(2, frame, fall_timers_2)
            time.sleep(0.5)
        except Exception as e:
            message_queue.put(("error", f"[Feed 2] Errore: {e}"))
            time.sleep(2)


# Thread per Feed 3
def process_frames_3():
    while not app_state.stop_flag_3:
        try:
            response = requests.get(
                SNAPSHOT_URL3, auth=HTTPDigestAuth(username3, password3), timeout=5
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                message_queue.put(("warning", "[Feed 3] Frame è None!"))
                time.sleep(2)
                continue
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            analyze_frame(3, frame, fall_timers_3)
            time.sleep(0.5)
        except Exception as e:
            message_queue.put(("error", f"[Feed 3] Errore: {e}"))
            time.sleep(2)


# Thread per Feed 4
def process_frames_4():
    while not app_state.stop_flag_4:
        try:
            response = requests.get(
                SNAPSHOT_URL4, auth=HTTPDigestAuth(username4, password4), timeout=5
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                message_queue.put(("warning", "[Feed 4] Frame è None!"))
                time.sleep(2)
                continue
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            analyze_frame(4, frame, fall_timers_4)
            time.sleep(0.5)
        except Exception as e:
            message_queue.put(("error", f"[Feed 4] Errore: {e}"))
            time.sleep(2)


# Funzione per avviare tutti i feed
def start_all_feeds():
    status = {"feed_1": True, "feed_2": True, "feed_3": True, "feed_4": True}
    write_feed_status(status)
    if (
        app_state.processing_thread is None
        or not app_state.processing_thread.is_alive()
    ):
        app_state.stop_flag = False
        app_state.processing_thread = threading.Thread(
            target=process_frames, daemon=True
        )
        app_state.processing_thread.start()

    if (
        app_state.processing_thread_2 is None
        or not app_state.processing_thread_2.is_alive()
    ):
        app_state.stop_flag_2 = False
        app_state.processing_thread_2 = threading.Thread(
            target=process_frames_2, daemon=True
        )
        app_state.processing_thread_2.start()

    if (
        app_state.processing_thread_3 is None
        or not app_state.processing_thread_3.is_alive()
    ):
        app_state.stop_flag_3 = False
        app_state.processing_thread_3 = threading.Thread(
            target=process_frames_3, daemon=True
        )
        app_state.processing_thread_3.start()

    if (
        app_state.processing_thread_4 is None
        or not app_state.processing_thread_4.is_alive()
    ):
        app_state.stop_flag_4 = False
        app_state.processing_thread_4 = threading.Thread(
            target=process_frames_4, daemon=True
        )
        app_state.processing_thread_4.start()

    st.success("Tutti i feed sono stati avviati!")


# Funzione per fermare tutti i feed
def stop_all_feeds():
    status = {"feed_1": False, "feed_2": False, "feed_3": False, "feed_4": False}
    write_feed_status(status)
    app_state.stop_flag = True
    app_state.stop_flag_2 = True
    app_state.stop_flag_3 = True
    app_state.stop_flag_4 = True
    st.warning("Tutti i feed sono stati fermati.")


# Interfaccia
st.markdown("### Contatore Totale")
total_counter = st.empty()

# Questa variabile di sessione è necessaria per mantenere lo stato
if "show_images" not in st.session_state:
    st.session_state.show_images = True

# Pulsanti separati per mostrare/nascondere immagini
show_images_col, hide_images_col = st.columns(2)
with show_images_col:
    if st.button("Mostra immagini"):
        st.session_state.show_images = True
        st.success("Visualizzazione immagini attivata")
        # Avvia i feed salvati nello YAML
        status = read_feed_status()
        if status.get("feed_1"):
            app_state.stop_flag = False
            if (
                app_state.processing_thread is None
                or not app_state.processing_thread.is_alive()
            ):
                app_state.processing_thread = threading.Thread(
                    target=process_frames, daemon=True
                )
                app_state.processing_thread.start()

        if status.get("feed_2"):
            app_state.stop_flag_2 = False
            if (
                app_state.processing_thread_2 is None
                or not app_state.processing_thread_2.is_alive()
            ):
                app_state.processing_thread_2 = threading.Thread(
                    target=process_frames_2, daemon=True
                )
                app_state.processing_thread_2.start()

        if status.get("feed_3"):
            app_state.stop_flag_3 = False
            if (
                app_state.processing_thread_3 is None
                or not app_state.processing_thread_3.is_alive()
            ):
                app_state.processing_thread_3 = threading.Thread(
                    target=process_frames_3, daemon=True
                )
                app_state.processing_thread_3.start()

        if status.get("feed_4"):
            app_state.stop_flag_4 = False
            if (
                app_state.processing_thread_4 is None
                or not app_state.processing_thread_4.is_alive()
            ):
                app_state.processing_thread_4 = threading.Thread(
                    target=process_frames_4, daemon=True
                )
                app_state.processing_thread_4.start()

with hide_images_col:
    if st.button("Nascondi immagini"):
        st.session_state.show_images = False
        st.info(
            "Visualizzazione immagini disattivata - L'analisi continua in background"
        )

        # Ripristina i feed in base allo stato YAML (come fa "Mostra immagini")
        status = read_feed_status()

        # Gestione Feed 1
        if status.get("feed_1"):
            app_state.stop_flag = False
            if (
                app_state.processing_thread is None
                or not app_state.processing_thread.is_alive()
            ):
                app_state.processing_thread = threading.Thread(
                    target=process_frames, daemon=True
                )
                app_state.processing_thread.start()
        else:
            app_state.stop_flag = True

        # Gestione Feed 2
        if status.get("feed_2"):
            app_state.stop_flag_2 = False
            if (
                app_state.processing_thread_2 is None
                or not app_state.processing_thread_2.is_alive()
            ):
                app_state.processing_thread_2 = threading.Thread(
                    target=process_frames_2, daemon=True
                )
                app_state.processing_thread_2.start()
        else:
            app_state.stop_flag_2 = True

        # Gestione Feed 3
        if status.get("feed_3"):
            app_state.stop_flag_3 = False
            if (
                app_state.processing_thread_3 is None
                or not app_state.processing_thread_3.is_alive()
            ):
                app_state.processing_thread_3 = threading.Thread(
                    target=process_frames_3, daemon=True
                )
                app_state.processing_thread_3.start()
        else:
            app_state.stop_flag_3 = True

        # Gestione Feed 4
        if status.get("feed_4"):
            app_state.stop_flag_4 = False
            if (
                app_state.processing_thread_4 is None
                or not app_state.processing_thread_4.is_alive()
            ):
                app_state.processing_thread_4 = threading.Thread(
                    target=process_frames_4, daemon=True
                )
                app_state.processing_thread_4.start()
        else:
            app_state.stop_flag_4 = True

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)


# Funzione per creare indicatore stato
def create_status_indicator(is_active):
    color = "green" if is_active else "red"
    return f"<span style='color: {color}; font-size: 20px;'>●</span>"


# Creazione dei placeholder per i contatori e gli indicatori
with col1:
    st.header("Feed Live 1")
    feed1_header_col, feed1_status_col = st.columns([3, 1])
    with feed1_status_col:
        feed1_status = st.empty()
    counter1_total = st.empty()
    counter1_standing = st.empty()
    counter1_fallen = st.empty()
    image_placeholder = st.empty()

with col2:
    st.header("Feed Live 2")
    feed2_header_col, feed2_status_col = st.columns([3, 1])
    with feed2_status_col:
        feed2_status = st.empty()
    counter2_total = st.empty()
    counter2_standing = st.empty()
    counter2_fallen = st.empty()
    image_placeholder2 = st.empty()

with col3:
    st.header("Feed Live 3")
    feed3_header_col, feed3_status_col = st.columns([3, 1])
    with feed3_status_col:
        feed3_status = st.empty()
    counter3_total = st.empty()
    counter3_standing = st.empty()
    counter3_fallen = st.empty()
    image_placeholder3 = st.empty()

with col4:
    st.header("Feed Live 4")
    feed4_header_col, feed4_status_col = st.columns([3, 1])
    with feed4_status_col:
        feed4_status = st.empty()
    counter4_total = st.empty()
    counter4_standing = st.empty()
    counter4_fallen = st.empty()
    image_placeholder4 = st.empty()

# Pulsanti per avviare/fermare tutti i feed
col_controls = st.columns(2)
with col_controls[0]:
    if st.button(
        "🟢 Avvia tutti i feed", help="Avvia tutti e 4 i feed contemporaneamente"
    ):
        start_all_feeds()
with col_controls[1]:
    if st.button(
        "🔴 Ferma tutti i feed", help="Ferma tutti e 4 i feed contemporaneamente"
    ):
        stop_all_feeds()

# Controlli individuali per avviare/fermare i feed
col5, col6, col7, col8 = st.columns(4)

with col5:
    if st.button("Avvia Feed 1"):
        if (
            app_state.processing_thread is None
            or not app_state.processing_thread.is_alive()
        ):
            app_state.stop_flag = False
            app_state.processing_thread = threading.Thread(
                target=process_frames, daemon=True
            )
            app_state.processing_thread.start()
            st.success("Feed 1 avviato!")
            status = read_feed_status()
            status["feed_1"] = True
            write_feed_status(status)

    if st.button("Ferma Feed 1"):
        app_state.stop_flag = True
        st.warning("Feed 1 fermato.")
        status = read_feed_status()
        status["feed_1"] = False
        write_feed_status(status)

with col6:
    if st.button("Avvia Feed 2"):
        if (
            app_state.processing_thread_2 is None
            or not app_state.processing_thread_2.is_alive()
        ):
            app_state.stop_flag_2 = False
            app_state.processing_thread_2 = threading.Thread(
                target=process_frames_2, daemon=True
            )
            app_state.processing_thread_2.start()
            st.success("Feed 2 avviato!")
            status = read_feed_status()
            status["feed_2"] = True
            write_feed_status(status)

    if st.button("Ferma Feed 2"):
        app_state.stop_flag_2 = True
        st.warning("Feed 2 fermato.")
        status = read_feed_status()
        status["feed_2"] = False
        write_feed_status(status)

with col7:
    if st.button("Avvia Feed 3"):
        if (
            app_state.processing_thread_3 is None
            or not app_state.processing_thread_3.is_alive()
        ):
            app_state.stop_flag_3 = False
            app_state.processing_thread_3 = threading.Thread(
                target=process_frames_3, daemon=True
            )
            app_state.processing_thread_3.start()
            st.success("Feed 3 avviato!")
            status = read_feed_status()
            status["feed_3"] = True
            write_feed_status(status)

    if st.button("Ferma Feed 3"):
        app_state.stop_flag_3 = True
        st.warning("Feed 3 fermato.")
        status = read_feed_status()
        status["feed_3"] = False
        write_feed_status(status)

with col8:
    if st.button("Avvia Feed 4"):
        if (
            app_state.processing_thread_4 is None
            or not app_state.processing_thread_4.is_alive()
        ):
            app_state.stop_flag_4 = False
            app_state.processing_thread_4 = threading.Thread(
                target=process_frames_4, daemon=True
            )
            app_state.processing_thread_4.start()
            st.success("Feed 4 avviato!")
            status = read_feed_status()
            status["feed_4"] = True
            write_feed_status(status)

    if st.button("Ferma Feed 4"):
        app_state.stop_flag_4 = True
        st.warning("Feed 4 fermato.")
        status = read_feed_status()
        status["feed_4"] = False
        write_feed_status(status)


# Loop per aggiornare interfaccia
def update_interface():
    while True:
        # Calcola totali complessivi
        total_people = (
            app_state.people_counts["total"]
            + app_state.people_counts_2["total"]
            + app_state.people_counts_3["total"]
            + app_state.people_counts_4["total"]
        )
        total_standing = (
            app_state.people_counts["standing"]
            + app_state.people_counts_2["standing"]
            + app_state.people_counts_3["standing"]
            + app_state.people_counts_4["standing"]
        )
        total_fallen = (
            app_state.people_counts["fallen"]
            + app_state.people_counts_2["fallen"]
            + app_state.people_counts_3["fallen"]
            + app_state.people_counts_4["fallen"]
        )

        # Aggiorna contatore totale
        total_counter.markdown(f"""
        **Totale Persone Rilevate:** {total_people}  
        **Totale In Piedi:** {total_standing}  
        **Totale A Terra:** {total_fallen}
        """)

        # Controlla stati dei thread
        feed1_active = (
            app_state.processing_thread is not None
            and app_state.processing_thread.is_alive()
            and not app_state.stop_flag
        )
        feed2_active = (
            app_state.processing_thread_2 is not None
            and app_state.processing_thread_2.is_alive()
            and not app_state.stop_flag_2
        )
        feed3_active = (
            app_state.processing_thread_3 is not None
            and app_state.processing_thread_3.is_alive()
            and not app_state.stop_flag_3
        )
        feed4_active = (
            app_state.processing_thread_4 is not None
            and app_state.processing_thread_4.is_alive()
            and not app_state.stop_flag_4
        )

        # Aggiorna indicatori stato
        feed1_status.markdown(
            create_status_indicator(feed1_active), unsafe_allow_html=True
        )
        feed2_status.markdown(
            create_status_indicator(feed2_active), unsafe_allow_html=True
        )
        feed3_status.markdown(
            create_status_indicator(feed3_active), unsafe_allow_html=True
        )
        feed4_status.markdown(
            create_status_indicator(feed4_active), unsafe_allow_html=True
        )

        # Aggiorna contatori Feed 1
        counter1_total.markdown(
            f"**Persone rilevate:** {app_state.people_counts['total']}"
        )
        counter1_standing.markdown(
            f"**In piedi:** {app_state.people_counts['standing']}"
        )
        counter1_fallen.markdown(f"**A terra:** {app_state.people_counts['fallen']}")

        # Aggiorna contatori Feed 2
        counter2_total.markdown(
            f"**Persone rilevate:** {app_state.people_counts_2['total']}"
        )
        counter2_standing.markdown(
            f"**In piedi:** {app_state.people_counts_2['standing']}"
        )
        counter2_fallen.markdown(f"**A terra:** {app_state.people_counts_2['fallen']}")

        # Aggiorna contatori Feed 3
        counter3_total.markdown(
            f"**Persone rilevate:** {app_state.people_counts_3['total']}"
        )
        counter3_standing.markdown(
            f"**In piedi:** {app_state.people_counts_3['standing']}"
        )
        counter3_fallen.markdown(f"**A terra:** {app_state.people_counts_3['fallen']}")

        # Aggiorna contatori Feed 4
        counter4_total.markdown(
            f"**Persone rilevate:** {app_state.people_counts_4['total']}"
        )
        counter4_standing.markdown(
            f"**In piedi:** {app_state.people_counts_4['standing']}"
        )
        counter4_fallen.markdown(f"**A terra:** {app_state.people_counts_4['fallen']}")

        # Aggiorna immagini solo se show_images è attivo
        if st.session_state.show_images:
            if app_state.last_annotated_frame is not None:
                frame_rgb = cv2.cvtColor(
                    app_state.last_annotated_frame, cv2.COLOR_BGR2RGB
                )
                image_placeholder.image(
                    frame_rgb, channels="RGB", use_container_width=True
                )

            if app_state.last_annotated_frame_2 is not None:
                frame_rgb_2 = cv2.cvtColor(
                    app_state.last_annotated_frame_2, cv2.COLOR_BGR2RGB
                )
                image_placeholder2.image(
                    frame_rgb_2, channels="RGB", use_container_width=True
                )

            if app_state.last_annotated_frame_3 is not None:
                frame_rgb_3 = cv2.cvtColor(
                    app_state.last_annotated_frame_3, cv2.COLOR_BGR2RGB
                )
                image_placeholder3.image(
                    frame_rgb_3, channels="RGB", use_container_width=True
                )

            if app_state.last_annotated_frame_4 is not None:
                frame_rgb_4 = cv2.cvtColor(
                    app_state.last_annotated_frame_4, cv2.COLOR_BGR2RGB
                )
                image_placeholder4.image(
                    frame_rgb_4, channels="RGB", use_container_width=True
                )
        else:
            # Se le immagini sono disabilitate, mostra un messaggio
            image_placeholder.markdown("**Visualizzazione immagini disattivata**")
            image_placeholder2.markdown("**Visualizzazione immagini disattivata**")
            image_placeholder3.markdown("**Visualizzazione immagini disattivata**")
            image_placeholder4.markdown("**Visualizzazione immagini disattivata**")

        # Gestione messaggi
        while not message_queue.empty():
            msg_type, msg_content = message_queue.get()
            if msg_type == "success":
                status_text.success(msg_content)
            elif msg_type == "error":
                status_text.error(msg_content)
            elif msg_type == "info":
                status_text.info(msg_content)
            elif msg_type == "warning":
                status_text.warning(msg_content)

        time.sleep(0.1)


if __name__ == "__main__":
    update_interface()
