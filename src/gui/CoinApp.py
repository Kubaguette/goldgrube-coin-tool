import sys
import os
import shutil 
import time   
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QProgressBar, QFrame, QMessageBox, QSizePolicy, 
                             QScrollArea, QStackedWidget, QLineEdit) # QLineEdit für die Suche
import cv2
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent

# Pfade setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# --- IMPORTS ---
try:
    from src.ai.inference import CoinPredictor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ KI-Modul nicht gefunden.")

try:
    from src.utils.coin_extractor import process_single_image
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    print("⚠️ Extractor-Modul nicht gefunden.")

try:
    from src.manager.dublicate_detector import DuplicateDetector
    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    DUPLICATE_DETECTOR_AVAILABLE = False
    print("⚠️ DuplicateDetector-Modul nicht gefunden.")

# --- STYLING (UI UPDATE) ---
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }

/* Sidebar */
#sidebar { background-color: #1e1e1e; border-right: 1px solid #333; }
#sidebarLabel { color: #4CAF50; font-family: 'Segoe UI'; font-weight: bold; font-size: 22px; }
QPushButton#sidebarBtn {
    background-color: transparent; color: #ccc; text-align: left; padding: 15px; border: none; font-size: 16px;
}
QPushButton#sidebarBtn:hover { background-color: #333; color: white; border-left: 4px solid #4CAF50; }

/* Content */
QLabel { color: white; font-family: 'Segoe UI', sans-serif; }

/* 1. Haupt-Ergebnis Karte (Der Gewinner) */
QFrame#card { 
    background-color: #222; 
    border: 2px solid #4CAF50; 
    border-radius: 12px; 
    padding: 20px; 
}

/* 2. Detail Karten (Vorder/Rückseite) */
QFrame#subCard { 
    background-color: #323232; 
    border: 1px solid #444; 
    border-radius: 8px; 
    padding: 10px; 
}

/* 3. Alternative Kandidaten Karte (NEU!) */
QFrame#altCard { 
    background-color: #323232; 
    border-radius: 10px; 
    padding: 15px; 
    border: 1px solid #444;
}

/* Dropzones */
QLabel#dropZone { background-color: #1e1e1e; border-radius: 15px; border: 2px dashed #444; }
QLabel#dropZone:hover { border-color: #666; }

/* Action Buttons */
QPushButton#actionBtn { 
    background-color: #1f6aa5; color: white; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 18px;
}
QPushButton#actionBtn:hover { background-color: #257cc0; }
QPushButton#actionBtn:disabled { background-color: #555; color: #888; }

/* RESET Button */
QPushButton#resetBtn { 
    background-color: #d32f2f; color: white; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 18px; 
}
QPushButton#resetBtn:hover { background-color: #b71c1c; }

/* Progress Bar */
QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; color: white; height: 30px; }
QProgressBar::chunk { background-color: #4CAF50; }

/* Suchfeld */
QLineEdit { 
    background-color: #333; 
    color: white; 
    border: 1px solid #555; 
    border-radius: 5px; 
    padding: 8px; 
    font-size: 16px; 
}
QLineEdit:focus { border: 1px solid #4CAF50; }

/* Popups */
QMessageBox { background-color: #2b2b2b; border: 1px solid #555; }
QMessageBox QLabel { color: white; background-color: transparent; }
QMessageBox QPushButton { background-color: #1f6aa5; color: white; padding: 6px 15px; border-radius: 4px; font-weight: bold; }
QMessageBox QPushButton:hover { background-color: #257cc0; }
"""

# --- HELPER: Sortiert Dictionary zu Top-Liste ---
def get_top_k(probs_dict, k=None):
    if not probs_dict: return []
    sorted_items = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
    if k: return sorted_items[:k]
    return sorted_items

# --- DRAG & DROP LABEL KLASSE ---
class DraggableLabel(QLabel):
    file_dropped = pyqtSignal(str, str) # Pfad, Side_ID

    def __init__(self, side_id, text, parent=None):
        super().__init__(text, parent)
        self.side_id = side_id
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("dropZone")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.setStyleSheet("background-color: #252525; border-radius: 15px; border: 2px dashed #4CAF50;")
                event.accept()
            else: event.ignore()
        else: event.ignore()

    def dragLeaveEvent(self, event): self.setStyleSheet("") 
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("") 
        urls = event.mimeData().urls()
        if urls: self.file_dropped.emit(urls[0].toLocalFile(), self.side_id)

# --- WORKER: KI ---
class DualAIWorker(QThread):
    finished = pyqtSignal(dict) 

    def __init__(self, predictor, path_front, path_back):
        super().__init__()
        self.predictor = predictor
        self.path_front = path_front
        self.path_back = path_back

    def run(self):
        if not self.predictor:
            self.finished.emit({})
            return

        dict_front = {}
        dict_back = {}

        if self.path_front and os.path.exists(self.path_front): 
            dict_front = self.predictor.predict(self.path_front)
        
        if self.path_back and os.path.exists(self.path_back): 
            dict_back = self.predictor.predict(self.path_back)

        combined_dict = {}
        all_keys = set(dict_front.keys()) | set(dict_back.keys())
        
        divisor = 0
        if dict_front: divisor += 1
        if dict_back: divisor += 1
        
        if divisor > 0:
            for key in all_keys:
                score_a = dict_front.get(key, 0.0)
                score_b = dict_back.get(key, 0.0)
                combined_dict[key] = (score_a + score_b) / divisor
        
        final_combined_list = get_top_k(combined_dict, 5)
        final_front_list = get_top_k(dict_front, 1)
        final_back_list = get_top_k(dict_back, 1)

        self.finished.emit({
            'combined': final_combined_list,
            'front': final_front_list,
            'back': final_back_list
        })

# --- WORKER: CROPPER ---
class CropperWorker(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    def __init__(self, raw_image_path, side_id):
        super().__init__()
        self.raw_path = raw_image_path
        self.side_id = side_id
    def run(self):
        try:
            if not CROPPER_AVAILABLE: raise Exception("Extractor Modul fehlt.")
            cropped_path = process_single_image(self.raw_path)
            self.finished.emit(cropped_path, self.side_id)
        except Exception as e: self.error.emit(str(e))

# --- WORKER: DUPLICATE CHECK ---
class DuplicateCheckWorker(QThread):
    found = pyqtSignal(dict, str)
    not_found = pyqtSignal()

    def __init__(self, detector, path_front, path_back):
        super().__init__()
        self.detector = detector
        self.path_front = path_front
        self.path_back = path_back

    def run(self):
        if not self.detector:
            self.not_found.emit()
            return
        
        # Helper function
        def check(path):
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None: return self.detector.find_match(img)
            return None

        match = check(self.path_front)
        if match: self.found.emit(match, "Vorderseite"); return

        match = check(self.path_back)
        if match: self.found.emit(match, "Rückseite"); return

        self.not_found.emit()

# --- WORKER: LIBRARY LOADER ---
class LibraryLoaderWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def run(self):
        csv_path = os.path.join(self.base_path, "data", "database", "master_coin_list.csv")
        if not os.path.exists(csv_path):
            self.error.emit(f"Datenbank nicht gefunden: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
        except Exception as e:
            self.error.emit(f"CSV Fehler: {e}")
            return

        # Cache für Fuzzy-Suche (CoinArchives)
        fuzzy_image_map = {}
        base_img_path = os.path.join(self.base_path, 'data', 'images_raw')
        if os.path.exists(base_img_path):
            try:
                for f in os.listdir(base_img_path):
                    if "_image" in f:
                        idx = f.find("_image")
                        if idx > 0:
                            coin_id = f[:idx]
                            if coin_id not in fuzzy_image_map: fuzzy_image_map[coin_id] = []
                            fuzzy_image_map[coin_id].append(os.path.join(base_img_path, f))
                for k in fuzzy_image_map: fuzzy_image_map[k].sort()
            except: pass

        results = []
        total = len(df)
        extensions = ['', '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']
        suffixes = ["", "_av", "_rv", "_av_rv", "_obv", "_rev"]

        for index, row in df.iterrows():
            try:
                # Daten extrahieren
                filename_no_ext = str(row.iloc[1]).strip()
                if filename_no_ext.endswith('.0'): filename_no_ext = filename_no_ext[:-2]
                
                # Bilder suchen (Heavy Lifting hier im Thread!)
                found_images = []
                base_candidates = [filename_no_ext]
                if "NumCafe" in filename_no_ext: base_candidates.append(filename_no_ext.replace("NumCafe", "NumCaf"))
                elif "NumCaf" in filename_no_ext: base_candidates.append(filename_no_ext.replace("NumCaf", "NumCafe"))

                if filename_no_ext and filename_no_ext.lower() != 'nan':
                    for base in base_candidates:
                        for suff in suffixes:
                            pat = base + suff
                            for ext in extensions:
                                cand = os.path.join(base_img_path, pat + ext)
                                if os.path.exists(cand) and cand not in found_images: found_images.append(cand); break
                        if base in fuzzy_image_map:
                            for fp in fuzzy_image_map[base]: 
                                if fp not in found_images: found_images.append(fp)

                # Ergebnis speichern
                results.append({
                    'desc': str(row.iloc[3]).strip(),
                    'tech_id': str(int(row.iloc[4])) if isinstance(row.iloc[4], float) and row.iloc[4].is_integer() else str(row.iloc[4]).strip(),
                    'source': str(row.iloc[0]).strip(),
                    'images': found_images,
                    'filename_debug': filename_no_ext
                })
            except: continue
            
            if index % 20 == 0: self.progress.emit(int((index / total) * 100))

        self.finished.emit(results)

# --- HAUPTFENSTER ---
class CoinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goldgrube - Angelikas & Jakubs KI-Tool")
        self.resize(1600, 1000)
        self.setStyleSheet(STYLESHEET)

        self.path_front = None
        self.path_back = None
        self.predictor = None
        if AI_AVAILABLE: self.predictor = CoinPredictor()
        
        self.duplicate_detector = None
        if DUPLICATE_DETECTOR_AVAILABLE:
            print("Initialisiere Duplikat-Erkennung...")
            self.duplicate_detector = DuplicateDetector()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.setup_sidebar(main_layout)

        # --- Zentrales Inhalts-Widget (Seiten-Manager) ---
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Seite 1: Scanner
        scanner_page = QWidget()
        self.setup_scanner_ui(scanner_page)
        self.stacked_widget.addWidget(scanner_page)

        # Seite 2: Bibliothek
        library_page = QWidget()
        self.setup_library_ui(library_page)
        self.stacked_widget.addWidget(library_page)

        self.library_loaded = False

    def setup_scanner_ui(self, scanner_page):
        content_layout = QHBoxLayout(scanner_page)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(30)
        
        # --- LEFT PANEL ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15) 
        
        lbl_title = QLabel("Münz-Scanner")
        lbl_title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        left_layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignCenter)

        drop_container = QWidget()
        drop_layout = QHBoxLayout(drop_container)
        drop_layout.setSpacing(15)
        drop_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_front = DraggableLabel("front", "Vorderseite\n(Hier Bild ziehen)")
        self.lbl_front.file_dropped.connect(self.start_crop_process)
        drop_layout.addWidget(self.lbl_front)

        self.lbl_back = DraggableLabel("back", "Rückseite\n(Hier Bild ziehen)")
        self.lbl_back.file_dropped.connect(self.start_crop_process)
        drop_layout.addWidget(self.lbl_back)
        
        left_layout.addWidget(drop_container, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: orange; font-size: 16px;")
        left_layout.addWidget(self.lbl_status)

        btn_open = QPushButton("📂 MANUELL AUSWÄHLEN")
        btn_open.setObjectName("actionBtn")
        btn_open.clicked.connect(self.open_file_dialog_manual)
        left_layout.addWidget(btn_open)

        self.btn_run = QPushButton("⚡ ANALYSIEREN")
        self.btn_run.setObjectName("actionBtn")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        left_layout.addWidget(self.btn_run)

        # NEU: Ladebalken (Initial versteckt)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Marquee (Unendlich)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        btn_reset = QPushButton("🔄 RESET")
        btn_reset.setObjectName("resetBtn")
        btn_reset.clicked.connect(self.reset_app)
        left_layout.addWidget(btn_reset)

        content_layout.addWidget(left_panel, 5) 

        # --- RIGHT PANEL ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_res_title = QLabel("Ergebnis")
        lbl_res_title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        right_layout.addWidget(lbl_res_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        self.res_container = QWidget()
        self.res_layout = QVBoxLayout(self.res_container)
        self.res_layout.setSpacing(15)
        self.res_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.res_container)
        right_layout.addWidget(scroll)

        self.lbl_placeholder = QLabel("👈 Lade mindestens eine Seite,\num zu starten.")
        self.lbl_placeholder.setStyleSheet("color: #777; font-size: 20px; margin-top: 50px;")
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.res_layout.addWidget(self.lbl_placeholder)

        content_layout.addWidget(right_panel, 4)

    def setup_library_ui(self, library_page):
        self.library_layout = QVBoxLayout(library_page)
        self.library_layout.setContentsMargins(30, 30, 30, 30)
        
        lbl_title = QLabel("Münz-Bibliothek")
        lbl_title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.library_layout.addWidget(lbl_title)

        # NEU: Suchleiste
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Bibliothek durchsuchen (Name, ID, Quelle, Beschreibung)...")
        self.search_bar.textChanged.connect(self.filter_library)
        self.library_layout.addWidget(self.search_bar)

        # ScrollArea für die Listeneinträge
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background: transparent; border: none;")
        self.library_layout.addWidget(scroll_area)

        # Container-Widget im Scroll-Bereich
        self.library_content_widget = QWidget()
        self.library_list_layout = QVBoxLayout(self.library_content_widget)
        self.library_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.library_list_layout.setSpacing(15)
        scroll_area.setWidget(self.library_content_widget)

        self.lib_placeholder = QLabel("Lade Bibliothek...")
        self.lib_placeholder.setStyleSheet("color: #777; font-size: 20px; margin-top: 50px;")
        self.lib_placeholder.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.library_list_layout.addWidget(self.lib_placeholder)
        
        # Ladebalken für Bibliothek
        self.lib_progress = QProgressBar()
        self.lib_progress.hide()
        self.library_layout.addWidget(self.lib_progress)

    def filter_library(self, text):
        """Filtert die Bibliothek in Echtzeit."""
        query = text.lower().strip()
        # Iteriere durch alle Widgets im Layout
        for i in range(self.library_list_layout.count()):
            widget_item = self.library_list_layout.itemAt(i)
            if not widget_item: continue
            
            widget = widget_item.widget()
            if widget and hasattr(widget, 'search_tags'):
                if query in widget.search_tags:
                    widget.show()
                else:
                    widget.hide()

    def load_library_data(self):
        self.lib_placeholder.setText("⏳ Lade Datenbank und suche Bilder...")
        self.lib_placeholder.show()
        self.lib_progress.setValue(0)
        self.lib_progress.show()
        
        # Worker starten (parent_dir ist global verfügbar)
        self.lib_worker = LibraryLoaderWorker(parent_dir)
        self.lib_worker.progress.connect(self.lib_progress.setValue)
        self.lib_worker.finished.connect(self.on_library_loaded)
        self.lib_worker.error.connect(lambda e: self.lib_placeholder.setText(f"Fehler: {e}"))
        self.lib_worker.start()

    def on_library_loaded(self, data):
        self.lib_progress.hide()
        self.lib_placeholder.hide()
        
        # Widgets erstellen (jetzt schnell, da Daten vorbereitet)
        for item in data:
            w = self.create_coin_widget(item)
            self.library_list_layout.addWidget(w)
            
        self.library_loaded = True

    def create_coin_widget(self, item_data):
        # Haupt-Container für einen Eintrag
        entry_frame = QFrame()
        entry_frame.setObjectName("subCard") # Wiederverwendung des Stils
        entry_layout = QHBoxLayout(entry_frame)
        entry_layout.setSpacing(20)

        # --- BILDER CONTAINER ---
        img_container = QWidget()
        img_layout = QHBoxLayout(img_container)
        img_layout.setContentsMargins(0,0,0,0)
        img_layout.setSpacing(10)

        found_images = item_data['images']
        desc = item_data['desc']
        tech_id = item_data['tech_id']
        source = item_data['source']

        if not found_images: 
            lbl = QLabel("Bild\nfehlt")
            lbl.setFixedSize(120, 120)
            lbl.setStyleSheet("border: 1px solid #555; border-radius: 60px; background-color: #2a2a2a; color: #777;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_layout.addWidget(lbl)
        else:
            # Bilder anzeigen (Max 2)
            for img_p in found_images[:2]:
                lbl = QLabel()
                lbl.setFixedSize(120, 120)
                lbl.setStyleSheet("border: 1px solid #555; border-radius: 10px; background-color: #2a2a2a;")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pix = QPixmap(img_p)
                lbl.setPixmap(pix.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                img_layout.addWidget(lbl)

        entry_layout.addWidget(img_container)

        # --- TEXT-INFOS ---
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)

        desc_label = QLabel(desc)
        desc_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)

        tech_id_label = QLabel(f"Tech-ID: {tech_id}")
        tech_id_label.setStyleSheet("color: #aaa; font-size: 14px;")
        info_layout.addWidget(tech_id_label)

        source_label = QLabel(f"Quelle: {source}")
        source_label.setStyleSheet("color: #aaa; font-size: 14px;")
        info_layout.addWidget(source_label)

        info_layout.addStretch()
        entry_layout.addLayout(info_layout, 1) # Textbereich soll sich ausdehnen

        # --- TAGS FÜR SUCHE SPEICHERN ---
        # Alle relevanten Texte kleingeschrieben in eine Variable packen
        entry_frame.search_tags = f"{desc} {tech_id} {source}".lower()

        return entry_frame

    def setup_sidebar(self, layout):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(20, 40, 20, 20)
        sb_layout.setSpacing(10)
        lbl_brand = QLabel("Goldgrube\nKI-Tool")
        lbl_brand.setObjectName("sidebarLabel")
        sb_layout.addWidget(lbl_brand)
        sb_layout.addSpacing(30)
        self.create_sidebar_btn(sb_layout, "🏠  Startseite", self.open_scanner)
        self.create_sidebar_btn(sb_layout, "🕸️  Webscraper", self.open_webscraper)
        self.create_sidebar_btn(sb_layout, "📚  Bibliothek", self.open_library)
        self.create_sidebar_btn(sb_layout, "ℹ️  Über", self.show_about)
        sb_layout.addStretch()
        self.create_sidebar_btn(sb_layout, "🚪  Schließen", self.close)
        layout.addWidget(sidebar)

    def create_sidebar_btn(self, layout, text, func):
        btn = QPushButton(text)
        btn.setObjectName("sidebarBtn")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(func)
        layout.addWidget(btn)

    def open_scanner(self):
        self.stacked_widget.setCurrentIndex(0)

    def open_library(self):
        self.stacked_widget.setCurrentIndex(1)
        if not self.library_loaded:
            self.load_library_data()

    def show_message(self, title, text, icon=QMessageBox.Icon.Information):
        msg = QMessageBox(self)
        msg.setWindowTitle(title); msg.setText(text); msg.setIcon(icon)
        msg.setStyleSheet(STYLESHEET)
        msg.exec()

    def show_about(self):
        txt_path = os.path.join(current_dir, "Text", "ueber.txt")
        content = "Goldgrube v1.0\n(Textdatei nicht gefunden)"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f: content = f.read()
            except Exception as e: content = f"Fehler:\n{e}"
        self.show_message("Über Goldgrube", content)

    def open_webscraper(self):
        self.show_message("Webscraper", "Phase 4 - Coming Soon!")

    def reset_app(self):
        self.path_front = None
        self.path_back = None
        self.lbl_front.clear(); self.lbl_front.setText("Vorderseite\n(Hier Bild ziehen)")
        self.lbl_back.clear(); self.lbl_back.setText("Rückseite\n(Hier Bild ziehen)")
        self.lbl_status.setText("")
        self.btn_run.setEnabled(False); self.btn_run.setText("⚡ ANALYSIEREN"); self.btn_run.setStyleSheet("background-color: #555; color: #888; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 18px;")
        self.btn_run.show(); self.progress_bar.hide()
        self.clear_results()
        self.lbl_placeholder = QLabel("👈 Lade mindestens eine Seite,\num zu starten."); self.lbl_placeholder.setStyleSheet("color: #777; font-size: 20px; margin-top: 50px;"); self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.res_layout.addWidget(self.lbl_placeholder)

    def open_file_dialog_manual(self):
        msg = QMessageBox(self); msg.setWindowTitle("Bild wählen"); msg.setText("Welche Seite?"); msg.setStyleSheet(STYLESHEET)
        bf = msg.addButton("Vorderseite", QMessageBox.ButtonRole.ActionRole)
        bb = msg.addButton("Rückseite", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("Abbrechen", QMessageBox.ButtonRole.RejectRole); msg.exec()
        t = "front" if msg.clickedButton() == bf else "back" if msg.clickedButton() == bb else None
        if t: 
            fp, _ = QFileDialog.getOpenFileName(self, "Bild", "", "Images (*.png *.jpg *.jpeg)")
            if fp: self.start_crop_process(fp, t)

    def start_crop_process(self, file_path, side_id):
        t_lbl = self.lbl_front if side_id == "front" else self.lbl_back
        t_lbl.setText("✂️ Schneide..."); self.lbl_status.setText(f"Bearbeite {side_id}...")
        self.btn_run.setEnabled(False)
        self.cropper = CropperWorker(file_path, side_id)
        self.cropper.finished.connect(self.handle_cropped_image)
        self.cropper.error.connect(self.handle_crop_error)
        self.cropper.start()

    def handle_cropped_image(self, cropped_path, side_id):
        # Datei verschieben um Overwriting zu verhindern
        new_fn = f"crop_{side_id}_{int(time.time())}.jpg"
        new_p = os.path.join(os.path.dirname(cropped_path), new_fn)
        try:
            shutil.move(cropped_path, new_p)
            final_p = new_p
        except: final_p = cropped_path

        if side_id == "front": self.path_front = final_p
        else: self.path_back = final_p

        self.lbl_status.setText(f"✅ {side_id} bereit!")
        t_lbl = self.lbl_front if side_id == "front" else self.lbl_back
        pix = QPixmap(final_p); t_lbl.setPixmap(pix.scaled(t_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if self.path_front or self.path_back:
            self.btn_run.setEnabled(True)
            self.btn_run.setStyleSheet("QPushButton#actionBtn { background-color: #2E8B57; color: white; border-radius: 5px; padding: 12px; font-weight: bold; font-size: 18px; } QPushButton#actionBtn:hover { background-color: #3aa869; }")

    def handle_crop_error(self, error_msg):
        self.lbl_status.setText("Fehler"); self.show_message("Fehler", error_msg, QMessageBox.Icon.Critical)

    def run_analysis(self):
        # UI Update: Button weg, Ladebalken her
        self.btn_run.hide()
        self.progress_bar.show()
        self.lbl_status.setText("🔍 Prüfe Datenbank...")
        self.clear_results()

        # 1. Duplikat-Check im Hintergrund starten
        self.dup_worker = DuplicateCheckWorker(self.duplicate_detector, self.path_front, self.path_back)
        self.dup_worker.found.connect(self.handle_duplicate_found_worker)
        self.dup_worker.not_found.connect(self.start_ai_analysis)
        self.dup_worker.start()

    def start_ai_analysis(self):
        self.lbl_status.setText("🧠 KI analysiert...")
        self.ai_worker = DualAIWorker(self.predictor, self.path_front, self.path_back)
        self.ai_worker.finished.connect(self.handle_ai_results)
        self.ai_worker.start()

    def handle_duplicate_found_worker(self, match, side):
        self.handle_duplicate_found(match, side)

    def handle_ai_results(self, results):
        self.restore_ui()
        self.lbl_status.setText("✅ Fertig")
        if not results: self.res_layout.addWidget(QLabel("Fehler")); return

        combined = results.get('combined', [])
        front_res = results.get('front', [])
        back_res = results.get('back', [])

        # 1. HAUPT-ERGEBNIS (Der Gewinner)
        if combined:
            top_name, top_conf = combined[0]
            main_card = QFrame(); main_card.setObjectName("card"); mc_l = QVBoxLayout(main_card)
            
            hdr = QLabel("GESAMTERGEBNIS (Kombiniert)"); hdr.setStyleSheet("color: #888; font-weight: bold; font-size: 14px;")
            mc_l.addWidget(hdr)
            
            n = QLabel(f"🥇 {top_name}"); n.setFont(QFont("Arial", 28, QFont.Weight.Bold)); n.setStyleSheet("color: #4CAF50; border: none;"); n.setWordWrap(True)
            mc_l.addWidget(n)
            
            mc_l.addWidget(QLabel(f"Sicherheit: {top_conf*100:.1f}%"))
            pb = QProgressBar(); pb.setValue(int(top_conf * 100)); mc_l.addWidget(pb)
            self.res_layout.addWidget(main_card)

        # 2. DETAILS (Nebeneinander)
        self.res_layout.addSpacing(15)
        det_w = QWidget(); det_l = QHBoxLayout(det_w); det_l.setContentsMargins(0,0,0,0); det_l.setSpacing(10)

        def mk_mini(t, r):
            c = QFrame(); c.setObjectName("subCard"); cl = QVBoxLayout(c); cl.setContentsMargins(10,10,10,10)
            cl.addWidget(QLabel(t))
            if r:
                nn, cc = r[0]
                nl = QLabel(nn); nl.setStyleSheet("font-size: 14px; font-weight: bold; color: white;"); nl.setWordWrap(True); cl.addWidget(nl)
                cl.addWidget(QLabel(f"{cc*100:.1f}%"))
            else: cl.addWidget(QLabel("-"))
            cl.addStretch(); return c

        if front_res: det_l.addWidget(mk_mini("Vorderseite", front_res))
        if back_res: det_l.addWidget(mk_mini("Rückseite", back_res))
        self.res_layout.addWidget(det_w)

        # 3. ALTERNATIVE KANDIDATEN (NEUES UI: Eigene Karte)
        if len(combined) > 1:
            self.res_layout.addSpacing(15)
            
            # Die neue Box (Karte)
            alt_card = QFrame()
            alt_card.setObjectName("altCard") # Nutzt den neuen CSS Style
            alt_layout = QVBoxLayout(alt_card)
            
            lbl_alt = QLabel("Weitere Möglichkeiten:")
            lbl_alt.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            lbl_alt.setStyleSheet("color: #ddd; border: none; margin-bottom: 10px;")
            alt_layout.addWidget(lbl_alt)
            
            # Zeilen einfügen
            for name, conf in combined[1:4]: # Top 2-4
                row = QWidget(); row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0,0,0,0)
                
                lbl_n = QLabel(name)
                lbl_n.setFont(QFont("Arial", 14))
                lbl_n.setStyleSheet("border: none; color: white;")
                
                lbl_c = QLabel(f"{conf*100:.1f}%")
                lbl_c.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                lbl_c.setStyleSheet("color: #888; border: none;")
                
                row_layout.addWidget(lbl_n)
                row_layout.addStretch()
                row_layout.addWidget(lbl_c)
                alt_layout.addWidget(row)
            
            self.res_layout.addWidget(alt_card)
        
        self.res_layout.addStretch()

    def clear_results(self):
        while self.res_layout.count():
            item = self.res_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def restore_ui(self):
        self.progress_bar.hide()
        self.btn_run.show()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("⚡ ANALYSIEREN")

    def handle_duplicate_found(self, match, side):
        """Zeigt Popup und Ergebnis an."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Münze bekannt")
        msg.setText(f"ACHTUNG: Münze ({side}) ist bereits bekannt.")
        msg.setInformativeText(f"Klasse: {match['class']}\nDatei: {match['filename']}\n(Score: {match['score']})")
        msg.exec()
        
        # Fake-Ergebnis für die GUI (100%)
        fake_results = {'combined': [(match['class'], 1.0)], 'front': [], 'back': []}
        self.handle_ai_results(fake_results)

if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    window = CoinApp()
    window.show()
    sys.exit(app.exec())