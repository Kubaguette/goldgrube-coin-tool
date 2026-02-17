import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm

class DuplicateDetector:
    def __init__(self, dataset_dir="data/dataset_clean", cache_file="data/database/orb_index.pkl"):
        self.base_dir = Path(os.getcwd())
        self.dataset_dir = self.base_dir / dataset_dir
        self.cache_file = self.base_dir / cache_file
        
        # WICHTIG: 2000 Features sind gut, aber wir müssen streng filtern
        self.orb = cv2.ORB_create(nfeatures=2500)
        
        # Hamming Distanz ist korrekt für ORB
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.index = [] 
        self.load_index()

    def load_index(self):
        """Lädt den Index oder baut ihn neu, falls nicht vorhanden."""
        if self.cache_file.exists():
            try:
                print(f"[DuplicateDetector] Lade Index: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    self.index = pickle.load(f)
                print(f"[DuplicateDetector] {len(self.index)} Münzen geladen.")
            except Exception as e:
                print(f"[DuplicateDetector] Fehler beim Laden: {e}. Baue neu...")
                self.build_index()
        else:
            print("[DuplicateDetector] Kein Index gefunden. Baue neu...")
            self.build_index()

    def build_index(self):
        """Scannt den Dataset-Ordner und erstellt Fingerabdrücke."""
        self.index = []
        if not self.dataset_dir.exists():
            print(f"[DuplicateDetector] WARNUNG: Ordner {self.dataset_dir} nicht gefunden!")
            return

        # Suche alle Bilder rekursiv
        image_files = []
        # Wir suchen jetzt explizit in den Unterordnern train/val/tech_id
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(self.dataset_dir.rglob(ext)))
            
        print(f"[DuplicateDetector] Indiziere {len(image_files)} Bilder...")
        
        for path in tqdm(image_files, desc="Erstelle Fingerabdrücke"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Features berechnen
            kp, des = self.orb.detectAndCompute(img, None)
            
            # Nur speichern, wenn das Bild genug Struktur hat (>50 Features)
            if des is not None and len(des) > 50:
                self.index.append({
                    'class': path.parent.name, # Tech-ID
                    'filename': path.name,
                    'des': des
                })
        
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.index, f)
        print(f"[DuplicateDetector] Index gespeichert ({len(self.index)} Einträge).")

    def find_match(self, query_img_bgr):
        """
        Sucht nach einem Duplikat mit sehr strengen Regeln.
        """
        if query_img_bgr is None: return None
        if not self.index: return None
        
        # 1. Features vom Query-Bild berechnen
        gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
        kp_q, des_q = self.orb.detectAndCompute(gray, None)
        
        if des_q is None or len(des_q) < 50:
            return None
            
        best_match = None
        best_score = 0
        
        # 2. Vergleich mit Datenbank
        for entry in self.index:
            db_des = entry['des']
            if db_des is None: continue
            
            # A: Cross-Check Matching (schon sehr gut)
            matches = self.bf.match(des_q, db_des)
            
            # B: STRENGE DISTANZ-FILTERUNG
            # Standard ist oft 60-70. Wir gehen auf 40 runter.
            # Das heißt: Die Punkte müssen fast identisch aussehen.
            strict_matches = [m for m in matches if m.distance < 40]
            score = len(strict_matches)
            
            if score > best_score:
                best_score = score
                best_match = entry
        
        # 3. DAS FINALE URTEIL (THRESHOLDS)
        # Wir brauchen absolute Sicherheit. 
        # Bei 2500 Features sind 300 Matches ein sehr starkes Indiz für Identität.
        # Alles darunter ist oft nur "gleicher Münz-Typ".
        MIN_MATCH_COUNT = 280 
        
        if best_score >= MIN_MATCH_COUNT:
            # Optional: Debug Print, damit du siehst wie gut der Match war
            print(f"[DuplicateDetector] Treffer! Score: {best_score} (File: {best_match['filename']})")
            
            return {
                'class': best_match['class'],
                'filename': best_match['filename'],
                'score': best_score,
                'confidence': 100.0
            }
            
        return None

if __name__ == "__main__":
    # Zum Testen Index bauen
    dd = DuplicateDetector()