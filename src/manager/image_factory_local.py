import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from rembg import remove, new_session
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
MODEL_NAME = "isnet-general-use"
session = new_session(MODEL_NAME)

def separate_touching_masks(binary_mask):
    """
    Nimmt eine binäre Maske (in der Objekte zusammenkleben können)
    und versucht, sie mittels Distance Transform & Watershed zu trennen.
    Gibt eine Liste von einzelnen Masken zurück (eine pro Objekt).
    """
    # 1. Noise Removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. Sure Background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 3. Finding Sure Foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Threshold: Alles was mind. 40% der maximalen Helligkeit hat, ist sicher eine Münze.
    if dist_transform.max() == 0:
        return [binary_mask] # Fallback für leere Masken
        
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    # 4. Finding Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Add 1 so bg is 1, not 0
    markers[unknown == 255] = 0 # Mark unknown region with 0

    # 6. Watershed ausführen
    mask_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_bgr, markers)

    # 7. Einzelne Masken extrahieren
    individual_masks = []
    unique_markers = np.unique(markers)
    
    for marker_id in unique_markers:
        if marker_id <= 1: # 0 ist Boundary, 1 ist Background
            continue
            
        single_obj_mask = np.zeros_like(binary_mask)
        single_obj_mask[markers == marker_id] = 255
        
        if cv2.countNonZero(single_obj_mask) < 500:
            continue
            
        individual_masks.append(single_obj_mask)
        
    return individual_masks

def apply_gentle_grabcut(img_rgba, mask_v65):
    """
    Wendet V79 Logik auf ein bereits maskiertes Objekt an.
    """
    contours, _ = cv2.findContours(mask_v65, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    pad = 10
    y1 = max(0, y-pad); y2 = min(img_rgba.shape[0], y+h+pad)
    x1 = max(0, x-pad); x2 = min(img_rgba.shape[1], x+w+pad)
    
    crop_bgr = img_rgba[y1:y2, x1:x2, :3]
    crop_mask = mask_v65[y1:y2, x1:x2]
    
    gc_mask = np.zeros(crop_bgr.shape[:2], np.uint8)
    gc_mask[crop_mask > 0] = cv2.GC_PR_FGD
    gc_mask[crop_mask == 0] = cv2.GC_BGD
    
    dist_transform = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
    if dist_transform is None: return None
    
    max_val = dist_transform.max()
    dynamic_thresh = max(max_val * 0.10, 2.0)
    _, sure_fg = cv2.threshold(dist_transform, dynamic_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    try:
        cv2.grabCut(crop_bgr, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except: pass 
    
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
    
    crop_final = img_rgba[y1:y2, x1:x2].copy()
    crop_final[:, :, 3] = final_mask
    return crop_final

def run_v79_extraction_multi(img_bgr):
    if img_bgr is None: return []
    
    is_success, buffer = cv2.imencode(".png", img_bgr)
    if not is_success: return []
    try:
        output_data = remove(buffer.tobytes(), session=session, post_process_mask=True)
    except Exception: return []

    nparr = np.frombuffer(output_data, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_rgba is None: return []

    alpha = img_rgba[:, :, 3]
    _, base_mask = cv2.threshold(alpha, 70, 255, cv2.THRESH_BINARY)
    
    separated_masks = separate_touching_masks(base_mask)
    
    if not separated_masks:
        if cv2.countNonZero(base_mask) > 500:
            separated_masks = [base_mask]
        else:
            return []

    final_images = []
    for mask_part in separated_masks:
        result_img = apply_gentle_grabcut(img_rgba, mask_part)
        if result_img is not None:
            final_images.append(result_img)
            
    return final_images

def find_local_image(original_id, source, file_index):
    original_id_str = str(original_id).strip()
    # Priorisiere exakte Matches
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        candidate = f"{original_id_str}{ext}"
        if candidate in file_index: return file_index[candidate]

    # Fallback für Coinarchives (Prefix-Suche)
    if source == 'coinarchives':
        prefix = f"{original_id_str}_"
        for fname in file_index:
            if fname.startswith(prefix): return file_index[fname]
    return None

# ==========================================
# MAIN (UPDATED FOR XLSX & CSV)
# ==========================================
def main():
    base_dir = Path(os.getcwd())
    database_dir = base_dir / "data" / "database"
    output_dir = base_dir / "data" / "images_cropped"
    raw_img_dir = base_dir / "data" / "images_raw"
    
    # 1. VERSUCHE DATENBANK ZU LADEN (XLSX > CSV)
    df = None
    loaded_filename = ""
    
    # Liste der möglichen Dateien (Priorität: XLSX)
    possible_files = [
        database_dir / "master_coin_list.xlsx",
        database_dir / "master_coin_list.xls",
        database_dir / "master_coin_list.csv"
    ]
    
    print("Suche Datenbank...")
    for file_path in possible_files:
        if file_path.exists():
            print(f"Datei gefunden: {file_path.name}")
            try:
                if file_path.suffix.lower() in ['.xlsx', '.xls']:
                    # Excel laden
                    df = pd.read_excel(file_path)
                    print("✅ Excel-Datei erfolgreich geladen.")
                else:
                    # CSV laden (mit Fallback-Strategie)
                    try:
                        df = pd.read_csv(file_path)
                        print("✅ CSV (Standard) geladen.")
                    except:
                        print("⚠️ Standard-CSV gescheitert. Versuche robusten Modus (Separator-Suche)...")
                        # on_bad_lines='skip': Ignoriert kaputte Zeilen (wie deine Zeile 3)
                        # sep=None: Sucht automatisch nach Komma, Semikolon oder Tab
                        df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
                        print("✅ CSV (Robust) geladen.")
                
                loaded_filename = file_path.name
                break # Wenn erfolgreich geladen, brich die Suche ab
            except Exception as e:
                print(f"❌ Fehler beim Laden von {file_path.name}: {e}")
                continue # Probier die nächste Datei
    
    if df is None:
        print("❌ KRITISCHER FEHLER: Keine lesbare Datenbank (xlsx/csv) gefunden!")
        print(f"Bitte stelle sicher, dass eine 'master_coin_list' im Ordner '{database_dir}' liegt.")
        return

    # 2. DATEN BEREINIGEN
    # Spaltennamen normalisieren (falls Leerzeichen drin sind)
    df.columns = df.columns.str.strip()
    
    # Check ob wichtige Spalten da sind
    required_cols = ['tech_id', 'original_id', 'source']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ FEHLER: Die Tabelle muss die Spalten {required_cols} enthalten!")
        print(f"Gefundene Spalten: {list(df.columns)}")
        return

    # Entferne Zeilen ohne tech_id
    df = df.dropna(subset=['tech_id'])
    df['tech_id'] = df['tech_id'].astype(str)
    
    # Filter 'unknown' raus
    df_clean = df[df['tech_id'] != 'unknown'].copy()
    
    # 3. BILDER SUCHEN
    print(f"Indiziere Bilder in {raw_img_dir}...")
    file_index = {}
    if raw_img_dir.exists():
        for f in raw_img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                file_index[f.name] = f
    else:
        print(f"❌ Warnung: Bilder-Ordner {raw_img_dir} existiert nicht!")
            
    print(f"Starte Verarbeitung von {len(df_clean)} Münzen aus '{loaded_filename}'...")
    
    success_count = 0
    missing_count = 0
    
    # 4. VERARBEITUNG
    for index, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0]):
        tech_id = row['tech_id'].strip() # Leerzeichen entfernen
        orig_id = row['original_id']
        source = row['source']
        
        target_folder = output_dir / tech_id
        target_folder.mkdir(parents=True, exist_ok=True)
        
        img_path = find_local_image(orig_id, source, file_index)
        
        if img_path:
            img_bgr = cv2.imread(str(img_path))
            
            # Multi-Crop Logik
            results = run_v79_extraction_multi(img_bgr)
            
            if results:
                for i, res_img in enumerate(results):
                    suffix = f"_{i}" if len(results) > 1 else ""
                    save_name = f"{source}_{orig_id}{suffix}.png"
                    save_path = target_folder / save_name
                    
                    cv2.imwrite(str(save_path), res_img)
                
                success_count += 1
        else:
            missing_count += 1

    print(f"\n--- FERTIG ---")
    print(f"Verwendete Datei: {loaded_filename}")
    print(f"Einträge verarbeitet: {success_count}")
    print(f"Bilder nicht gefunden: {missing_count}")

if __name__ == "__main__":
    main()