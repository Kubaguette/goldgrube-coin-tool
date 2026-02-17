import cv2
import numpy as np
import os
from pathlib import Path
from rembg import remove, new_session

# ==========================================
# SETUP: V75 (V65 BASE + GRABCUT REFINER)
# ==========================================
MODEL_NAME = "isnet-general-use" 
ALPHA_THRESHOLD = 70 
MORPH_ITERATIONS = 3
MIN_AREA = 1000

# GRABCUT EINSTELLUNGEN
# Wie oft soll der Algorithmus über das Bild laufen?
# 5 ist ein guter Standardwert für Präzision vs. Geschwindigkeit.
GRABCUT_ITERATIONS = 5 
# ==========================================

print(f"Lade Rembg Session ({MODEL_NAME})...")
session = new_session(MODEL_NAME)
print("System bereit.")

def refine_with_grabcut(img_bgr_crop, v65_mask):
    """
    Verfeinert die Maske mit GrabCut.
    Schützt den Kern der Münze, bewertet die Ränder neu.
    """
    # 1. Maske für GrabCut vorbereiten
    # Werte: 0 (BGD), 1 (FGD), 2 (PR_BGD), 3 (PR_FGD)
    gc_mask = np.zeros(img_bgr_crop.shape[:2], np.uint8)
    
    # Alles, was V65 gefunden hat, ist erst mal "Wahrscheinlicher Vordergrund"
    gc_mask[v65_mask > 0] = cv2.GC_PR_FGD
    
    # Alles, was V65 als Hintergrund sah, ist "Sicherer Hintergrund"
    gc_mask[v65_mask == 0] = cv2.GC_BGD
    
    # WICHTIG: Man muss den "Sicheren Kern" definieren (Gegen Löcher!)
    # Erodiere die V65 Maske aggressiv. Was übrig bleibt, ist definitiv Münze.
    # Das schützt die Mitte vor dem Löschen.
    kernel = np.ones((5,5), np.uint8)
    # 5-8 Iterationen, damit wir weit weg vom Rand/Logo sind
    sure_foreground = cv2.erode(v65_mask, kernel, iterations=6) 
    gc_mask[sure_foreground > 0] = cv2.GC_FGD
    
    # 2. GrabCut Modelle initialisieren (Interner Speicher)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # 3. GrabCut ausführen
    try:
        cv2.grabCut(img_bgr_crop, gc_mask, None, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"    GrabCut Fehler: {e}")
        return v65_mask # Fallback auf Originalmaske
    
    # 4. Ergebnis auswerten
    # Wir nehmen alles, was Sicherer (1) oder Wahrscheinlicher (3) Vordergrund ist
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
    
    return final_mask

def process_image_v75(image_path, output_folder):
    img_path_str = str(image_path)
    filename = Path(image_path).stem
    
    print(f"\nProzesse {filename}...", flush=True)
    
    # Originalbild laden (für GrabCut brauchen wir das BGR Bild ohne Alpha)
    original_img_bgr = cv2.imread(img_path_str)
    
    with open(img_path_str, 'rb') as i:
        input_data = i.read()
        
    try:
        output_data = remove(input_data, session=session, post_process_mask=True)
    except Exception as e:
        print(f"    Fehler bei Rembg: {e}")
        return

    nparr = np.frombuffer(output_data, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_rgba is None: return

    alpha = img_rgba[:, :, 3]
    
    # 1. V65 BASIS
    _, mask = cv2.threshold(alpha, ALPHA_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    max_area = max([cv2.contourArea(c) for c in contours])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    saved_count = 0
    h_full, w_full = mask.shape
    total_area = h_full * w_full
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: continue
        
        # Bild-im-Bild Schutz
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        extent = area / (w_box * h_box)
        if extent > 0.90 and area > (total_area * 0.8):
            print(f"    Skip Objekt {i}: Rahmen (Extent {extent:.2f})")
            continue
            
        # --- SCHRITT 2: GRABCUT REFINEMENT ---
        # Wir schneiden den Bereich aus, um GrabCut lokal laufen zu lassen (schneller & präziser)
        pad = 10
        y1 = max(0, y-pad); y2 = min(h_full, y+h_box+pad)
        x1 = max(0, x-pad); x2 = min(w_full, x+w_box+pad)
        
        crop_bgr = original_img_bgr[y1:y2, x1:x2]
        crop_mask_v65 = mask[y1:y2, x1:x2]
        
        # Größen-Check (falls Padding an Rändern clippt)
        if crop_bgr.shape[:2] != crop_mask_v65.shape[:2]:
            continue 

        # GrabCut anwenden
        refined_mask = refine_with_grabcut(crop_bgr, crop_mask_v65)
        
        # --- SCHRITT 3: ISLAND FILTER (Nachbereitung) ---
        # Falls GrabCut die Brücke durchtrennt hat, müssen wir das abgetrennte Logo löschen.
        # Wir nutzen connectedComponents auf der NEUEN Maske.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
        
        final_crop_mask = refined_mask
        
        if num_labels > 2: # Mehr als 1 Objekt (Hintergrund + Münze + evtl Logo)
            # Finde das größte Objekt (die Münze)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_area = stats[largest_label, cv2.CC_STAT_AREA]
            
            # Neue Maske erstellen nur mit der Münze
            clean_mask = np.zeros_like(refined_mask)
            clean_mask[labels == largest_label] = 255
            
            # Prüfen ob wir versehentlich die Münze gelöscht hätten (Sicherheitsnetz)
            if largest_area > (area * 0.5): 
                final_crop_mask = clean_mask
            else:
                # GrabCut hat Mist gebaut, wir bleiben bei der V65 Maske
                final_crop_mask = crop_mask_v65

        # --- SAVE ---
        saved_count += 1
        
        # Crop vom RGBA erstellen
        crop_rgba = img_rgba[y1:y2, x1:x2].copy()
        
        # Die verfeinerte Maske anwenden
        crop_rgba[:, :, 3] = cv2.bitwise_and(crop_rgba[:, :, 3], final_crop_mask)

        save_name = f"{filename}_{saved_count}.png"
        cv2.imwrite(os.path.join(output_folder, save_name), crop_rgba)
        print(f"    + Gespeichert: {save_name} (GrabCut Refined)")

if __name__ == "__main__":
    input_folder = "testbilder"
    output_folder = "output_v75_grabcut"
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    else:
        os.makedirs(output_folder, exist_ok=True)
        files = sorted(list(Path(input_folder).glob("*.*")))
        files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if files:
            print(f"Starte V75 (V65 + GrabCut) für {len(files)} Bilder...")
            for f in files:
                process_image_v75(f, output_folder)