import cv2
import numpy as np
import os
from pathlib import Path
from rembg import remove, new_session

# ==========================================
# SETUP: V79 (GENTLE GRABCUT)
# ==========================================
MODEL_NAME = "isnet-general-use" 
ALPHA_THRESHOLD = 70 
MORPH_ITERATIONS = 3
MIN_AREA = 1000

GRABCUT_ITERATIONS = 5 

# CHANGE: "Luft zum Atmen"
# Vorher 0.45. Jetzt 0.10.
# Das bedeutet: Alles, was tiefer als 10% der Maximaldicke liegt, ist SICHER.
# Wir erlauben GrabCut nur noch, am absoluten Rand (und an dünnen Brücken) zu schneiden.
CORE_SAFETY_RATIO = 0.10
# ==========================================

print(f"Lade Rembg Session ({MODEL_NAME})...")
session = new_session(MODEL_NAME)
print("System bereit.")

def refine_with_grabcut_gentle(img_bgr_crop, v65_mask):
    """
    Verfeinert die Maske mit GrabCut, ist aber extrem defensiv ("Gentle").
    Schützt fast die gesamte Münze und lässt GrabCut nur dünne Auswüchse prüfen.
    """
    # 1. Maske initialisieren
    gc_mask = np.zeros(img_bgr_crop.shape[:2], np.uint8)
    
    # Standard: V65 Maske ist "Möglicher Vordergrund"
    gc_mask[v65_mask > 0] = cv2.GC_PR_FGD
    gc_mask[v65_mask == 0] = cv2.GC_BGD
    
    # 2. MAXIMALER KERNSCHUTZ
    dist_transform = cv2.distanceTransform(v65_mask, cv2.DIST_L2, 5)
    max_val = dist_transform.max()
    
    # Wir berechnen den Threshold sehr niedrig (10% der Dicke).
    # Das Ergebnis: Der "Sichere Kern" (Sure FG) wird riesig und geht fast bis an den Rand.
    # Nur die dünne Brücke zum Logo (die flacher ist als dieser Threshold) bleibt "unsicher".
    dynamic_thresh = max_val * CORE_SAFETY_RATIO
    
    # Sicherheits-Minimum: Zumindest 2 Pixel Rand lassen wir GrabCut, 
    # damit es überhaupt eine Kante finden kann.
    # Aber wir cappen es nicht nach oben -> Münzen bekommen Luft.
    final_thresh = max(dynamic_thresh, 2.0)
    
    _, sure_fg = cv2.threshold(dist_transform, final_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Den riesigen sicheren Kern eintragen
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    
    # 3. GrabCut
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    try:
        cv2.grabCut(img_bgr_crop, gc_mask, None, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"    GrabCut Fehler: {e}")
        return v65_mask 
    
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
    return final_mask

def process_image_v79(image_path, output_folder):
    img_path_str = str(image_path)
    filename = Path(image_path).stem
    
    print(f"\nProzesse {filename}...", flush=True)
    
    original_img_bgr = cv2.imread(img_path_str)
    
    with open(img_path_str, 'rb') as i:
        input_data = i.read()
    try:
        output_data = remove(input_data, session=session, post_process_mask=True)
    except Exception: return

    nparr = np.frombuffer(output_data, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_rgba is None: return

    alpha = img_rgba[:, :, 3]
    
    # V65 BASIS
    _, mask = cv2.threshold(alpha, ALPHA_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    saved_count = 0
    h_full, w_full = mask.shape
    total_area = h_full * w_full
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: continue
        
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        extent = area / (w_box * h_box)
        if extent > 0.90 and area > (total_area * 0.8):
            print(f"    Skip Objekt {i}: Rahmen (Extent {extent:.2f})")
            continue
            
        # --- SCHRITT 2: GENTLE GRABCUT ---
        pad = 10
        y1 = max(0, y-pad); y2 = min(h_full, y+h_box+pad)
        x1 = max(0, x-pad); x2 = min(w_full, x+w_box+pad)
        
        crop_bgr = original_img_bgr[y1:y2, x1:x2]
        crop_mask_v65 = mask[y1:y2, x1:x2]
        
        if crop_bgr.shape[:2] != crop_mask_v65.shape[:2]: continue 

        refined_mask = refine_with_grabcut_gentle(crop_bgr, crop_mask_v65)
        
        # --- SCHRITT 3: INSEL BEREINIGUNG ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
        
        final_crop_mask = refined_mask
        
        if num_labels > 2: 
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_area = stats[largest_label, cv2.CC_STAT_AREA]
            
            # Da wir sehr sanft waren, sollte die Fläche fast identisch sein.
            # Wenn GrabCut immer noch > 50% löscht, stimmt was nicht.
            original_area = cv2.countNonZero(crop_mask_v65)
            
            if largest_area > (original_area * 0.5): 
                clean_mask = np.zeros_like(refined_mask)
                clean_mask[labels == largest_label] = 255
                final_crop_mask = clean_mask
            else:
                final_crop_mask = crop_mask_v65

        # --- SAVE ---
        saved_count += 1
        crop_rgba = img_rgba[y1:y2, x1:x2].copy()
        crop_rgba[:, :, 3] = cv2.bitwise_and(crop_rgba[:, :, 3], final_crop_mask)

        save_name = f"{filename}_{saved_count}.png"
        cv2.imwrite(os.path.join(output_folder, save_name), crop_rgba)
        print(f"    + Gespeichert: {save_name} (Gentle GrabCut)")

if __name__ == "__main__":
    input_folder = "testbilder"
    output_folder = "output_v79_gentle"
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    else:
        os.makedirs(output_folder, exist_ok=True)
        files = sorted(list(Path(input_folder).glob("*.*")))
        files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if files:
            print(f"Starte V79 (Gentle GrabCut) für {len(files)} Bilder...")
            for f in files:
                process_image_v79(f, output_folder)