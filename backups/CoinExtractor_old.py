import cv2
import numpy as np
import os
from pathlib import Path

# ==========================================
# EINSTELLUNGEN (V31 BASIS)
# ==========================================
MIN_COIN_AREA = 800      
GRABCUT_ITERATIONS = 5
EDGE_SOFTNESS = 3

# V19/V31 TEXTUR SETTINGS
TEXTURE_THRESHOLD = 15
DENSITY_WINDOW_SIZE = 21 
DENSITY_THRESHOLD = 60
STRUCTURE_CLOSING_SIZE = 21 
# RIM_RECOVERY in V31 war 5. Wir reduzieren es leicht, 
# damit der Stoff-Rand von Anfang an dünner ist.
RIM_RECOVERY_AMOUNT = 3 
FINAL_POLISH_SIZE = 3

# *** NEU IN V37: FABRIC KILLER SETTINGS ***
ENABLE_FABRIC_KILLER = True
# Wie empfindlich reagiert der Filter auf das Muster?
# Niedriger = Aggressiver (löscht mehr). Höher = Vorsichtiger.
LAPLACIAN_THRESHOLD = 40 
# Wie breit ist die Randzone, in der wir nach Stoff suchen?
RIM_ZONE_THICKNESS = 7
# ==========================================

def extract_coins_v37(image_path, output_folder, debug=True):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Fehler: Bild nicht gefunden: {image_path}")
        return

    filename = Path(image_path).stem
    if debug:
        debug_folder = os.path.join(output_folder, "debug_steps")
        os.makedirs(debug_folder, exist_ok=True)

    # --- SCHRITT 1: SPLITTER (Positionen finden - Wie V31) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    
    mean_val = np.mean(blurred)
    if mean_val < 100: 
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: 
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((9,9), np.uint8)
    mask_split = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    bounding_boxes = []
    height, width = img.shape[:2]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_COIN_AREA or area > (width * height * 0.95):
            continue
        valid_contours.append(cnt)
        bounding_boxes.append(cv2.boundingRect(cnt))

    if len(valid_contours) > 0:
        sorted_data = sorted(zip(valid_contours, bounding_boxes), key=lambda b: b[1][0])
        valid_contours, bounding_boxes = zip(*sorted_data)
        print(f"Analyse '{filename}': {len(valid_contours)} Objekte gefunden.")
    else:
        print("Keine Objekte gefunden.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # --- SCHRITT 2: CORE & FABRIC KILLER ---
    for i, cnt in enumerate(valid_contours):
        current_suffix = f"part_{i}" 
        print(f"  + Verarbeite Objekt {i+1} ({current_suffix})...")

        x, y, w, h = bounding_boxes[i]
        pad = 40 
        x_gc = max(0, x - pad); y_gc = max(0, y - pad)
        w_gc = min(width - x_gc, w + 2*pad); h_gc = min(height - y_gc, h + 2*pad)
        roi = img[y_gc:y_gc+h_gc, x_gc:x_gc+w_gc]
        
        # A) V31 BASIS (GrabCut + Density Core)
        # ------------------------------------
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect_h, rect_w = roi.shape[:2]
        rect_gc = (10, 10, rect_w-20, rect_h-20)
        try:
            cv2.grabCut(roi, mask, rect_gc, bgdModel, fgdModel, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
        except: continue
        mask_grabcut = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        roi_smooth = cv2.bilateralFilter(roi, 9, 75, 75)
        roi_gray = cv2.cvtColor(roi_smooth, cv2.COLOR_BGR2GRAY)
        roi_mask_gc = mask_grabcut
        sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobelx, sobely)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, rough_mask = cv2.threshold(gradient, TEXTURE_THRESHOLD, 255, cv2.THRESH_BINARY)
        rough_mask = cv2.bitwise_and(rough_mask, rough_mask, mask=roi_mask_gc)
        density_map = cv2.boxFilter(rough_mask, -1, (DENSITY_WINDOW_SIZE, DENSITY_WINDOW_SIZE), normalize=True)
        _, dense_blob = cv2.threshold(density_map, DENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel_close = np.ones((STRUCTURE_CLOSING_SIZE, STRUCTURE_CLOSING_SIZE), np.uint8)
        solid_blob_v17 = cv2.morphologyEx(dense_blob, cv2.MORPH_CLOSE, kernel_close)
        mask_core = np.zeros_like(mask_grabcut)
        cnts_tex, _ = cv2.findContours(solid_blob_v17, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_tex:
            c_max = max(cnts_tex, key=cv2.contourArea)
            cv2.drawContours(mask_core, [c_max], -1, 1, thickness=-1)
        else:
            if np.sum(mask_grabcut) > 0: mask_core = mask_grabcut
            else: continue

        # Inflation & Polish (V31 Ergebnis)
        kernel_inflate = np.ones((3,3), np.uint8)
        mask_inflated = cv2.dilate(mask_core, kernel_inflate, iterations=RIM_RECOVERY_AMOUNT)
        mask_clamped = cv2.bitwise_and(mask_inflated, mask_grabcut)
        kernel_polish = np.ones((FINAL_POLISH_SIZE, FINAL_POLISH_SIZE), np.uint8)
        mask_v31_result = cv2.morphologyEx(mask_clamped, cv2.MORPH_OPEN, kernel_polish)
        
        mask_final = mask_v31_result * 255

        # B) THE FABRIC KILLER (Laplacian Trimmer)
        # ----------------------------------------
        if ENABLE_FABRIC_KILLER:
            # 1. Textur-Aktivitäts-Karte erstellen (Laplace)
            # Wir nutzen das Original-Graubild (ungeglättet) für maximale Details
            roi_gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Laplace findet Kanten zweiter Ordnung (sehr feine Details)
            laplacian = cv2.Laplacian(roi_gray_raw, cv2.CV_64F)
            # Absolutwerte nehmen und in 0-255 umwandeln
            texture_activity = cv2.convertScaleAbs(laplacian)

            # 2. Stoff-Maske erstellen
            # Alles was extrem hohe Aktivität hat, ist Stoffmuster
            _, fabric_mask = cv2.threshold(texture_activity, LAPLACIAN_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Rauschen entfernen
            fabric_mask = cv2.morphologyEx(fabric_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

            # 3. Randzone definieren (Wo dürfen wir schneiden?)
            # Wir suchen die Kante der aktuellen V31-Maske
            mask_outline = cv2.Canny(mask_v31_result * 255, 100, 200)
            # Wir verbreitern die Kante zu einem Ring
            rim_zone = cv2.dilate(mask_outline, np.ones((RIM_ZONE_THICKNESS, RIM_ZONE_THICKNESS),np.uint8), iterations=1)

            # 4. Die Operation: Finde Stoff IN der Randzone
            fabric_at_rim = cv2.bitwise_and(fabric_mask, rim_zone)

            # 5. Subtrahiere diesen Stoff von der Hauptmaske
            mask_trimmed = cv2.subtract(mask_v31_result * 255, fabric_at_rim)

            # 6. Aufräumen (Sicherstellen, dass die Maske solide bleibt)
            # Falls wir ein Loch in den Rand geschnitten haben, füllen wir das Objekt neu.
            cnts_trim, _ = cv2.findContours(mask_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_trim:
                c_trim_max = max(cnts_trim, key=cv2.contourArea)
                mask_final = np.zeros_like(mask_trimmed)
                # thickness=-1 füllt alles wieder auf
                cv2.drawContours(mask_final, [c_trim_max], -1, 255, thickness=-1)
            else:
                mask_final = mask_trimmed # Fallback

            if debug and i == 1: # Debug nur für das kritische 2. Objekt (Test 3.2)
                cv2.imwrite(os.path.join(debug_folder, f"{filename}_{current_suffix}_01_texture_activity.jpg"), texture_activity)
                cv2.imwrite(os.path.join(debug_folder, f"{filename}_{current_suffix}_02_fabric_mask.jpg"), fabric_mask)
                cv2.imwrite(os.path.join(debug_folder, f"{filename}_{current_suffix}_03_rim_zone.jpg"), rim_zone)
                cv2.imwrite(os.path.join(debug_folder, f"{filename}_{current_suffix}_04_fabric_at_rim.jpg"), fabric_at_rim)

        # --- SPEICHERN ---
        final_mask_full = np.zeros((height, width), np.uint8)
        final_mask_full[y_gc:y_gc+h_gc, x_gc:x_gc+w_gc] = mask_final
        
        if EDGE_SOFTNESS > 0:
            final_mask_full = cv2.GaussianBlur(final_mask_full, (EDGE_SOFTNESS, EDGE_SOFTNESS), 0)

        b, g, r = cv2.split(img)
        rgba = [b, g, r, final_mask_full]
        dst = cv2.merge(rgba, 4)

        x_f, y_f, w_f, h_f = cv2.boundingRect(final_mask_full)
        if w_f > 0 and h_f > 0:
            crop = dst[y_f:y_f+h_f, x_f:x_f+w_f]
            save_name = f"{filename}_{current_suffix}.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, crop)
            print(f" -> Gespeichert: {save_name} (V37 Fabric-Killer)")
        else:
            print(f"    Fehler: Leeres Bild bei Objekt {i}")

if __name__ == "__main__":
    for test_img in ["test1.jpg", "test2.jpg", "test3.jpg"]:
        if os.path.exists(test_img):
            print(f"--- Starte V37 für {test_img} ---")
            extract_coins_v37(test_img, f"output_v37_{test_img.split('.')[0]}")