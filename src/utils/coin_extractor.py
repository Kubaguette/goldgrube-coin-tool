import numpy as np
import cv2
from rembg import remove, new_session
import os
import tempfile
from PIL import Image

# ==========================================
# CONFIG & SESSION
# ==========================================
# Wir laden die Session nur einmal beim Import, das spart Zeit
# Das ist der Extractor für die GUI - Wichtig!
try:
    print("⚙️ Lade RemBG Modell (IsNet)...")
    MODEL_NAME = "isnet-general-use"
    session = new_session(MODEL_NAME)
except Exception as e:
    print(f"⚠️ Fehler beim Laden von RemBG: {e}")
    session = None

def separate_touching_masks(binary_mask):
    """
    Trennt zusammenklebende Münzen mittels Watershed.
    """
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    mask_bgr = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_bgr, markers)

    individual_masks = []
    unique_markers = np.unique(markers)
    
    for marker_id in unique_markers:
        if marker_id <= 1: continue
        single_obj_mask = np.zeros_like(binary_mask)
        single_obj_mask[markers == marker_id] = 255
        if cv2.countNonZero(single_obj_mask) < 500: continue
        individual_masks.append(single_obj_mask)
        
    return individual_masks

def apply_gentle_grabcut(img_rgba, mask_v65):
    """Wendet GrabCut auf das maskierte Objekt an."""
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

def process_single_image(image_path):
    """
    Hauptfunktion für die GUI.
    Nimmt einen Pfad, gibt den Pfad zum besten ausgeschnittenen Temp-Bild zurück.
    """
    if session is None:
        raise Exception("RemBG Session konnte nicht geladen werden.")

    # Bild laden
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise Exception("Bild konnte nicht geladen werden.")

    # 1. RemBG
    is_success, buffer = cv2.imencode(".png", img_bgr)
    output_data = remove(buffer.tobytes(), session=session, post_process_mask=True)
    nparr = np.frombuffer(output_data, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 2. Watershed Separation
    alpha = img_rgba[:, :, 3]
    _, base_mask = cv2.threshold(alpha, 70, 255, cv2.THRESH_BINARY)
    separated_masks = separate_touching_masks(base_mask)

    if not separated_masks:
        if cv2.countNonZero(base_mask) > 500:
            separated_masks = [base_mask]
        else:
            raise Exception("Keine Münze im Bild gefunden.")

    # 3. Grabcut & Auswahl
    results = []
    for mask_part in separated_masks:
        res = apply_gentle_grabcut(img_rgba, mask_part)
        if res is not None:
            results.append(res)
            
    if not results:
        raise Exception("Extraktion fehlgeschlagen.")

    # Wir nehmen für die GUI das größte gefundene Objekt (falls mehrere da sind)
    best_image = max(results, key=lambda x: x.shape[0] * x.shape[1])

    # 4. Als temporäre Datei speichern (damit die GUI/KI es laden kann)
    # Wir nutzen PNG um Transparenz zu erhalten, auch wenn die KI meist JPG will
    # (Die KI transformiert es sich dann selbst)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "processed_coin_temp.png")
    
    cv2.imwrite(temp_path, best_image)
    
    return temp_path