import os
import shutil
import random
from pathlib import Path

# --- KONFIGURATION ---
SOURCE_DIR = Path("data/images_cropped")       
TARGET_DIR = Path("data/dataset_clean") # Neuer Ordnername, um sicherzugehen
SPLIT_RATIO = 0.80 # 80% Training, 20% Val

def split_data():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    
    (TARGET_DIR / 'train').mkdir(parents=True)
    (TARGET_DIR / 'val').mkdir(parents=True)

    classes = [d for d in os.listdir(SOURCE_DIR) if (SOURCE_DIR / d).is_dir()]
    print(f"Verteile {len(classes)} Klassen...")

    total_images = 0
    
    for class_name in classes:
        class_src = SOURCE_DIR / class_name
        images = [f for f in os.listdir(class_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Wichtig: Mischen, damit wir nicht nur die ersten X Bilder nehmen
        random.shuffle(images)
        
        # Aufteilung
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Ordner anlegen
        (TARGET_DIR / 'train' / class_name).mkdir(exist_ok=True)
        (TARGET_DIR / 'val' / class_name).mkdir(exist_ok=True)
        
        # Kopieren
        for img in train_imgs:
            shutil.copy2(class_src / img, TARGET_DIR / 'train' / class_name / img)
        for img in val_imgs:
            shutil.copy2(class_src / img, TARGET_DIR / 'val' / class_name / img)
            
        print(f"  {class_name}: {len(train_imgs)} Train | {len(val_imgs)} Val")
        total_images += len(images)

    print(f"\n✅ Fertig! {total_images} Bilder liegen in 'data/dataset_clean'")

if __name__ == "__main__":
    split_data()