import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import random

# ================= KONFIGURATION =================
DATA_DIR = Path("data/dataset_clean") 
# Wir nehmen das beste Modell aus dem letzten Lauf
MODEL_PATH = Path("models/RESNET_TTA.pth") 

# WICHTIG: ResNet wurde mit 300px trainiert (in den Transforms)
IMG_SIZE = 300 
BATCH_SIZE = 32
DEVICE = torch.device("cpu") 
# =================================================

# --- MODELL ARCHITEKTUR (ResNet34) ---
def get_resnet_model(num_classes, device):
    # Wir müssen die Architektur exakt so nachbauen wie im Training
    model = models.resnet34(weights=None) # Keine Pre-Weights, wir laden eigene
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(device)

def analyze_top3():
    print(f"🔍 Starte Top-3 Analyse für Modell: {MODEL_PATH.name}")
    
    if not MODEL_PATH.exists():
        print(f"❌ FEHLER: Datei nicht gefunden: {MODEL_PATH}")
        return

    # 1. Daten laden
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.CenterCrop(IMG_SIZE), # CenterCrop ist Standard für Validierung
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dir = DATA_DIR / 'val'
    val_dataset = datasets.ImageFolder(val_dir, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle True für zufällige Beispiele
    classes = val_dataset.classes
    print(f"📂 Validation Set: {len(val_dataset)} Bilder, {len(classes)} Klassen")

    # 2. Modell laden
    model = get_resnet_model(num_classes=len(classes), device=DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) 
        print("✅ ResNet34 Modell erfolgreich geladen.")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        print("   Hast du vielleicht versucht, ein ArcFace-Modell in ResNet zu laden?")
        return

    model.eval()
    
    top1_correct = 0
    top3_correct = 0
    total_samples = 0
    
    # Für Detail-Bericht
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}
    
    print("⏳ Analysiere Vorhersagen...")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            # Hole die Top 3 Vorhersagen (Werte und Indizes)
            # top3_probs: Wahrscheinlichkeiten, top3_idxs: Klassen-IDs
            top3_probs, top3_idxs = torch.topk(probabilities, k=3, dim=1)
            
            for i in range(inputs.size(0)):
                label = labels[i].item()
                predictions = top3_idxs[i].tolist()
                
                # Top 1 Check
                if predictions[0] == label:
                    top1_correct += 1
                    class_correct[classes[label]] += 1
                
                # Top 3 Check (Ist die richtige Klasse unter den ersten 3?)
                if label in predictions:
                    top3_correct += 1
                    
                class_total[classes[label]] += 1
                total_samples += 1

    # --- GESAMT STATISTIK ---
    acc1 = 100 * top1_correct / total_samples
    acc3 = 100 * top3_correct / total_samples
    
    print("\n" + "="*65)
    print(f"🏆 GESAMT ERGEBNIS")
    print("="*65)
    print(f"Top-1 Accuracy (Exakter Treffer):  {acc1:.2f}%")
    print(f"Top-3 Accuracy (In den Top 3):     {acc3:.2f}%")
    print("-" * 65)

    # --- DETAIL BERICHT ---
    print(f"\n📊 KLASSEN-PERFORMANCE (Top-1)")
    print(f"{'KLASSE':<30} | {'ANZ.':<5} | {'ACCURACY':<10} | {'STATUS'}")
    print("-" * 65)
    
    for cls_name in classes:
        total = class_total[cls_name]
        if total == 0: continue
        
        acc = (class_correct[cls_name] / total) * 100
        status = "✅" if acc >= 70 else "🟠" if acc >= 50 else "🔴"
        print(f"{cls_name:<30} | {total:<5} | {acc:.1f}%     | {status}")

    # --- BEISPIEL VORHERSAGEN ---
    print("\n" + "="*65)
    print("🔮 LIVE VORHERSAGE (5 Zufällige Beispiele)")
    print("="*65)
    
    # Wir nehmen einen neuen Batch für die Demo
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        top3_probs, top3_idxs = torch.topk(probs, k=3, dim=1)
        
        for i in range(min(5, len(labels))):
            true_label = classes[labels[i]]
            print(f"\n🖼️  Bild {i+1}: Wahre Klasse = '{true_label}'")
            print(f"   Vorhersagen:")
            
            for k in range(3):
                pred_class = classes[top3_idxs[i][k]]
                pred_prob = top3_probs[i][k].item() * 100
                
                # Markiere den Treffer
                marker = "👈 RICHTIG" if pred_class == true_label else ""
                print(f"   {k+1}. {pred_class:<25} ({pred_prob:.1f}%) {marker}")

if __name__ == "__main__":
    analyze_top3()