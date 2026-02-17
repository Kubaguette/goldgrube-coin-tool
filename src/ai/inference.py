import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
from pathlib import Path

# --- KONFIGURATION ---
# Wir suchen das Modell relativ zu diesem Skript
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "RESNET_TTA.pth"

# WICHTIG: Die Klassen exakt wie auf deinem Screenshot (alphabetisch sortiert!)
CLASSES = [
    'boier_early_stradonice',
    'boier_late_Karlstein',
    'boier_roseldorf_1',
    'boier_roseldorf_2',
    'boier_roseldorf_3',
    'norisch_karlstein',
    'norisch_magdalensberg',
    'vind_manching',
    'vind_manching_duehren',
    'vind_regional_pollanten'
]

class CoinPredictor:
    def __init__(self):
        self.device = torch.device("cpu") # Für einzelne Bilder reicht CPU völlig
        print(f"🧠 Lade KI-Modell von: {MODEL_PATH}")
        
        self.is_ready = False
        try:
            self.model = self._load_model()
            self.transform = self._get_transforms()
            self.is_ready = True
            print("✅ ResNet34 Modell erfolgreich geladen!")
        except Exception as e:
            print(f"❌ FEHLER beim Laden des Modells: {e}")
            self.is_ready = False

    def _load_model(self):
        # 1. Architektur aufbauen (Muss exakt dem Training entsprechen: ResNet34)
        model = models.resnet34(weights=None)
        
        # 2. Den "Kopf" des Modells anpassen (10 Klassen)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(CLASSES))
        )
        
        # 3. Gewichte laden
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modell-Datei nicht gefunden: {MODEL_PATH}")
            
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        # Exakt die gleichen Transforms wie beim Training/Validation
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.CenterCrop(280),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Führt eine Vorhersage mit TTA durch und gibt ALLE Wahrscheinlichkeiten zurück.
        Rückgabe-Format: Dictionary {'klassen_name': wahrscheinlichkeit, ...}
        """
        if not self.is_ready:
            print("⚠️ Modell ist nicht bereit.")
            return {}

        try:
            # Bild laden
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 1. Vorhersage: Original
                out1 = F.softmax(self.model(img_t), dim=1)
                
                # 2. Vorhersage: Horizontal gespiegelt (TTA)
                img_h = transforms.functional.hflip(img_t)
                out2 = F.softmax(self.model(img_h), dim=1)
                
                # 3. Vorhersage: Vertikal gespiegelt (TTA)
                img_v = transforms.functional.vflip(img_t)
                out3 = F.softmax(self.model(img_v), dim=1)
                
                # DURCHSCHNITT BILDEN (Ensemble)
                final_prob = (out1 + out2 + out3) / 3.0
                
                # WICHTIG: Wir geben jetzt ALLES zurück, nicht nur Top 3
                # Damit kann die GUI zwei Bilder mathematisch korrekt verrechnen.
                result_dict = {}
                for i, class_name in enumerate(CLASSES):
                    # .item() wandelt den Tensor-Wert in eine normale Python-Float um
                    result_dict[class_name] = final_prob[0][i].item()
                    
                return result_dict

        except Exception as e:
            print(f"❌ Fehler bei der Vorhersage: {e}")
            return {}