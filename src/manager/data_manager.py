import pandas as pd
import os
import re
from pathlib import Path

class CoinDataManager:
    def __init__(self, base_dir=None):
        #Starte im Hauptordner 'CoinAI_Project' starten
        if base_dir is None:
            self.base_dir = Path(os.getcwd())
        else:
            self.base_dir = Path(base_dir)
            
        self.raw_data_dir = self.base_dir / "data" / "raw"
        self.database_dir = self.base_dir / "data" / "database"
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        self.keyword_map = {} # Wörterbuch: "Suchbegriff" -> "Tech ID"

    def load_mappings(self):
        """
        Lädt die Excel deiner Kollegin und baut das Wörterbuch für die KI.
        """
        excel_path = self.raw_data_dir / "Kategorisierung_und_Suchbegriffe.xlsx"
        print(f"Lade Kategorien von: {excel_path}")
        
        if not excel_path.exists():
            print(f"FEHLER: Datei nicht gefunden: {excel_path}")
            return

        # Lade Tabelle2 (wo die Synonyme stehen)
        try:
            df = pd.read_excel(excel_path, sheet_name="Tabelle2")
        except:
            # Fallback, falls das Blatt anders heißt
            df = pd.read_excel(excel_path)
            
        count = 0
        for _, row in df.iterrows():
            tech_id = str(row.get('Tech ID', '')).strip()
            
            # Überspringe leere Zeilen
            if not tech_id or tech_id.lower() == 'nan':
                continue

            # Sammle alle Suchbegriffe (DE und EN)
            terms = []
            if pd.notna(row.get('Suchbegriffe DE')):
                terms.extend(str(row['Suchbegriffe DE']).split(','))
            if pd.notna(row.get('Suchbegriffe E')):
                terms.extend(str(row['Suchbegriffe E']).split(','))
                
            # Ins Wörterbuch eintragen
            for term in terms:
                term = term.strip().lower()
                if term:
                    self.keyword_map[term] = tech_id
                    count += 1
                    
        print(f"Mapping geladen: {count} Suchbegriffe für {len(set(self.keyword_map.values()))} Kategorien.")

    def find_tech_id(self, text):
        """
        Der 'Detektiv': Sucht im Text nach bekannten Begriffen.
        Gibt die Tech ID zurück oder 'unknown'.
        """
        if not isinstance(text, str):
            return "unknown"
            
        text_lower = text.lower()
        
        # Wir suchen nach dem längsten Match zuerst (um "Manching 2" vor "Manching" zu finden)
        # Sortiere Keywords nach Länge absteigend
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Einfacher Check: Ist das Wort im Text?
            # Wir nutzen Word-Boundaries (\b), damit "Manching" nicht in "Manchinger" gefunden wird (optional)
            if keyword in text_lower:
                return self.keyword_map[keyword]
                
        return "unknown"

    def process_coinarchives(self):
        """Verarbeitet den CoinArchives Datensatz"""
        path = self.raw_data_dir / "coinarchives_dataset.csv"
        if not path.exists(): return pd.DataFrame()
        
        print(f"Verarbeite CoinArchives...")
        df = pd.read_csv(path)
        
        results = []
        for _, row in df.iterrows():
            # Wir kombinieren Titel und Beschreibung für die Suche
            full_text = f"{row.get('Titel', '')} {row.get('Beschreibung', '')}"
            tech_id = self.find_tech_id(full_text)
            
            results.append({
                "source": "coinarchives",
                "original_id": row.get('LotID'),
                "image_ref": row.get('Bild-Link'), # Hier ist es ein Web-Link
                "description": row.get('Titel'),
                "tech_id": tech_id,
                "has_image": True # Coinarchives hat immer Links
            })
            
        return pd.DataFrame(results)

    def process_numismatik_cafe(self):
        """Verarbeitet den Numismatik Cafe Datensatz"""
        # Versuche XLSX, sonst CSV
        path = self.raw_data_dir / "numismatik_cafe_dataset.xlsx"
        if not path.exists():
            path = self.raw_data_dir / "numismatik_cafe_dataset.csv" # Fallback
            
        if not path.exists(): return pd.DataFrame()
            
        print(f"Verarbeite Numismatik Cafe...")
        try:
            df = pd.read_excel(path)
        except:
            df = pd.read_csv(path)
            
        results = []
        for _, row in df.iterrows():
            # Hier nutzen wir die spezifische Spalte "Typ nach Möller"
            type_text = str(row.get('Typ nach Möller (basierend auf Kellner 1990)', ''))
            tech_id = self.find_tech_id(type_text)
            
            # Bildpfad Logik: Wir nehmen an, das Bild heißt wie die ID + .jpg
            obj_id = str(row.get('Objekt-ID Intern', ''))
            img_name = f"{obj_id}.jpg"
            
            results.append({
                "source": "numismatik_cafe",
                "original_id": obj_id,
                "image_ref": img_name, # Hier ist es ein lokaler Dateiname
                "description": row.get('Benennung im Forum'),
                "tech_id": tech_id,
                "has_image": False # Müssen wir später prüfen, ob Datei existiert
            })
            
        return pd.DataFrame(results)

    def process_occ(self):
        """Verarbeitet den OCC Datensatz"""
        path = self.raw_data_dir / "occ_dataset.csv"
        if not path.exists(): return pd.DataFrame()
        
        print(f"Verarbeite OCC...")
        df = pd.read_csv(path, sep=';') # OCC war Semikolon-getrennt
        
        results = []
        for _, row in df.iterrows():
            type_text = str(row.get('Type | Code | Code', ''))
            tech_id = self.find_tech_id(type_text)
            
            results.append({
                "source": "occ",
                "original_id": row.get('Id'),
                "image_ref": None, # OCC hat keine Bilder im Datensatz
                "description": type_text,
                "tech_id": tech_id,
                "has_image": False
            })
            
        return pd.DataFrame(results)

    def run_pipeline(self):
        """Führt alles zusammen"""
        print("--- Starte Data-Manager Pipeline ---")
        self.load_mappings()
        
        df1 = self.process_coinarchives()
        df2 = self.process_numismatik_cafe()
        df3 = self.process_occ()
        
        # Alles zusammenfügen
        master_df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # Statistik
        total = len(master_df)
        labeled = len(master_df[master_df['tech_id'] != 'unknown'])
        print(f"\n--- Ergebnis ---")
        print(f"Gesamt Datensätze: {total}")
        print(f"Erfolgreich gelabelt: {labeled} ({labeled/total*100:.1f}%)")
        print(f"Unbekannt: {total - labeled}")
        
        # Speichern
        out_path = self.database_dir / "master_coin_list.csv"
        master_df.to_csv(out_path, index=False)
        print(f"Master-Datenbank gespeichert unter: {out_path}")

if __name__ == "__main__":
    dm = CoinDataManager()
    dm.run_pipeline()