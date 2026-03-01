# Projekt: Goldgrube

## Kapitel 1: Einrichten vom Projekt

* **IDE:** Dieses Projekt wurde für die Nutzung mit **Visual Studio Code** unter **Bazzite (Linux)** eingerichtet.
* **Python-Version:** Das Programm benötigt ausdrücklich **Python 3.13.14**. Nur unter dieser Version ist gewährleistet, dass die Anwendung und alle Pakete zuverlässig laufen.
* **Abhängigkeiten:** Alle benötigten Pakete sind in der `requirements.txt` gelistet und müssen über das Terminal installiert werden.

Standard-Installation:

```pip install -r requirements.txt```

## Kapitel 2: Das Training
Um das KI-Modell (ResNet) mit neuen Daten zu trainieren, müssen folgende Schritte in dieser Reihenfolge ausgeführt werden:

* **Datenbank aktualisieren**: Die Datei ```master_coin_list.csv``` fungiert als Datenbank. Für jede neue Münze muss hier ein neuer Eintrag erstellt werden.

* **Wichtig**: Der Wert unter ```original_id``` muss exakt mit dem Dateinamen des jeweiligen Bildes übereinstimmen!

* **Bilder bereitstellen**: Alle Roh-Bilddateien müssen in folgendem Ordner vorliegen:
```data/images_raw```

* **Bilder zuschneiden und sortieren**: Dieses Skript verarbeitet die Rohbilder und erstellt einen neuen Ordner mit den fertig ausgeschnittenen und korrekt geordneten Bildern:
Bash: ```python image_factory_local.py```

* **Datensatz aufteilen**: Dieses Skript teilt die verarbeiteten Bilder in zwei Sets auf (80% Trainingsdaten und 20% Validierungsdaten):
Bash: ```python split_dataset.py```

* **Training starten**: Führe dieses Skript aus, um den Trainingsprozess zu beginnen. Das fertige Modell wird nach Abschluss im Ordner models generiert:
Bash: ```python train_resnet_clean_tta.py```

## Kapitel 3: Programm starten
Wenn das Projekt eingerichtet und ein Modell vorhanden ist, kann die Hauptanwendung gestartet werden:

Bash: ```python CoinApp.py```
