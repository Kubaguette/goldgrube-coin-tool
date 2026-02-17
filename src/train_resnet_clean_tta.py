import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import time
import copy
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 

# ================= KONFIGURATION (ULTIMATE EDITION) =================
DATA_DIR = Path("data/dataset_clean") 
MODEL_SAVE_PATH = Path("models")
MODEL_SAVE_PATH.mkdir(exist_ok=True)

IMG_SIZE = 280 
BATCH_SIZE = 32
NUM_WORKERS = 12

EPOCHS = 50 

# BEREINIGTE BOOST LISTE
BOOST_CLASSES = [
    'norisch_magdalensberg',    
    'vind_regional_pollanten',  
    'norisch_karlstein',        
    'boier_roseldorf_3',        
    'boier_early_stradonice',   
    'vind_manching_duehren'     
]
# ====================================================================

# --- HELPER ---
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# --- MODELL ---
def get_resnet_model(num_classes, device):
    print("🧠 Lade Backbone: ResNet34")
    model = models.resnet34(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(device)

# --- DATASET ---
class HybridAugmentDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, aggressive_transform=None, rare_classes=None):
        super().__init__(root, transform=None) 
        self.default_transform = transform
        self.aggressive_transform = aggressive_transform
        self.rare_indices = []
        if rare_classes:
            self.rare_indices = [self.class_to_idx[c] for c in rare_classes if c in self.class_to_idx]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if target in self.rare_indices and self.aggressive_transform is not None:
            sample = self.aggressive_transform(sample)
        else:
            if self.default_transform is not None: sample = self.default_transform(sample)
        return sample, target

def get_data_loaders(data_dir, batch_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    normal_transforms = transforms.Compose([
        transforms.Resize((300, 300)), 
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm
    ])
    
    # Moderate Augmentation für ResNet (nicht zu hart)
    aggressive_transforms = transforms.Compose([
        transforms.Resize((300, 300)), 
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)), 
        transforms.RandomRotation(360), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4), transforms.GaussianBlur(3),
        transforms.ToTensor(), transforms.RandomErasing(p=0.1), norm
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((300, 300)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), norm
    ])
    
    train_dataset = HybridAugmentDataset(root=data_dir/'train', transform=normal_transforms, 
                                         aggressive_transform=aggressive_transforms, rare_classes=BOOST_CLASSES)
    val_dataset = datasets.ImageFolder(data_dir/'val', val_transforms)
    
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    boost_indices = [train_dataset.class_to_idx[c] for c in BOOST_CLASSES if c in train_dataset.class_to_idx]
    
    class_weights = []
    for idx, count in enumerate(class_counts):
        weight = 1. / count if count > 0 else 0
        if idx in boost_indices: weight *= 3.0 
        class_weights.append(weight)
        
    sampler = WeightedRandomSampler(weights=[class_weights[t] for t in targets], num_samples=len(targets), replacement=True)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    }
    return dataloaders, len(train_dataset.classes), {'train': len(train_dataset), 'val': len(val_dataset)}, val_dataset.classes

# --- TTA & ANALYSE ---
def evaluate_model_detailed(model, dataloader, class_names, device):
    print("\n📊 Starte detaillierte Analyse (TTA + Confusion Matrix)...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    corrects_tta = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Final Evaluation"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # TTA: Original + Flip H + Flip V
            out1 = F.softmax(model(inputs), dim=1)
            out2 = F.softmax(model(transforms.functional.hflip(inputs)), dim=1)
            out3 = F.softmax(model(transforms.functional.vflip(inputs)), dim=1)
            
            avg_prob = (out1 + out2 + out3) / 3.0
            _, preds = torch.max(avg_prob, 1)
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        corrects_tta += torch.sum(preds == labels.data)
        total += inputs.size(0)

    # 1. TTA Accuracy
    tta_acc = corrects_tta.double() / total
    print(f"\n🚀 Final TTA Accuracy: {tta_acc:.2%}")
    
    # 2. Confusion Matrix Plotten
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    except:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        
    plt.title(f'Confusion Matrix (Acc: {tta_acc:.1%})')
    plt.ylabel('Wahre Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    plt.tight_layout()
    
    plot_path = MODEL_SAVE_PATH / "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"📄 Confusion Matrix gespeichert unter: {plot_path}")
    
    # 3. Text Report
    print("\n📝 Detaillierter Bericht pro Klasse:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return tta_acc.item()

# --- TRAINING LOOP ---
def train_resnet(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    total_start_time = time.time()
    
    print(f"\n" + "="*85)
    print(f"🚀 ULTIMATE RESNET TRAINING | OneCycleLR | TTA | Device: {device}")
    print("="*85)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        results = {'train': {'loss': 0, 'acc': 0}, 'val': {'loss': 0, 'acc': 0}}

        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            loss_meter = AverageMeter(); acc_meter = AverageMeter()
            
            # Progress Bar Setup
            desc = f"Ep {epoch+1}/{num_epochs} [{phase}]"
            pbar = tqdm(dataloaders[phase], desc=f"{desc:<15}", leave=False, 
                        bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
            
            for inputs, labels in pbar:
                inputs = inputs.to(device); labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # Step für OneCycleLR (Batch-basiert)
                        if scheduler is not None:
                            scheduler.step()

                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update((torch.sum(preds == labels.data).double() / inputs.size(0)).item(), inputs.size(0))
                
                # Update LR im Pbar
                curr_lr_step = optimizer.param_groups[0]['lr']
                if pbar.n % 5 == 0:
                    pbar.set_postfix(L=f"{loss_meter.avg:.3f}", A=f"{acc_meter.avg:.1%}", LR=f"{curr_lr_step:.1e}" if phase=='train' else "")

            results[phase]['loss'] = loss_meter.avg
            results[phase]['acc'] = acc_meter.avg
        
        # --- ZEIT BERECHNUNG ---
        total_elapsed = time.time() - total_start_time
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_str = f"{eta_h}h {eta_m:02d}m"

        # --- ICONS ---
        gap = results['val']['loss'] - results['train']['loss']
        gap_icon = "🟢" if gap < 0.3 else "⚠️" if gap < 0.6 else "🔴"
        
        if results['val']['acc'] > best_acc:
            best_acc = results['val']['acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            record_icon = "🏆 REKORD!"
        else:
            record_icon = ""

        # --- AUSGABE ---
        print(f"Ep {epoch+1:02d} | ETA: {eta_str} | "
              f"Tr: L={results['train']['loss']:.3f} A={results['train']['acc']:.1%} | "
              f"Val: L={results['val']['loss']:.3f} A={results['val']['acc']:.1%} | {gap_icon} {record_icon}")

    return best_model_wts, best_acc

def main():
    device = torch.device("cpu")
    dataloaders, num_classes, sizes, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"📂 Gefundene Klassen: {num_classes}")
    
    model = get_resnet_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # OneCycleLR Setup
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    steps_per_epoch = len(dataloaders['train'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01, 
        epochs=EPOCHS, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3 
    )
    
    # Training
    best_weights, best_acc = train_resnet(model, dataloaders, criterion, optimizer, scheduler, EPOCHS, device)
    
    # Speichern
    torch.save(best_weights, MODEL_SAVE_PATH / "RESNET_TTA.pth")
    print(f"\n💾 Modell gespeichert.")
    
    # Final Analysis
    model.load_state_dict(best_weights)
    evaluate_model_detailed(model, dataloaders['val'], class_names, device)

if __name__ == "__main__":
    main()