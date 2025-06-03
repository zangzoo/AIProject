
# train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import wandb
from tqdm.auto import tqdm 

from dataset import SignDataset
from models import SignTransformer
from augment import augment_sequence

def run_fold(fold, train_idx, val_idx, all_files, labels, label_map, config, device):
    # DataLoader 준비
    train_ds = SignDataset([all_files[i] for i in train_idx], label_map, augment=augment_sequence)
    val_ds   = SignDataset([all_files[i] for i in val_idx],   label_map, augment=None)
    tr_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # 모델/옵티마이저/스케줄러 초기화
    model = SignTransformer(num_classes=len(label_map), input_dim=config.input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val = 0.0
    for epoch in range(1, config.epochs+1):
        # — Train
        model.train()
        tloss = tcorrect = ttotal = 0
        for x, y in tqdm(tr_loader,
                          desc=f"Fold {fold} ▶ Train E{epoch}/{config.epochs}",
                          leave=False):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = out.argmax(1)
            ttotal += y.size(0)
            tcorrect += (preds==y).sum().item()
            tloss += loss.item()
        train_acc  = tcorrect/ttotal
        train_loss = tloss/len(tr_loader)

        # — Validation
        model.eval()
        vloss = vcorrect = vtotal = 0
        ys = []; ps = []
        with torch.no_grad():
            for x, y in tqdm(va_loader,
                          desc=f"Fold {fold} ▶ Val   E{epoch}/{config.epochs}",
                          leave=False):
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                vloss += loss.item()
                preds = out.argmax(1)
                vcorrect += (preds==y).sum().item()
                vtotal   += y.size(0)
                ys += y.cpu().tolist()
                ps += preds.cpu().tolist()
        val_acc  = vcorrect/vtotal
        val_loss = vloss/len(va_loader)

        # — Best 모델 저장
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f'best_model_fold{fold}.pt')
            print(f"[Fold {fold}] Saved best model at epoch {epoch} with val_acc={val_acc:.4f}")

        # — W&B & 터미널 로그
        cm = confusion_matrix(ys, ps, labels=list(range(len(label_map))))
        per_cls = cm.diagonal()/cm.sum(axis=1)
        log_dict = {
            'fold': fold, 'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        }
        log_dict.update({f'val_acc_cls_{i}': a for i,a in enumerate(per_cls)})
        wandb.log(log_dict)

        print(f"[Fold {fold}] Epoch {epoch}/{config.epochs} | "
              f"Tr Loss {train_loss:.4f}, Tr Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        scheduler.step()

    return best_val

def main():
    # 1) 데이터 준비
    data_dir = Path('data')
    all_files = list(data_dir.rglob('*.mp4'))
    labels    = [p.parent.name for p in all_files]
    label_map = {l:i for i,l in enumerate(sorted(set(labels)))}

    # 2) W&B 초기화 (global config)
    wandb.init(project='Sign_Translation', entity="jangjang0022-sungshin-women-s-university", reinit=True,
               config={
                   'epochs': 50, 'batch_size': 8, 'lr': 1e-4, 'weight_decay':1e-5,
                   'input_dim': 1662
               })
    config = wandb.config
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    # 3) Stratified K-Fold (k=5)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, labels), 1):
        print(f"\n===== Starting fold {fold} =====")
        score = run_fold(fold, train_idx, val_idx,
                         all_files, labels, label_map,
                         config, device)
        fold_scores.append(score)

    # 4) 교차검증 결과 요약
    mean = sum(fold_scores)/len(fold_scores)
    std  = (sum((s-mean)**2 for s in fold_scores)/len(fold_scores))**0.5
    print(f"\nCross-Validation Accuracy: {mean:.4f} ± {std:.4f}")

if __name__ == '__main__':
    main()
