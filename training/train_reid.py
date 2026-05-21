#!/usr/bin/env python3
"""
Fase 3 — Treino do modelo de ReID para andebol.

Arquitectura: ResNet50 pretrained + embedding head (512-dim)
Loss:         Triplet loss (hard mining) + Cross-entropy
Dataset:      training/reid_dataset/train/ e val/
GPU:          RTX 4070 (~20-30 min)

Output:
  training/reid_model/best.pt     ← modelo PyTorch
  training/reid_model/best.onnx   ← export para integração com BoT-SORT

Uso:
  python training/train_reid.py
  python training/train_reid.py --epochs 80 --batch-p 10
"""
import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
import cv2

# ── Configuração ──────────────────────────────────────────────────────────────
DATA_DIR   = 'training/reid_dataset'
OUTPUT_DIR = 'training/reid_model'
EMBED_DIM  = 512
IMG_H, IMG_W = 256, 128
EPOCHS     = 60
P          = 8    # identidades por batch
K          = 4    # imagens por identidade  (batch = P*K = 32)
LR         = 3e-4
MARGIN     = 0.3  # margem do triplet loss
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

class ReIDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []   # (img_path, label_idx)
        self.id2label  = {}

        for i, id_dir in enumerate(sorted(Path(root).iterdir())):
            if not id_dir.is_dir():
                continue
            self.id2label[id_dir.name] = i
            for img_path in sorted(id_dir.glob('*.jpg')):
                self.samples.append((str(img_path), i))

        self.n_ids = len(self.id2label)
        print(f"  Dataset: {self.n_ids} identidades, {len(self.samples)} imagens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, label


class PKSampler(Sampler):
    """Amostra P identidades × K imagens por batch."""
    def __init__(self, dataset, P, K):
        self.P = P
        self.K = K
        # Agrupa índices por identidade
        self.id_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.id_to_indices[label].append(idx)
        self.ids = [i for i, idxs in self.id_to_indices.items() if len(idxs) >= 2]

    def __iter__(self):
        batch = []
        ids_shuffled = self.ids.copy()
        random.shuffle(ids_shuffled)
        for pid in ids_shuffled:
            idxs = self.id_to_indices[pid]
            chosen = random.choices(idxs, k=self.K)
            batch.extend(chosen)
            if len(batch) >= self.P * self.K:
                yield batch[:self.P * self.K]
                batch = []

    def __len__(self):
        return len(self.ids) // self.P


# ── Modelo ────────────────────────────────────────────────────────────────────

class ReIDModel(nn.Module):
    def __init__(self, n_classes, embed_dim=512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone  = backbone
        self.embed     = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_logits=False):
        feat   = self.backbone(x)
        embed  = self.embed(feat)
        embed_norm = F.normalize(embed, dim=1)
        if return_logits:
            logits = self.classifier(embed)
            return embed_norm, logits
        return embed_norm


# ── Triplet loss com hard mining ──────────────────────────────────────────────

def triplet_hard(embeddings, labels, margin):
    """
    Batch hard triplet loss.
    Para cada anchor, escolhe o positive mais distante e o negative mais próximo.
    """
    dist = torch.cdist(embeddings, embeddings, p=2)
    n = len(labels)
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    for i in range(n):
        pos_mask = (labels == labels[i]) & (torch.arange(n, device=embeddings.device) != i)
        neg_mask = labels != labels[i]
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        hardest_pos = dist[i][pos_mask].max()
        hardest_neg = dist[i][neg_mask].min()
        loss += F.relu(hardest_pos - hardest_neg + margin)
        count += 1
    return loss / max(count, 1)


# ── Treino ────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, margin, device):
    model.train()
    total_loss = total_tri = total_ce = n = 0
    ce_fn = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        embeds, logits = model(imgs, return_logits=True)

        loss_tri = triplet_hard(embeds, labels, margin)
        loss_ce  = ce_fn(logits, labels)
        loss     = loss_tri + 0.5 * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_tri  += loss_tri.item()
        total_ce   += loss_ce.item()
        n += 1

    return total_loss/n, total_tri/n, total_ce/n


@torch.no_grad()
def val_epoch(model, loader, margin, device):
    model.eval()
    total_loss = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        embeds = model(imgs)
        loss = triplet_hard(embeds, labels, margin)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default=DATA_DIR)
    parser.add_argument('--output',  default=OUTPUT_DIR)
    parser.add_argument('--epochs',  type=int,   default=EPOCHS)
    parser.add_argument('--batch-p', type=int,   default=P,
                        help='Identidades por batch (default: 8)')
    parser.add_argument('--batch-k', type=int,   default=K,
                        help='Imagens por identidade (default: 4)')
    parser.add_argument('--lr',      type=float, default=LR)
    parser.add_argument('--margin',  type=float, default=MARGIN)
    parser.add_argument('--embed',   type=int,   default=EMBED_DIM)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🏐 TREINO DO MODELO ReID")
    print(f"{'='*50}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_p}P × {args.batch_k}K = {args.batch_p*args.batch_k}")
    print(f"  LR:      {args.lr}  |  Margin: {args.margin}")

    os.makedirs(args.output, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"\n  A carregar datasets...")
    train_ds = ReIDDataset(os.path.join(args.data, 'train'), train_tf)
    val_ds   = ReIDDataset(os.path.join(args.data, 'val'),   val_tf)

    train_sampler = PKSampler(train_ds, args.batch_p, args.batch_k)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                               num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_ds, batch_size=32, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    print(f"\n  A inicializar modelo (ResNet50 pretrained)...")
    model     = ReIDModel(train_ds.n_ids, args.embed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=20, gamma=0.5)

    best_val  = float('inf')
    best_path = os.path.join(args.output, 'best.pt')

    print(f"\n  {'Epoch':>6}  {'Train':>8}  {'Tri':>8}  {'CE':>8}  {'Val':>8}  {'LR':>8}")
    print(f"  {'-'*52}")

    for epoch in range(1, args.epochs + 1):
        train_loss, tri_loss, ce_loss = train_epoch(
            model, train_loader, optimizer, args.margin, device)
        val_loss = val_epoch(model, val_loader, args.margin, device)
        scheduler.step()

        mark = ' ✓' if val_loss < best_val else ''
        print(f"  {epoch:>6}  {train_loss:>8.4f}  {tri_loss:>8.4f}"
              f"  {ce_loss:>8.4f}  {val_loss:>8.4f}"
              f"  {scheduler.get_last_lr()[0]:>8.2e}{mark}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'embed_dim': args.embed,
                'val_loss':  val_loss,
            }, best_path)

    # ── Export ONNX ───────────────────────────────────────────────────────────
    print(f"\n  A exportar para ONNX...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy   = torch.zeros(1, 3, IMG_H, IMG_W, device=device)
    onnx_path = os.path.join(args.output, 'best.onnx')
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=12, verbose=False)

    print(f"\n{'='*50}")
    print(f"✅ Treino completo!")
    print(f"   Melhor val loss: {best_val:.4f}  (epoch {ckpt['epoch']})")
    print(f"   Modelo PyTorch:  {best_path}")
    print(f"   Modelo ONNX:     {onnx_path}")
    print(f"\n➡️  Próximo passo:")
    print(f"   python training/integrate_reid.py")


if __name__ == '__main__':
    main()
