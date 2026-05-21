#!/usr/bin/env python3
"""
Extracção automática de dados ReID usando ByteTrack como anotador.

Para cada vídeo de treino:
  1. Extrai um clip de CLIP_SEC segundos
  2. Corre detecção + ByteTrack
  3. Mantém tracks com >= MIN_FRAMES (fiáveis)
  4. Amostra CROPS_PER_TRACK crops por track
  5. Guarda em reid_dataset/<identidade>/

Sem anotação manual — o tracker faz o trabalho.

Output:
  training/reid_dataset/
    train/
      v1_track_003/  ← identidade = jogador
        crop_00.jpg
        crop_01.jpg
        ...
      v1_track_007/
        ...
    val/
      ...

Uso:
  python training/extract_reid_auto.py
  python training/extract_reid_auto.py --clip-sec 90 --min-frames 40
"""
import cv2
import json
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from detector import HandballDetector
from tracker  import HandballTracker

# ── Configuração ──────────────────────────────────────────────────────────────
VIDEO_DIR   = 'training/videos'
OUTPUT_DIR  = 'training/reid_dataset'
MODEL_PATH  = 'training/runs/handball/v3_final/weights/best.pt'

CLIP_SEC       = 60    # segundos de cada vídeo a processar
CLIP_START_SEC = 30    # começar aqui (saltar intro/aquecimento)
MIN_FRAMES     = 30    # mínimo de frames num track para ser considerado fiável
CROPS_PER_TRACK = 20   # crops a extrair por track (amostrados uniformemente)
VAL_RATIO      = 0.2   # 20% dos tracks para validação

VIDEOS = [
    'v1_ehf_cl_final',
    'v2_ehf_cl_semi_final',
    'v3_olympic_games',
    'v4_PO02_viseu_portugal',
    'v5_PO01_vfc_portugal',
    'v6_PO01_xico_portugal',
    'v7_PO02_aac_portugal',
]
# ─────────────────────────────────────────────────────────────────────────────


def extract_crops_from_track(frames_data, cap, track_id, n_crops):
    """
    Amostra n_crops frames uniformemente do track e extrai crops.
    Devolve lista de imagens BGR (128×256).
    """
    if len(frames_data) < 2:
        return []

    step   = max(1, len(frames_data) // n_crops)
    sample = frames_data[::step][:n_crops]
    crops  = []

    for fd in sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fd['frame'])
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1, x2, y2 = [int(v) for v in fd['bbox']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (128, 256)))

    return crops


def process_video(video_name, video_path, detector, out_root,
                  clip_start_sec, clip_sec, min_frames, crops_per_track):
    """
    Corre tracking num clip do vídeo e extrai crops por identidade.
    Devolve lista de (identity_name, [crop_img, ...]).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Não abriu: {video_path}")
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(clip_start_sec * fps)
    end_frame    = min(total_frames, int((clip_start_sec + clip_sec) * fps))
    n_frames     = end_frame - start_frame

    print(f"\n  📹 {video_name}  [{clip_start_sec}s → {clip_start_sec+clip_sec}s]"
          f"  ({n_frames} frames)")

    tracker      = HandballTracker(track_buffer=60, match_thresh=0.8)
    trajectories = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for local_id in tqdm(range(n_frames), desc=f"  Tracking", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        tracks = tracker.update(boxes, scores, classes, frame_rgb)

        for track in tracks:
            x1, y1, x2, y2, tid, conf, cls = track
            tid = int(tid)
            if int(cls) != 1:   # só jogadores
                continue
            if tid not in trajectories:
                trajectories[tid] = []
            trajectories[tid].append({
                'frame': start_frame + local_id,
                'bbox':  [float(x1), float(y1), float(x2), float(y2)],
                'conf':  float(conf),
            })

    # Filtrar tracks fiáveis
    valid = {tid: frs for tid, frs in trajectories.items()
             if len(frs) >= min_frames}
    print(f"  Tracks totais: {len(trajectories)}  →  "
          f"válidos (≥{min_frames} frames): {len(valid)}")

    # Extrair crops
    identities = []
    for tid, frs in valid.items():
        identity_name = f"{video_name[:2]}_track_{tid:04d}"
        crops = extract_crops_from_track(frs, cap, tid, crops_per_track)
        if len(crops) >= 4:   # mínimo 4 crops por identidade
            identities.append((identity_name, crops))

    cap.release()
    print(f"  Identidades com ≥4 crops: {len(identities)}")
    return identities


def save_dataset(all_identities, out_root, val_ratio):
    """
    Guarda crops organizados em train/ e val/.
    Split: últimos val_ratio% das identidades → val.
    """
    n_val   = max(1, int(len(all_identities) * val_ratio))
    n_train = len(all_identities) - n_val

    splits = (
        [('train', all_identities[:n_train])] +
        [('val',   all_identities[n_train:])]
    )

    stats = {}
    for split, identities in splits:
        split_dir = os.path.join(out_root, split)
        n_crops_total = 0
        for identity_name, crops in identities:
            id_dir = os.path.join(split_dir, identity_name)
            os.makedirs(id_dir, exist_ok=True)
            for i, crop in enumerate(crops):
                cv2.imwrite(os.path.join(id_dir, f'crop_{i:02d}.jpg'), crop)
            n_crops_total += len(crops)
        stats[split] = {'identities': len(identities), 'crops': n_crops_total}

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extracção automática de dados ReID')
    parser.add_argument('--clip-sec',        type=int,   default=CLIP_SEC)
    parser.add_argument('--clip-start',      type=int,   default=CLIP_START_SEC)
    parser.add_argument('--min-frames',      type=int,   default=MIN_FRAMES)
    parser.add_argument('--crops-per-track', type=int,   default=CROPS_PER_TRACK)
    parser.add_argument('--val-ratio',       type=float, default=VAL_RATIO)
    parser.add_argument('--output',          default=OUTPUT_DIR)
    parser.add_argument('--video',           default=None,
                        help='Processar só este vídeo')
    args = parser.parse_args()

    print("🏐 EXTRACÇÃO AUTOMÁTICA DE DADOS ReID")
    print("=" * 55)
    print(f"  Clip: {args.clip_start}s → {args.clip_start+args.clip_sec}s "
          f"({args.clip_sec}s por vídeo)")
    print(f"  Min frames por track: {args.min_frames}")
    print(f"  Crops por track: {args.crops_per_track}")

    print(f"\n  A carregar modelo de detecção...")
    detector = HandballDetector(MODEL_PATH, conf_threshold=0.2)

    videos = [args.video] if args.video else VIDEOS
    all_identities = []

    for video_name in videos:
        video_path = os.path.join('training/videos', video_name + '.mp4')
        if not os.path.exists(video_path):
            print(f"\n  ❌ Não encontrado: {video_path}")
            continue
        identities = process_video(
            video_name, video_path, detector,
            args.output,
            args.clip_start, args.clip_sec,
            args.min_frames, args.crops_per_track)
        all_identities.extend(identities)

    if not all_identities:
        print("\n❌ Nenhuma identidade extraída.")
        return

    print(f"\n{'='*55}")
    print(f"  Total de identidades: {len(all_identities)}")
    print(f"  Total de crops: {sum(len(c) for _,c in all_identities)}")
    print(f"\n  A guardar dataset...")

    stats = save_dataset(all_identities, args.output, args.val_ratio)

    print(f"\n✅ Dataset guardado em: {args.output}/")
    print(f"   train: {stats['train']['identities']} identidades, "
          f"{stats['train']['crops']} crops")
    print(f"   val:   {stats['val']['identities']} identidades, "
          f"{stats['val']['crops']} crops")
    print(f"\n➡️  Próximo passo:")
    print(f"   python training/train_reid.py")


if __name__ == '__main__':
    main()
