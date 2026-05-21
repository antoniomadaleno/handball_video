"""
pipeline_sam2.py - Tracking Pipeline usando SAM2

Fluxo:
  1. Extrai frames do vídeo para diretório temporário (SAM2 precisa de ficheiros)
  2. YOLOv8l procura o melhor frame de inicialização (máx. jogadores detetados)
  3. SAM2 propaga as máscaras para a frente e para trás a partir desse frame
  4. Converte máscaras → bboxes/centros e guarda JSON no mesmo formato do pipeline.py

Output: output/trajectories_video_teste_1.json
"""
import cv2
import json
import numpy as np
import os
import sys
import tempfile
import shutil

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import HandballDetector

VIDEO_PATH  = 'videos/video_teste_1.mp4'
MODEL_PATH  = 'training/runs/handball/v4_large/weights/best.pt'
OUTPUT_JSON = 'output/trajectories_video_teste_1.json'
SAM2_CFG    = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
SAM2_CKPT   = 'training/sam2_weights/sam2.1_hiera_base_plus.pt'

SCAN_FRAMES = 40    # quantos frames analisar para encontrar o melhor ponto de init
CONF_THRESH = 0.20  # igual ao ByteTrack pipeline


def extract_frames(video_path, out_dir):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   A extrair {total} frames para disco...")
    for i in tqdm(range(total), desc="Extração"):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), frame)

    cap.release()
    return fps, total, fw, fh


def nms(dets):
    """NMS simples por IoU > 0.5, mantém a deteção com maior confiança."""
    dets = sorted(dets, key=lambda x: -x[1])
    kept = []
    for b, s, c in dets:
        overlap = False
        for kb, ks, kc in kept:
            if kc != c:
                continue
            ix1 = max(b[0], kb[0]); iy1 = max(b[1], kb[1])
            ix2 = min(b[2], kb[2]); iy2 = min(b[3], kb[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = (b[2]-b[0])*(b[3]-b[1]) + (kb[2]-kb[0])*(kb[3]-kb[1]) - inter
            if union > 0 and inter / union > 0.5:
                overlap = True
                break
        if not overlap:
            kept.append((b, s, c))
    return kept


def find_best_init_frame(video_path, detector, scan_frames=SCAN_FRAMES):
    """
    Varre os primeiros scan_frames e devolve (frame_idx, deteções_NMS)
    do frame com mais jogadores detetados.
    """
    cap = cv2.VideoCapture(video_path)
    best_idx  = 0
    best_dets = []
    best_n    = -1

    for idx in range(scan_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        dets = [(b, float(s), int(c))
                for b, s, c in zip(boxes, scores, classes)]
        dets = nms(dets)
        n_players = sum(1 for _, _, c in dets if c == 1)
        if n_players > best_n:
            best_n, best_idx, best_dets = n_players, idx, dets

    cap.release()
    return best_idx, best_dets


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def process_video(video_path=VIDEO_PATH, model_path=MODEL_PATH,
                  output_json=OUTPUT_JSON):

    print("🏐 HANDBALL TRACKING PIPELINE  (SAM2)")
    print("=" * 50)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    # ── 1. Extração de frames ─────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix='sam2_frames_')
    try:
        fps, total_frames, fw, fh = extract_frames(video_path, tmp_dir)
        print(f"📹 {video_path}  |  {total_frames} frames  |  {fps:.1f} fps\n")

        # ── 2. Melhor frame de inicialização ──────────────────────────────────
        print(f"   A procurar melhor frame de init (primeiros {SCAN_FRAMES})...")
        detector  = HandballDetector(model_path, conf_threshold=CONF_THRESH)
        init_idx, init_dets = find_best_init_frame(video_path, detector)

        n_pl = sum(1 for _, _, c in init_dets if c == 1)
        n_ba = sum(1 for _, _, c in init_dets if c == 0)
        print(f"   Melhor frame: {init_idx}  →  {len(init_dets)} deteções "
              f"({n_pl} jogadores, {n_ba} bolas)")

        if not init_dets:
            print("❌ Nenhuma deteção nos primeiros frames. Verifica o modelo/vídeo.")
            return

        # ── 3. SAM2 ───────────────────────────────────────────────────────────
        print("\n   A carregar SAM2...")
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)

        obj_classes = {}  # obj_id → 'player' | 'ball'

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=tmp_dir)
            predictor.reset_state(inference_state)

            for obj_id, (bbox, conf, cls) in enumerate(init_dets, start=1):
                x1, y1, x2, y2 = bbox
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=init_idx,
                    obj_id=obj_id,
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                )
                obj_classes[obj_id] = 'ball' if cls == 0 else 'player'

            print(f"   {len(init_dets)} objetos inicializados no frame {init_idx}")

            frame_results = {}  # frame_idx → {obj_id: mask}

            # ── propagação para a frente ──────────────────────────────────────
            print(f"   Propagação → frente ({init_idx} a {total_frames-1})...")
            for frame_idx, obj_ids, masks in tqdm(
                    predictor.propagate_in_video(
                        inference_state, start_frame_idx=init_idx),
                    total=total_frames - init_idx,
                    desc="SAM2 →"):
                frame_results.setdefault(frame_idx, {})
                for oid, mask in zip(obj_ids, masks):
                    m = mask.squeeze().cpu().numpy() > 0.0
                    frame_results[frame_idx][int(oid)] = m

            # ── propagação para trás (se init_idx > 0) ───────────────────────
            if init_idx > 0:
                print(f"   Propagação ← trás (0 a {init_idx-1})...")
                for frame_idx, obj_ids, masks in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=init_idx,
                            reverse=True),
                        total=init_idx,
                        desc="SAM2 ←"):
                    frame_results.setdefault(frame_idx, {})
                    for oid, mask in zip(obj_ids, masks):
                        m = mask.squeeze().cpu().numpy() > 0.0
                        frame_results[frame_idx][int(oid)] = m

        # ── 4. Converter para trajectórias ────────────────────────────────────
        print("\n   A converter máscaras para trajectórias...")

        trajectories = {oid: {'class': cls_name, 'frames': []}
                        for oid, cls_name in obj_classes.items()}

        for frame_idx in tqdm(sorted(frame_results), desc="Converter"):
            for obj_id, mask in frame_results[frame_idx].items():
                if obj_id not in trajectories:
                    continue
                bb = mask_to_bbox(mask)
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                trajectories[obj_id]['frames'].append({
                    'frame':      frame_idx,
                    'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                    'center':     [float((x1+x2)/2), float((y1+y2)/2)],
                    'confidence': 1.0,
                })

        out = {str(k): v for k, v in trajectories.items() if v['frames']}

        with open(output_json, 'w') as f:
            json.dump(out, f, indent=2)

        n_players = sum(1 for t in out.values() if t['class'] == 'player')
        n_balls   = sum(1 for t in out.values() if t['class'] == 'ball')
        avg_len   = np.mean([len(t['frames']) for t in out.values()]) if out else 0

        print(f"\n✅ Tracking completo!")
        print(f"   Tracks: {len(out)}  ({n_players} jogadores, {n_balls} bolas)")
        print(f"   Comprimento médio: {avg_len:.1f} frames")
        print(f"💾 {output_json}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    process_video()
