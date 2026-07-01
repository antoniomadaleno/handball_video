"""
pipeline_sam2.py - Tracking Pipeline usando SAM2

Fluxo:
  1. Extrai frames do vídeo para diretório temporário
  2. YOLOv8l procura o melhor frame de inicialização
  3. SAM2 propaga em chunks (evita OOM em vídeos longos)
  4. Converte máscaras → bboxes/centros e guarda JSON

Output: output/trajectories_<nome_video>.json
"""
import cv2
import json
import numpy as np
import os
import sys
import tempfile
import shutil

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import HandballDetector

VIDEO_PATH  = 'videos/video_teste_1.mp4'
MODEL_PATH  = 'training/runs/handball/v7_combined/weights/best.pt'
OUTPUT_JSON = 'output/trajectories_video_teste_1.json'
SAM2_CFG    = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
SAM2_CKPT   = 'training/sam2_weights/sam2.1_hiera_base_plus.pt'

SCAN_FRAMES = 40     # frames a analisar para encontrar melhor init
CONF_THRESH = 0.20
CHUNK_SIZE  = 350    # frames por chunk (ajusta conforme VRAM disponível)


# ── utilitários ───────────────────────────────────────────────────────────────

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
    cap = cv2.VideoCapture(video_path)
    best_idx, best_dets, best_n = 0, [], -1
    for idx in range(scan_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        dets = nms([(b, float(s), int(c)) for b, s, c in zip(boxes, scores, classes)])
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


# ── ball tracking (YOLO por frame + Kalman) ───────────────────────────────────

def track_ball(video_path, detector, total_frames, ball_conf=0.15):
    """
    Deteta a bola frame-a-frame com YOLO (SAM2 é mau a propagar objectos
    pequenos rápidos). Junta posições + suaviza com Kalman, interpola gaps.
    """
    cap = cv2.VideoCapture(video_path)
    raw = {}

    for fi in tqdm(range(total_frames), desc="Ball YOLO"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        ball_idx = np.where(classes == 0)[0]
        if len(ball_idx) == 0:
            continue
        if len(ball_idx) > 1:
            best = ball_idx[np.argmax(scores[ball_idx])]
        else:
            best = ball_idx[0]
        if float(scores[best]) < ball_conf:
            continue
        raw[fi] = (boxes[best], float(scores[best]))

    cap.release()

    if not raw:
        print("   ⚠️  Sem deteções de bola.")
        return []

    print(f"   Bola detectada em {len(raw)}/{total_frames} frames "
          f"({len(raw)*100//total_frames}%)")

    # Kalman 2D com velocidade constante
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 5.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100

    first_fi = min(raw)
    bb0, _   = raw[first_fi]
    cx0, cy0 = (bb0[0]+bb0[2])/2, (bb0[1]+bb0[3])/2
    kf.statePost = np.array([[cx0], [cy0], [0.], [0.]], dtype=np.float32)

    bws = [b[2]-b[0] for b, _ in raw.values()]
    bhs = [b[3]-b[1] for b, _ in raw.values()]
    avg_w, avg_h = float(np.median(bws)), float(np.median(bhs))

    frames_out = []
    prev_fi = first_fi - 1
    MAX_GAP = 25

    for fi in range(first_fi, total_frames):
        kf.predict()

        if fi in raw:
            bbox, conf = raw[fi]
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            kf.correct(np.array([[cx], [cy]], dtype=np.float32))
            frames_out.append({
                'frame':        fi,
                'bbox':         [float(bbox[0]), float(bbox[1]),
                                 float(bbox[2]), float(bbox[3])],
                'center':       [float(cx), float(cy)],
                'confidence':   conf,
                'interpolated': False,
            })
            prev_fi = fi
        else:
            gap = fi - prev_fi
            if gap > MAX_GAP:
                continue
            px = float(kf.statePost[0])
            py = float(kf.statePost[1])
            frames_out.append({
                'frame':        fi,
                'bbox':         [px - avg_w/2, py - avg_h/2,
                                 px + avg_w/2, py + avg_h/2],
                'center':       [px, py],
                'confidence':   0.0,
                'interpolated': True,
            })

    print(f"   Total após Kalman+interpolação: {len(frames_out)} frames")
    return frames_out


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ── processamento por chunk ───────────────────────────────────────────────────

def process_chunk(predictor, device, frames_dir, chunk_start, chunk_end,
                  init_local_idx, boxes_cls):
    """
    Processa um chunk de frames. Devolve:
      results: {global_frame_idx: {obj_id: bbox [x1,y1,x2,y2]}}
      last_bboxes: {obj_id: bbox} do último frame com detecção
    """
    from sam2.build_sam import build_sam2_video_predictor  # já carregado

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        inference_state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )
        predictor.reset_state(inference_state)

        for obj_id, (bbox, cls) in enumerate(boxes_cls, start=1):
            x1, y1, x2, y2 = bbox
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_local_idx,
                obj_id=obj_id,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
            )

        results = {}
        last_bboxes = {}
        n_local = chunk_end - chunk_start

        for local_idx, obj_ids, masks in tqdm(
                predictor.propagate_in_video(
                    inference_state, start_frame_idx=init_local_idx),
                total=n_local - init_local_idx,
                desc=f"SAM2 [{chunk_start}→{chunk_end-1}]",
                leave=False):
            global_idx = chunk_start + local_idx
            results[global_idx] = {}
            for oid, mask in zip(obj_ids, masks):
                m = mask.squeeze().cpu().numpy() > 0.0
                bb = mask_to_bbox(m)
                if bb is not None:
                    results[global_idx][int(oid)] = bb
                    last_bboxes[int(oid)] = bb

    torch.cuda.empty_cache()
    return results, last_bboxes


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path=VIDEO_PATH, model_path=MODEL_PATH,
                  output_json=OUTPUT_JSON):

    print("🏐 HANDBALL TRACKING PIPELINE  (SAM2)")
    print("=" * 50)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    tmp_dir = tempfile.mkdtemp(prefix='sam2_frames_')
    try:
        # ── 1. extração ───────────────────────────────────────────────────────
        fps, total_frames, fw, fh = extract_frames(video_path, tmp_dir)
        print(f"📹 {video_path}  |  {total_frames} frames  |  {fps:.1f} fps\n")

        # ── 2. melhor frame de init ───────────────────────────────────────────
        print(f"   A procurar melhor frame de init (primeiros {SCAN_FRAMES})...")
        detector = HandballDetector(model_path, conf_threshold=CONF_THRESH)
        init_idx, init_dets = find_best_init_frame(video_path, detector)

        n_pl = sum(1 for _, _, c in init_dets if c == 1)
        n_ba = sum(1 for _, _, c in init_dets if c == 0)
        print(f"   Melhor frame: {init_idx}  →  {len(init_dets)} deteções "
              f"({n_pl} jogadores, {n_ba} bolas)\n")

        if not init_dets:
            print("❌ Nenhuma deteção nos primeiros frames.")
            return

        # SAM2 só trata dos JOGADORES. A bola é tracked à parte por YOLO
        # (SAM2 perde sempre objectos pequenos rápidos)
        player_dets = [d for d in init_dets if int(d[2]) == 1]
        if not player_dets:
            print("❌ Sem jogadores no frame de init.")
            return

        obj_classes = {i+1: 'player' for i in range(len(player_dets))}
        n_obj = len(player_dets)
        init_dets = player_dets   # daqui em diante só players

        # ── 3. SAM2 em chunks ─────────────────────────────────────────────────
        print(f"   A carregar SAM2...")
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=device)

        all_results = {}   # global_frame → {obj_id: bbox}
        last_bboxes = {}   # obj_id → bbox (posição no fim do chunk anterior)

        # boxes para o primeiro chunk: vêm do YOLO
        current_boxes = [(b, c) for b, _, c in init_dets]

        n_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"   {total_frames} frames  |  chunks de {CHUNK_SIZE}  |  {n_chunks} chunks\n")

        for chunk_i in range(n_chunks):
            chunk_start = chunk_i * CHUNK_SIZE
            chunk_end   = min(chunk_start + CHUNK_SIZE, total_frames)

            # pasta temporária com os frames deste chunk (numerados 0, 1, 2, ...)
            chunk_dir = os.path.join(tmp_dir, f'chunk_{chunk_i:04d}')
            os.makedirs(chunk_dir)
            for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
                link_or_copy(
                    os.path.join(tmp_dir, f'{global_i:06d}.jpg'),
                    os.path.join(chunk_dir, f'{local_i:06d}.jpg'),
                )

            # frame de init dentro deste chunk
            if chunk_i == 0:
                init_local = max(0, init_idx - chunk_start)
                boxes_cls  = current_boxes
            else:
                init_local = 0
                # re-inicializa com as posições do fim do chunk anterior
                boxes_cls = []
                for obj_id in range(1, n_obj + 1):
                    if obj_id in last_bboxes:
                        cls = 0 if obj_classes[obj_id] == 'ball' else 1
                        boxes_cls.append((last_bboxes[obj_id], cls))
                    else:
                        # objeto desapareceu — usa última posição conhecida
                        found = None
                        for fi in range(chunk_start - 1, -1, -1):
                            if fi in all_results and obj_id in all_results[fi]:
                                found = all_results[fi][obj_id]
                                break
                        if found:
                            cls = 0 if obj_classes[obj_id] == 'ball' else 1
                            boxes_cls.append((found, cls))

            chunk_results, last_bboxes = process_chunk(
                predictor, device, chunk_dir,
                chunk_start, chunk_end,
                init_local, boxes_cls,
            )
            all_results.update(chunk_results)
            shutil.rmtree(chunk_dir)

            n_done = chunk_end
            print(f"   Chunk {chunk_i+1}/{n_chunks} concluído  "
                  f"({n_done}/{total_frames} frames  "
                  f"{n_done*100//total_frames}%)")

        # ── 4. converter para trajectórias ────────────────────────────────────
        print("\n   A converter máscaras para trajectórias...")
        trajectories = {oid: {'class': cls, 'frames': []}
                        for oid, cls in obj_classes.items()}

        for global_idx in sorted(all_results):
            for obj_id, bb in all_results[global_idx].items():
                if obj_id not in trajectories:
                    continue
                x1, y1, x2, y2 = bb
                trajectories[obj_id]['frames'].append({
                    'frame':      global_idx,
                    'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                    'center':     [float((x1+x2)/2), float((y1+y2)/2)],
                    'confidence': 1.0,
                })

        # ── 5. Ball tracking (YOLO por frame + Kalman) ────────────────────────
        print("\n🏐 A tracking da bola (YOLO + Kalman)...")
        ball_frames = track_ball(video_path, detector, total_frames)
        if ball_frames:
            ball_track_id = n_obj + 1   # ID após o último jogador
            trajectories[ball_track_id] = {
                'class':  'ball',
                'frames': ball_frames,
            }

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  default=VIDEO_PATH)
    parser.add_argument('--output', default=OUTPUT_JSON)
    args = parser.parse_args()
    process_video(video_path=args.video, output_json=args.output)
