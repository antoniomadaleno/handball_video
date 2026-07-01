"""
Análise da deteção da bola num vídeo.

Corre o detector YOLO em todos os frames e produz:
  - Estatísticas (% deteções, gaps, confiança)
  - Gráfico com timeline, histograma e gaps
  - Vídeo anotado com as deteções da bola

Uso:
    python src/analyze_ball.py
    python src/analyze_ball.py --video videos/<x>.mp4 --conf 0.1
"""
import cv2
import json
import numpy as np
import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')   # sem GUI — só guarda PNG
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import HandballDetector
from tqdm import tqdm

VIDEO_PATH = 'videos/video_teste_1.mp4'
MODEL_PATH = 'training/runs/handball/v7_combined/weights/best.pt'
CONF       = 0.15   # threshold baixo para analisar a sensibilidade

# ── SAHI: deteção em patches (slicing) ────────────────────────────────────────

def sliced_detect(detector, frame_rgb, slice_size=640, overlap=0.3,
                  ball_only=True):
    """
    Corre o detector em patches sobrepostos da imagem.
    A bola pequena fica relativamente maior em cada patch, melhorando o recall.

    slice_size: tamanho de cada patch (px)
    overlap:    fração de sobreposição entre patches (0.0-0.5)
    ball_only:  só devolve deteções da bola (classe 0)
    """
    h, w = frame_rgb.shape[:2]
    stride = int(slice_size * (1 - overlap))

    all_boxes, all_scores, all_classes = [], [], []

    y = 0
    while y < h:
        x = 0
        while x < w:
            y2 = min(y + slice_size, h)
            x2 = min(x + slice_size, w)
            y1 = max(0, y2 - slice_size)
            x1 = max(0, x2 - slice_size)

            patch = frame_rgb[y1:y2, x1:x2]
            boxes, scores, classes = detector.detect(patch)

            for b, s, c in zip(boxes, scores, classes):
                if ball_only and int(c) != 0:
                    continue
                # offset para coordenadas do frame inteiro
                all_boxes.append([
                    float(b[0] + x1), float(b[1] + y1),
                    float(b[2] + x1), float(b[3] + y1),
                ])
                all_scores.append(float(s))
                all_classes.append(int(c))

            if x2 >= w:
                break
            x += stride
        if y2 >= h:
            break
        y += stride

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # NMS para fundir deteções duplicadas entre patches
    boxes_np   = np.array(all_boxes, dtype=np.float32)
    scores_np  = np.array(all_scores, dtype=np.float32)
    classes_np = np.array(all_classes, dtype=int)

    # converte xyxy → xywh para NMSBoxes
    xywh = boxes_np.copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores_np.tolist(),
                            score_threshold=0.0, nms_threshold=0.4)
    if len(keep) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    keep = np.array(keep).flatten()
    return boxes_np[keep], scores_np[keep], classes_np[keep]


def analyze(video_path, model_path, conf_threshold, write_video=True,
            use_sahi=False, slice_size=640, overlap=0.3):
    detector = HandballDetector(model_path, conf_threshold=conf_threshold)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mode = f"SAHI ({slice_size}px, overlap={overlap})" if use_sahi else "normal"
    print(f"\n📹 {video_path}  |  {total} frames  |  {fps:.1f} fps")
    print(f"   Conf threshold: {conf_threshold}  |  Modo: {mode}\n")

    # detecções da bola por frame
    detections = []   # [(frame_idx, bbox, conf), ...]
    ball_count_per_frame = []   # nº de deteções de bola por frame

    suffix = '_sahi' if use_sahi else ''
    writer = None
    if write_video:
        out_path = f"output/ball_analysis{suffix}_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    for fi in tqdm(range(total), desc="Análise"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if use_sahi:
            boxes, scores, classes = sliced_detect(
                detector, frame_rgb,
                slice_size=slice_size, overlap=overlap, ball_only=True,
            )
        else:
            boxes, scores, classes = detector.detect(frame_rgb)

        ball_idx = np.where(classes == 0)[0]
        ball_count_per_frame.append(len(ball_idx))

        for i in ball_idx:
            b, s = boxes[i], float(scores[i])
            detections.append({
                'frame':      fi,
                'bbox':       [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                'center':     [float((b[0]+b[2])/2), float((b[1]+b[3])/2)],
                'confidence': s,
            })
            if writer:
                x1, y1, x2, y2 = [int(v) for v in b]
                col = (0, 255, 255) if s >= 0.3 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"ball {s:.2f}", (x1, max(y1-6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        if writer:
            cv2.rectangle(frame, (0, 0), (210, 30), (0, 0, 0), -1)
            txt = f"F {fi}/{total-1}  balls={len(ball_idx)}"
            cv2.putText(frame, txt, (5, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    return detections, ball_count_per_frame, total, fps


def report(detections, counts, total, fps, conf_threshold, video_path, suffix=''):
    n_frames_with_ball   = sum(1 for c in counts if c > 0)
    n_frames_no_ball     = total - n_frames_with_ball
    pct_with_ball        = n_frames_with_ball / total * 100

    if detections:
        confs = [d['confidence'] for d in detections]
        mean_conf = np.mean(confs)
        median_conf = np.median(confs)
    else:
        mean_conf = median_conf = 0

    # gaps: sequências consecutivas sem deteção
    gaps = []
    cur = 0
    for c in counts:
        if c == 0:
            cur += 1
        else:
            if cur > 0:
                gaps.append(cur)
                cur = 0
    if cur > 0:
        gaps.append(cur)

    print("\n" + "="*55)
    print("  📊 ANÁLISE DA DETEÇÃO DA BOLA")
    print("="*55)
    print(f"\n  Total de frames:        {total}")
    print(f"  Frames com bola:        {n_frames_with_ball}  ({pct_with_ball:.1f}%)")
    print(f"  Frames sem bola:        {n_frames_no_ball}  ({100-pct_with_ball:.1f}%)")
    print(f"  Total de deteções:      {len(detections)}")
    print(f"\n  Confiança média:        {mean_conf:.3f}")
    print(f"  Confiança mediana:      {median_conf:.3f}")
    if confs:
        print(f"  Confiança min/max:      {min(confs):.3f} / {max(confs):.3f}")
    print(f"\n  Nº de gaps:             {len(gaps)}")
    if gaps:
        print(f"  Gap médio:              {np.mean(gaps):.1f} frames  ({np.mean(gaps)/fps:.2f}s)")
        print(f"  Gap máximo:             {max(gaps)} frames  ({max(gaps)/fps:.2f}s)")

    print("\n  Recomendação:")
    if pct_with_ball >= 70:
        print(f"  ✅ Deteção razoável ({pct_with_ball:.0f}%) — usar YOLO + interpolação")
    elif pct_with_ball >= 40:
        print(f"  ⚠️  Deteção fraca ({pct_with_ball:.0f}%) — tracking dedicado (Kalman)")
    else:
        print(f"  ❌ Deteção insuficiente ({pct_with_ball:.0f}%) — re-treinar modelo")

    # ── gráfico ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))

    # timeline
    timeline = [1 if c > 0 else 0 for c in counts]
    axes[0].fill_between(range(total), 0, timeline, step='pre', color='#27ae60', alpha=0.7)
    axes[0].set_xlim(0, total)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Bola detetada")
    axes[0].set_title(f"Timeline — {pct_with_ball:.1f}% dos frames com bola")
    axes[0].set_yticks([0, 1])

    # histograma de confianças
    if confs:
        axes[1].hist(confs, bins=30, color='#3498db', edgecolor='black')
        axes[1].axvline(conf_threshold, color='red', linestyle='--',
                        label=f'threshold ({conf_threshold})')
        axes[1].set_xlabel("Confiança")
        axes[1].set_ylabel("Nº deteções")
        axes[1].set_title(f"Distribuição de confiança  (média={mean_conf:.2f})")
        axes[1].legend()

    # gaps
    if gaps:
        axes[2].hist(gaps, bins=min(30, len(set(gaps))), color='#e74c3c', edgecolor='black')
        axes[2].set_xlabel("Comprimento do gap (frames)")
        axes[2].set_ylabel("Nº ocorrências")
        axes[2].set_title(f"Distribuição de gaps  (máx={max(gaps)} frames)")

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_png = f"output/ball_analysis{suffix}_{base}.png"
    plt.savefig(out_png, dpi=100)
    print(f"\n💾 Gráfico:    {out_png}")
    plt.close()

    # JSON com as deteções
    out_json = f"output/ball_detections{suffix}_{base}.json"
    with open(out_json, 'w') as f:
        json.dump({
            'video':            video_path,
            'total_frames':     total,
            'conf_threshold':   conf_threshold,
            'pct_with_ball':    pct_with_ball,
            'detections':       detections,
        }, f, indent=2)
    print(f"💾 Detecções: {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=VIDEO_PATH)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--conf',  type=float, default=CONF,
                        help='Threshold de confiança (default: 0.15)')
    parser.add_argument('--no-video', action='store_true',
                        help='Não gerar vídeo anotado (mais rápido)')
    parser.add_argument('--sahi', action='store_true',
                        help='Usar SAHI (deteção em patches sobrepostos)')
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Tamanho de cada patch SAHI (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.3,
                        help='Sobreposição entre patches SAHI (default: 0.3)')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    detections, counts, total, fps = analyze(
        args.video, args.model, args.conf,
        write_video=not args.no_video,
        use_sahi=args.sahi,
        slice_size=args.slice_size,
        overlap=args.overlap,
    )
    suffix = '_sahi' if args.sahi else ''
    report(detections, counts, total, fps, args.conf, args.video, suffix=suffix)


if __name__ == '__main__':
    main()
