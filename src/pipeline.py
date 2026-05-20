"""
pipeline.py - Tracking com BoT-SORT (ReID + motion).

BoT-SORT usa features de aparência para reconhecer jogadores mesmo
depois de oclusões, ao contrário do ByteTrack que usa só posição/IoU.
"""
import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

VIDEO_PATH  = 'videos/video_teste_1.mp4'
MODEL_PATH  = 'training/runs/handball/v3_final/weights/best.pt'
OUTPUT_JSON = 'output/trajectories_video_teste_1.json'
CONF        = 0.25   # confiança mínima de detecção
TRACKER     = 'botsort.yaml'   # BoT-SORT com ReID  (alternativa: 'bytetrack.yaml')


def process_video(video_path=VIDEO_PATH, model_path=MODEL_PATH,
                  output_json=OUTPUT_JSON, conf=CONF, tracker=TRACKER):

    print("🏐 HANDBALL TRACKING PIPELINE  (BoT-SORT + ReID)")
    print("=" * 55)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir: {video_path}")
        return
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"📹 {video_path}  |  {total_frames} frames  |  {fps:.1f} fps")
    print(f"🔍 Tracker: {tracker}  |  conf={conf}\n")

    trajectories = {}

    # model.track() faz detecção + tracking num só passo
    # stream=True processa frame a frame sem carregar tudo na memória
    results = model.track(
        source=video_path,
        persist=True,       # mantém estado do tracker entre frames
        tracker=tracker,
        conf=conf,
        iou=0.5,
        stream=True,
        verbose=False,
    )

    for frame_id, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes    = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        classes  = result.boxes.cls.cpu().numpy().astype(int)
        confs    = result.boxes.conf.cpu().numpy()

        for box, tid, cls, cf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = box
            if tid not in trajectories:
                trajectories[tid] = {
                    'class': 'ball' if cls == 0 else 'player',
                    'frames': []
                }
            trajectories[tid]['frames'].append({
                'frame':      frame_id,
                'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                'center':     [float((x1+x2)/2), float((y1+y2)/2)],
                'confidence': float(cf),
            })

        if frame_id % 50 == 0:
            print(f"   Frame {frame_id}/{total_frames}  tracks activos: {len(result.boxes.id)}")

    with open(output_json, 'w') as f:
        json.dump({str(k): v for k, v in trajectories.items()}, f, indent=2)

    n_players = sum(1 for t in trajectories.values() if t['class'] == 'player')
    n_balls   = sum(1 for t in trajectories.values() if t['class'] == 'ball')
    avg_len   = np.mean([len(t['frames']) for t in trajectories.values()])

    print(f"\n✅ Tracking completo!")
    print(f"   Tracks totais: {len(trajectories)}  ({n_players} jogadores, {n_balls} bolas)")
    print(f"   Comprimento médio: {avg_len:.1f} frames")
    print(f"💾 {output_json}")


if __name__ == '__main__':
    process_video()
