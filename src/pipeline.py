"""
pipeline.py - Complete Tracking Pipeline
"""
import cv2
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import HandballDetector
from tracker import HandballTracker
from tqdm import tqdm

VIDEO_PATH  = 'videos/video_teste_1.mp4'
MODEL_PATH  = 'training/runs/handball/v3_final/weights/best.pt'
OUTPUT_JSON = 'output/trajectories_video_teste_1.json'


def process_video(video_path=VIDEO_PATH, model_path=MODEL_PATH,
                  output_json=OUTPUT_JSON):

    print("🏐 HANDBALL TRACKING PIPELINE")
    print("=" * 50)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    detector = HandballDetector(model_path, conf_threshold=0.2)
    tracker  = HandballTracker(track_buffer=50, match_thresh=0.8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir: {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Vídeo: {video_path}")
    print(f"   FPS: {fps}  |  Frames: {total_frames}  |  Duração: {total_frames/fps:.1f}s\n")

    trajectories = {}

    for frame_id in tqdm(range(total_frames), desc="Tracking"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        tracks = tracker.update(boxes, scores, classes, frame_rgb)

        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            track_id = int(track_id)

            if track_id not in trajectories:
                trajectories[track_id] = {
                    'class': 'ball' if int(cls) == 0 else 'player',
                    'frames': []
                }

            trajectories[track_id]['frames'].append({
                'frame':      frame_id,
                'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                'center':     [float((x1+x2)/2), float((y1+y2)/2)],
                'confidence': float(conf),
            })

    cap.release()

    with open(output_json, 'w') as f:
        json.dump({str(k): v for k, v in trajectories.items()}, f, indent=2)

    n_players = sum(1 for t in trajectories.values() if t['class'] == 'player')
    n_balls   = sum(1 for t in trajectories.values() if t['class'] == 'ball')
    avg_len   = np.mean([len(t['frames']) for t in trajectories.values()])

    print(f"\n✅ Tracking completo!")
    print(f"   Tracks: {len(trajectories)}  ({n_players} jogadores, {n_balls} bolas)")
    print(f"   Comprimento médio: {avg_len:.1f} frames")
    print(f"💾 {output_json}")


if __name__ == '__main__':
    process_video()
