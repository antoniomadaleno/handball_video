"""
tracker.py - FIXED (passa img)
"""
import numpy as np
from boxmot import ByteTrack

class HandballTracker:
    def __init__(self, track_buffer=50, match_thresh=0.8):
        print(f"🎯 A inicializar tracker:")
        print(f"   track_buffer: {track_buffer}")
        print(f"   match_thresh: {match_thresh}")
        
        self.tracker = ByteTrack(
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=30
        )
        print("✅ Tracker inicializado!")
    
    def update(self, boxes, scores, classes, frame):  # ← ADICIONA FRAME
        """
        Args:
            boxes: numpy (N, 4)
            scores: numpy (N,)
            classes: numpy (N,)
            frame: numpy array (H, W, 3) - RGB frame
        """
        if len(boxes) == 0:
            return np.empty((0, 7))
        
        dets = np.column_stack([boxes, scores, classes])
        tracks = self.tracker.update(dets, frame)  # ← PASSA FRAME
        
        if len(tracks) > 0:
            return tracks[:, :7]
        else:
            return np.empty((0, 7))


# TESTE
if __name__ == '__main__':
    import cv2
    from detector import HandballDetector
    import os
    
    video_path = 'video_teste_27_35.mp4'
    
    if not os.path.exists(video_path):
        print(f"❌ Vídeo não encontrado!")
        exit(1)
    
    detector = HandballDetector('runs/handball/v3_final/weights/best.pt', conf_threshold=0.2)
    tracker = HandballTracker(track_buffer=50, match_thresh=0.8)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Erro ao abrir vídeo!")
        exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Vídeo: {fps} fps, {total} frames\n")
    
    print(f"🎬 A processar 10 frames...\n")
    
    for frame_id in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        tracks = tracker.update(boxes, scores, classes, frame_rgb)  # ← PASSA FRAME
        
        n_players = (tracks[:, 6] == 1).sum() if len(tracks) > 0 else 0
        n_balls = (tracks[:, 6] == 0).sum() if len(tracks) > 0 else 0
        unique_ids = len(np.unique(tracks[:, 4])) if len(tracks) > 0 else 0
        
        print(f"Frame {frame_id}: {len(tracks)} tracks ({n_players}P, {n_balls}B), {unique_ids} IDs")
    
    cap.release()
    print("\n✅ Teste completo!")