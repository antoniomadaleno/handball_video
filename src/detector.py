"""
detector.py - Handball Detection Wrapper
"""
from ultralytics import YOLO
import numpy as np

class HandballDetector:
    def __init__(self, model_path, conf_threshold=0.2):
        print(f"🏐 A carregar modelo: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print(f"✅ Modelo carregado! Conf threshold: {conf_threshold}")
    
    def detect(self, frame):
        # Run detection
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        # Parse results
        if len(results.boxes) == 0:
            # Sem detecções
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        
        boxes = results.boxes.xyxy.cpu().numpy()      # (N, 4)
        scores = results.boxes.conf.cpu().numpy()     # (N,)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,)
        
        return boxes, scores, classes


# TESTE
if __name__ == '__main__':
    import cv2
    
    # Carrega detector
    detector = HandballDetector('runs/handball/v3_final/weights/best.pt')
    
    # Testa num frame
    video_path = 'video_teste_27_35.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Converte BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteta
        boxes, scores, classes = detector.detect(frame_rgb)
        
        print(f"\n📊 Resultados frame 0:")
        print(f"   Detecções: {len(boxes)}")
        print(f"   Players: {(classes == 1).sum()}")
        print(f"   Balls: {(classes == 0).sum()}")
        print(f"   Scores: min={scores.min():.2f}, max={scores.max():.2f}")
    else:
        print("❌ Erro ao ler vídeo!")