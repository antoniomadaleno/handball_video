#!/usr/bin/env python3
"""
Visualiza trajectórias dos players no vídeo
"""
import cv2
import json
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

# Cores para tracks (BGR)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 165, 0), (255, 20, 147)
]

def get_color(track_id):
    """Cor consistente por track_id"""
    return COLORS[int(track_id) % len(COLORS)]

def visualize_tracks(trajectories_path, video_path, output_path, 
                     trail_length=30, show_ids=True, show_boxes=True):
    """
    Visualiza trajectórias no vídeo
    
    Args:
        trajectories_path: JSON com trajectories
        video_path: Vídeo original
        output_path: Vídeo output
        trail_length: Número de frames do rastro
        show_ids: Mostrar track IDs
        show_boxes: Mostrar bounding boxes
    """
    
    print("🎬 VISUALIZADOR DE TRAJECTÓRIAS")
    print("=" * 60)
    
    # Load trajectories
    print(f"📂 A carregar: {trajectories_path}")
    with open(trajectories_path, 'r') as f:
        data = json.load(f)
    
    # Organizar por frame
    frame_data = defaultdict(list)
    all_tracks = {}
    
    for track_id, track_info in data.items():
        all_tracks[track_id] = []
        for frame_info in track_info['frames']:
            frame_num = frame_info['frame']
            frame_data[frame_num].append({
                'track_id': track_id,
                'bbox': frame_info['bbox'],
                'center': frame_info['center'],
                'confidence': frame_info['confidence'],
                'class': track_info['class']
            })
            all_tracks[track_id].append({
                'frame': frame_num,
                'center': frame_info['center']
            })
    
    print(f"✅ {len(data)} tracks carregados")
    print(f"📊 {len(frame_data)} frames com deteções")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir vídeo: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Vídeo: {width}x{height} @ {fps}fps")
    print(f"   Total frames: {total_frames}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🎨 A criar visualização...")
    print(f"   Trail length: {trail_length} frames")
    print(f"   Show IDs: {show_ids}")
    print(f"   Show boxes: {show_boxes}")
    print()
    
    # Process frames
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw trails (rastros das trajectórias)
        for track_id, positions in all_tracks.items():
            color = get_color(track_id)
            
            # Filtrar posições relevantes (últimos trail_length frames)
            relevant_positions = [
                p for p in positions 
                if p['frame'] <= frame_num and p['frame'] > frame_num - trail_length
            ]
            
            # Desenhar linha do rastro
            if len(relevant_positions) > 1:
                points = [tuple(map(int, p['center'])) for p in relevant_positions]
                
                # Linha com fade (mais transparente no passado)
                for i in range(len(points) - 1):
                    alpha = (i + 1) / len(points)  # 0 a 1
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, points[i], points[i+1], color, thickness)
        
        # Draw current frame detections
        if frame_num in frame_data:
            for det in frame_data[frame_num]:
                track_id = det['track_id']
                bbox = det['bbox']
                center = det['center']
                conf = det['confidence']
                color = get_color(track_id)
                
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = map(int, center)
                
                # Bounding box
                if show_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Centro (círculo maior)
                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
                
                # ID e confiança
                if show_ids:
                    label = f"ID:{track_id}"
                    label_conf = f"{conf:.2f}"
                    
                    # Background para texto
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1-25), (x1+tw+5, y1), color, -1)
                    
                    cv2.putText(frame, label, (x1+2, y1-12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, label_conf, (x1+2, y1-2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Frame info
        info_text = f"Frame: {frame_num}/{total_frames-1} | Tracks ativos: {len(frame_data.get(frame_num, []))}"
        cv2.rectangle(frame, (10, 10), (500, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_num % 50 == 0:
            print(f"   Processado: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Visualização completa!")
    print(f"📁 Guardado: {output_path}")
    print(f"   Tamanho: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualiza trajectórias de handball')
    parser.add_argument('--trajectories', type=str, default='trajectories_zmerding.json',
                       help='Caminho do ficheiro JSON com trajectories')
    parser.add_argument('--video', type=str, default='video_teste_zmerding.mp4',
                       help='Caminho do vídeo original')
    parser.add_argument('--output', type=str, default='output_tracks.mp4',
                       help='Caminho do vídeo de output')
    parser.add_argument('--trail', type=int, default=30,
                       help='Comprimento do rastro em frames (default: 30)')
    parser.add_argument('--no-ids', action='store_true',
                       help='Não mostrar IDs')
    parser.add_argument('--no-boxes', action='store_true',
                       help='Não mostrar bounding boxes')
    
    args = parser.parse_args()
    
    visualize_tracks(
        trajectories_path=args.trajectories,
        video_path=args.video,
        output_path=args.output,
        trail_length=args.trail,
        show_ids=not args.no_ids,
        show_boxes=not args.no_boxes
    )