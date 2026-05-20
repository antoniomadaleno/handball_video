#!/usr/bin/env python3
"""
Extrai estatísticas de trajectories.json
Estrutura: {track_id: {class: "...", frames: [...]}}
"""
import json
import sys
from collections import defaultdict

def extract_statistics(filepath):
    """Extrai estatísticas completas"""
    
    print("📊 A analisar trajectories.json...\n")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    total_tracks = len(data)
    print(f"🎯 Total de tracks: {total_tracks}\n")
    
    # Análise por track
    track_stats = {}
    ball_track_id = None
    all_frames = set()
    
    for track_id, track_data in data.items():
        obj_class = track_data.get('class', 'unknown')
        frames = track_data.get('frames', [])
        
        # Recolher frames
        frame_nums = [f['frame'] for f in frames]
        all_frames.update(frame_nums)
        
        # Calcular estatísticas
        confidences = [f['confidence'] for f in frames]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        track_stats[track_id] = {
            'class': obj_class,
            'total_detections': len(frames),
            'frames': sorted(frame_nums),
            'avg_confidence': avg_conf,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0
        }
        
        if obj_class == 'ball':
            ball_track_id = track_id
    
    # Estatísticas globais
    total_frames = len(all_frames)
    min_frame = min(all_frames) if all_frames else 0
    max_frame = max(all_frames) if all_frames else 0
    
    print("="*60)
    print("📈 ESTATÍSTICAS GLOBAIS")
    print("="*60)
    print(f"Frames no vídeo: {min_frame} a {max_frame} (total: {max_frame - min_frame + 1})")
    print(f"Frames com deteções: {total_frames}")
    
    # Estatísticas por classe
    players = [tid for tid, stats in track_stats.items() if stats['class'] == 'player']
    balls = [tid for tid, stats in track_stats.items() if stats['class'] == 'ball']
    
    print(f"\n📊 Por classe:")
    print(f"   Players: {len(players)} tracks")
    print(f"   Bola: {len(balls)} track(s)")
    
    # Detalhes de cada track
    print(f"\n🎯 DETALHES DOS TRACKS:")
    print("="*60)
    
    for track_id in sorted(track_stats.keys(), key=int):
        stats = track_stats[track_id]
        print(f"\n📍 Track {track_id} ({stats['class'].upper()})")
        print(f"   Deteções: {stats['total_detections']}")
        print(f"   Confiança: média={stats['avg_confidence']:.3f}, "
              f"min={stats['min_confidence']:.3f}, max={stats['max_confidence']:.3f}")
        print(f"   Frames: {stats['frames'][0]} a {stats['frames'][-1]}")
        
        # Calcular gaps para esta track
        gaps = []
        for i in range(len(stats['frames']) - 1):
            gap = stats['frames'][i+1] - stats['frames'][i] - 1
            if gap > 0:
                gaps.append(gap)
        
        if gaps:
            print(f"   Gaps: {len(gaps)} gaps, "
                  f"médio={sum(gaps)/len(gaps):.1f}, "
                  f"máximo={max(gaps)}")
            
            # Mostrar maiores gaps
            large_gaps = [(g, i) for i, g in enumerate(gaps) if g > 5]
            if large_gaps:
                print(f"   ⚠️  Gaps >5 frames:")
                for gap_size, idx in sorted(large_gaps, reverse=True)[:3]:
                    start = stats['frames'][idx]
                    end = stats['frames'][idx + 1]
                    print(f"      {gap_size} frames: entre frame {start} e {end}")
    
    # Análise específica da bola
    if ball_track_id:
        print(f"\n" + "="*60)
        print("🏐 ANÁLISE DA BOLA")
        print("="*60)
        ball_stats = track_stats[ball_track_id]
        ball_frames = ball_stats['frames']
        
        print(f"Track ID: {ball_track_id}")
        print(f"Deteções: {ball_stats['total_detections']}")
        print(f"Cobertura: {ball_frames[0]} a {ball_frames[-1]} "
              f"({len(ball_frames)}/{ball_frames[-1]-ball_frames[0]+1} frames)")
        print(f"Taxa de deteção: {len(ball_frames)/(ball_frames[-1]-ball_frames[0]+1)*100:.1f}%")
        
        # Gaps na bola
        ball_gaps = []
        for i in range(len(ball_frames) - 1):
            gap = ball_frames[i+1] - ball_frames[i] - 1
            if gap > 0:
                ball_gaps.append((gap, ball_frames[i], ball_frames[i+1]))
        
        if ball_gaps:
            print(f"\n⚠️  GAPS NA BOLA: {len(ball_gaps)} gaps")
            print(f"   Gap médio: {sum(g[0] for g in ball_gaps)/len(ball_gaps):.1f} frames")
            print(f"   Gap máximo: {max(g[0] for g in ball_gaps)} frames")
            
            # Maiores gaps
            large_gaps = [g for g in ball_gaps if g[0] > 3]
            if large_gaps:
                print(f"\n   📍 Gaps que precisam interpolação (>3 frames):")
                for gap_size, start, end in sorted(large_gaps, reverse=True):
                    print(f"      {gap_size} frames: {start} → {end}")
    else:
        print(f"\n⚠️  AVISO: Nenhuma track de bola encontrada!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extract_stats.py trajectories.json")
        sys.exit(1)
    
    extract_statistics(sys.argv[1])