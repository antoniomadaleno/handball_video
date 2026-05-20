#!/usr/bin/env python3
"""Transforma trajectórias de pixels para coordenadas reais (metros)."""
import cv2
import numpy as np
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', default='output/trajectories_video_teste_1.json')
    parser.add_argument('--homography',   default='output/homography.json')
    parser.add_argument('--output',       default='output/trajectories_2d.json')
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    with open(args.homography) as f:
        H = np.array(json.load(f)['homography_matrix'], dtype=np.float64)

    print(f"🔄 A transformar {len(tracks)} tracks...")

    result = {}
    total, out_of_bounds = 0, 0

    for track_id, track in tracks.items():
        new_frames = []
        for fr in track['frames']:
            cx, cy = fr['center']
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            rx, ry = cv2.perspectiveTransform(pt, H)[0][0].tolist()
            total += 1
            if not (-2 <= rx <= 42 and -2 <= ry <= 22):
                out_of_bounds += 1
            new_frames.append({**fr, 'real_coords': [float(rx), float(ry)]})
        result[track_id] = {**track, 'frames': new_frames}

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    pct = out_of_bounds / total * 100 if total else 0
    print(f"✅ Transformação completa!")
    print(f"   Pontos fora dos limites: {out_of_bounds}/{total} ({pct:.1f}%)")
    if pct > 10:
        print("   ⚠️  Muitos pontos fora — verifica a homografia")
    print(f"💾 Guardado: {args.output}")


if __name__ == '__main__':
    main()
