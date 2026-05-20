#!/usr/bin/env python3
"""Visualizações 2D do campo: trajectórias e heatmap."""
import cv2
import numpy as np
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from court import draw_court, COURT_W, COURT_H

COLORS = [
    (255, 50,  50), (50, 255,  50), (50,  50, 255), (255, 255,  50),
    (255, 50, 255), (50, 255, 255), (255, 150,  50), (150,  50, 255),
    (50, 150, 255), (200, 200,  50), (255, 100, 100), (100, 255, 100),
    (100, 100, 255), (200, 150,  50), (150, 200,  50),
]


def save_trajectories(tracks, path, scale=20):
    court, _, _ = draw_court(scale)
    for tid, track in tracks.items():
        if track['class'] != 'player':
            continue
        color = COLORS[int(tid) % len(COLORS)]
        # só frames realmente detectados (ignora interpolados)
        real_frames = [f for f in track['frames']
                       if not f.get('interpolated', False)
                       and -1 <= f['real_coords'][0] <= 41
                       and -1 <= f['real_coords'][1] <= 21]
        # liga pontos consecutivos mas quebra a linha se houver gap > 5 frames
        for j in range(1, len(real_frames)):
            gap = real_frames[j]['frame'] - real_frames[j-1]['frame']
            if gap > 5:
                continue   # não liga pontos com gap — mostra só o que foi detectado
            p1 = (int(real_frames[j-1]['real_coords'][0]*scale),
                  int(real_frames[j-1]['real_coords'][1]*scale))
            p2 = (int(real_frames[j]['real_coords'][0]*scale),
                  int(real_frames[j]['real_coords'][1]*scale))
            cv2.line(court, p1, p2, color, 1)
        if real_frames:
            lx = int(real_frames[-1]['real_coords'][0]*scale)
            ly = int(real_frames[-1]['real_coords'][1]*scale)
            cv2.circle(court, (lx, ly), 5, color, -1)
            cv2.putText(court, tid, (lx+4, ly-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(path, court)
    print(f"✅ Trajectórias: {path}")


def save_heatmap(tracks, path, scale=20, sigma=15):
    W, H = int(COURT_W * scale), int(COURT_H * scale)
    heat = np.zeros((H, W), dtype=np.float32)
    for track in tracks.values():
        if track['class'] != 'player':
            continue
        for fr in track['frames']:
            if fr.get('interpolated', False):
                continue   # não conta posições inventadas no heatmap
            rx, ry = fr['real_coords']
            if 0 <= rx <= COURT_W and 0 <= ry <= COURT_H:
                heat[int(ry * scale), int(rx * scale)] += 1

    heat = cv2.GaussianBlur(heat, (sigma*2+1, sigma*2+1), sigma)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    court, _, _ = draw_court(scale)
    blended = cv2.addWeighted(court, 0.35, heat_color, 0.65, 0)
    cv2.imwrite(path, blended)
    print(f"✅ Heatmap: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', default='output/trajectories_2d.json')
    parser.add_argument('--output-dir',   default='output')
    parser.add_argument('--scale', type=int, default=20)
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    n_players = sum(1 for t in tracks.values() if t['class'] == 'player')
    print(f"📊 {len(tracks)} tracks ({n_players} jogadores)")

    os.makedirs(args.output_dir, exist_ok=True)
    save_trajectories(tracks, f"{args.output_dir}/field_trajectories.png", args.scale)
    save_heatmap(tracks,      f"{args.output_dir}/field_heatmap.png",      args.scale)
    print(f"\n✅ Visualizações em {args.output_dir}/")


if __name__ == '__main__':
    main()
