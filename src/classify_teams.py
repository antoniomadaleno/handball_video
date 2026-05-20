#!/usr/bin/env python3
"""
Classifica jogadores por cor do equipamento.

Para cada track:
  1. Amostra N frames do vídeo
  2. Recorta a zona do jersey (20%-65% da altura do bbox)
  3. Calcula o hue dominante em HSV
  4. Agrupa os tracks em equipas por K-means no hue

Adiciona 'jersey_hue' e 'team' a cada track no JSON.

Uso:
    python src/classify_teams.py
    python src/classify_teams.py --teams 3    # Team A, Team B, Árbitros
    python src/classify_teams.py --show       # mostra amostras de cor
"""
import cv2
import numpy as np
import json
import argparse
import os
from collections import Counter

VIDEO       = 'videos/video_teste_1.mp4'
TRAJ_IN     = 'output/trajectories_video_teste_1.json'
TRAJ_OUT    = 'output/trajectories_video_teste_1.json'
SAMPLES     = 8    # frames a amostrar por track
TEAM_LABELS = ['A', 'B', 'REF']


# ── extracção de cor ──────────────────────────────────────────────────────────

def dominant_hue(img_bgr):
    """
    Hue dominante (0-180) da zona de jersey em HSV.
    Exclui pixels muito escuros (preto) e muito claros (branco).
    Devolve -1 se não houver pixels válidos.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    mask = (s > 40) & (v > 40) & (v < 230)   # exclui preto, branco e cinza
    valid_hues = h[mask]
    if len(valid_hues) < 20:
        return -1
    # histograma de hue com bins de 5 graus
    hist, _ = np.histogram(valid_hues, bins=36, range=(0, 180))
    return int(np.argmax(hist)) * 5


def jersey_crop(frame, bbox):
    """Recorta a zona do jersey: 20%-65% da altura do bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    top    = y1 + int(h * 0.20)
    bottom = y1 + int(h * 0.65)
    top    = max(0, top)
    bottom = min(frame.shape[0], bottom)
    crop   = frame[top:bottom, x1:x2]
    return crop if crop.size > 0 else None


def track_jersey_hue(frames_data, cap, n_samples=SAMPLES):
    """
    Devolve hue médio de um track, amostrado de n_samples frames.
    frames_data: lista de dicts com 'frame' e 'bbox'.
    cap: cv2.VideoCapture já aberto.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # selecciona frames com alta confiança e espaçados
    candidates = sorted(frames_data, key=lambda f: -f.get('confidence', 0))
    step = max(1, len(candidates) // n_samples)
    sampled = candidates[::step][:n_samples]

    hues = []
    for fd in sampled:
        fn = fd['frame']
        if fn >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        crop = jersey_crop(frame, fd['bbox'])
        if crop is None:
            continue
        h = dominant_hue(crop)
        if h >= 0:
            hues.append(h)

    return float(np.median(hues)) if hues else -1.0


# ── clustering ────────────────────────────────────────────────────────────────

def hue_to_vec(hue_deg):
    """Converte hue circular (0-180°) em vector 2D para K-means correcto."""
    rad = np.radians(hue_deg * 2)   # *2 porque hue vai de 0-180
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)


def cluster_teams(hue_dict, n_teams):
    """
    Agrupa track IDs em n_teams grupos por K-means no hue.
    hue_dict: {tid: hue_value}  (ignora hue == -1)
    Devolve {tid: team_label}
    """
    valid = {tid: h for tid, h in hue_dict.items() if h >= 0}
    if not valid:
        return {tid: 'A' for tid in hue_dict}

    tids   = list(valid.keys())
    vecs   = np.array([hue_to_vec(h) for h in valid.values()])

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k        = min(n_teams, len(tids))
    _, labels, centers = cv2.kmeans(
        vecs, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Ordena centros por hue para labels consistentes
    center_hues = []
    for c in centers:
        angle = np.degrees(np.arctan2(c[1], c[0]))
        center_hues.append((angle % 360) / 2)   # volta a 0-180

    order = np.argsort(center_hues)
    remap = {old: TEAM_LABELS[new] for new, old in enumerate(order)}

    result = {}
    for tid, lbl in zip(tids, labels.flatten()):
        result[tid] = remap[int(lbl)]
    # tracks sem hue válido ficam como desconhecidos
    for tid in hue_dict:
        if tid not in result:
            result[tid] = '?'
    return result


# ── visualização de diagnóstico ───────────────────────────────────────────────

def show_team_samples(tracks, team_assignments, cap, n=3):
    """Mostra amostras de jersey por equipa usando matplotlib."""
    import random
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    teams = sorted(set(team_assignments.values()))
    fig, axes = plt.subplots(len(teams), n, figsize=(n*2, len(teams)*2.5))
    if len(teams) == 1:
        axes = [axes]
    # garantir sempre 2D
    axes = [ax if hasattr(ax, '__len__') else [ax] for ax in axes]

    for row_i, team in enumerate(teams):
        tids = [tid for tid, t in team_assignments.items() if t == team]
        sample = random.sample(tids, min(n, len(tids)))
        for col_i in range(n):
            ax = axes[row_i][col_i]
            ax.axis('off')
            if col_i >= len(sample):
                continue
            tid = sample[col_i]
            frs = tracks[tid]['frames']
            fd  = max(frs, key=lambda f: f.get('confidence', 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fd['frame'])
            ret, frame = cap.read()
            if not ret:
                continue
            crop = jersey_crop(frame, fd['bbox'])
            if crop is not None and crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
            hue = tracks[tid].get('jersey_hue', -1)
            ax.set_title(f"T{team} id={tid}\nhue={hue:.0f}°", fontsize=7)

    plt.suptitle("Amostras de jersey por equipa", fontsize=10)
    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Classificação de equipas por cor')
    parser.add_argument('--trajectories', default=TRAJ_IN)
    parser.add_argument('--output',       default=TRAJ_OUT)
    parser.add_argument('--video',        default=VIDEO)
    parser.add_argument('--teams',  type=int, default=3,
                        help='Número de grupos de cor (default: 3 = A, B, REF)')
    parser.add_argument('--show', action='store_true',
                        help='Mostrar amostras de jersey por equipa')
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Não abriu: {args.video}")
        return

    players = {tid: t for tid, t in tracks.items() if t['class'] == 'player'}
    print(f"\n🎨 CLASSIFICAÇÃO DE EQUIPAS POR COR")
    print(f"   {len(players)} jogadores  |  {args.teams} grupos")

    # 1. Calcular hue por track
    hue_dict = {}
    for i, (tid, t) in enumerate(players.items()):
        hue = track_jersey_hue(t['frames'], cap)
        hue_dict[tid] = hue
        hue_label = f"{hue:.0f}°" if hue >= 0 else "N/A"
        print(f"   [{i+1}/{len(players)}] Track {tid:>3}: hue={hue_label}")

    # 2. Clustering
    team_assignments = cluster_teams(hue_dict, args.teams)

    # 3. Resultado
    print(f"\n📊 RESULTADO:")
    for team in sorted(set(team_assignments.values())):
        members = [tid for tid, t in team_assignments.items() if t == team]
        hues    = [f"{hue_dict[tid]:.0f}°" for tid in members if hue_dict.get(tid, -1) >= 0]
        print(f"   Equipa {team}: tracks {members}  hues={hues}")

    # 4. Guardar no JSON
    for tid, t in tracks.items():
        t['jersey_hue'] = hue_dict.get(tid, -1)
        t['team']       = team_assignments.get(tid, '?')

    with open(args.output, 'w') as f:
        json.dump(tracks, f, indent=2)
    print(f"\n✅ Guardado: {args.output}")

    if args.show:
        show_team_samples(tracks, team_assignments, cap)

    cap.release()


if __name__ == '__main__':
    main()
