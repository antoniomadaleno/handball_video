#!/usr/bin/env python3
"""
Valida a homografia seguindo um jogador específico.

Modo 1 — selecção interactiva (recomendado):
    python homography/validate.py
    → Abre frame do vídeo no matplotlib, clicas no jogador, segue-o.

Modo 2 — por track ID:
    python homography/validate.py --track 5

O seguimento usa nearest-neighbour com filtro de equipa/dorsal se disponível.
Se o jogador não for detectado por mais de --max-gap frames, pára de tentar
reconectar (evita ir buscar o jogador errado).
"""
import cv2
import numpy as np
import json
import argparse
import os
import sys

import matplotlib
matplotlib.use('TkAgg')          # backend com janela própria, não precisa GTK
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from court import draw_court, COURT_W, COURT_H


# ── índice de detecções por frame ─────────────────────────────────────────────

def build_frame_index(tracks):
    index = {}
    for tid, track in tracks.items():
        team   = track.get('team',   None)
        dorsal = track.get('dorsal', None)
        for f in track['frames']:
            fn = f['frame']
            index.setdefault(fn, []).append({
                'cx':          f['center'][0],
                'cy':          f['center'][1],
                'track_id':    tid,
                'real_x':      f['real_coords'][0],
                'real_y':      f['real_coords'][1],
                'bbox':        f['bbox'],
                'interpolated': f.get('interpolated', False),
                'team':        team,
                'dorsal':      dorsal,
            })
    return index


# ── selecção interactiva via matplotlib ───────────────────────────────────────

def pick_player(video_path, frame_index, ref_frame=30):
    """
    Mostra frame do vídeo com jogadores marcados.
    Utilizador clica num — devolve (cx, cy, frame_num).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref   = min(ref_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print("❌ Não consegui ler o frame.")
        return None, None, None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fh, fw    = frame_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame_rgb)
    ax.set_title("Clica no jogador que queres seguir  (fecha a janela para cancelar)",
                 fontsize=11)
    ax.axis('off')

    # Marca todos os jogadores detectados
    for det in frame_index.get(ref, []):
        ax.plot(det['cx'], det['cy'], 'o',
                markerfacecolor='none', markeredgecolor='yellow',
                markeredgewidth=2, markersize=14)
        label = str(det['track_id'])
        if det.get('dorsal'):
            label += f" #{det['dorsal']}"
        elif det.get('team'):
            label += f" T{det['team']}"
        ax.text(det['cx'] + 8, det['cy'] - 8, label,
                color='yellow', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    clicked = {}

    def on_click(event):
        if event.inaxes == ax and event.button == 1:
            clicked['x'] = event.xdata
            clicked['y'] = event.ydata
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    if 'x' not in clicked:
        print("Cancelado.")
        return None, None, None

    cx_click, cy_click = clicked['x'], clicked['y']

    # Encontrar detecção mais próxima
    best, best_dist = None, float('inf')
    for det in frame_index.get(ref, []):
        d = np.linalg.norm([det['cx'] - cx_click, det['cy'] - cy_click])
        if d < best_dist:
            best_dist, best = d, det

    if best is None or best_dist > 200:
        print("❌ Nenhum jogador detectado perto do clique.")
        return None, None, None

    info = f"track={best['track_id']}"
    if best.get('dorsal'):
        info += f"  dorsal=#{best['dorsal']}"
    if best.get('team'):
        info += f"  equipa={best['team']}"
    print(f"✅ Seleccionado: {info}  pos=({best['cx']:.0f},{best['cy']:.0f})px")
    return best['cx'], best['cy'], ref


# ── seguidor por posição ──────────────────────────────────────────────────────

def greedy_follow(frame_index, start_frame, start_px, start_py,
                  max_dist=150, decel=0.88,
                  target_team=None, target_dorsal=None,
                  max_gap=20):
    """
    Segue um jogador frame a frame.
    - Filtra por equipa/dorsal se disponível.
    - Se não detectar por mais de max_gap frames consecutivos, marca como
      PERDIDO e para de tentar reconectar (evita ir buscar jogador errado).
    """
    result = []
    pos = np.array([start_px, start_py], dtype=float)
    vel = np.array([0.0, 0.0])
    frames_lost = 0

    for fn in sorted(f for f in frame_index if f >= start_frame):
        if frames_lost > max_gap:
            result.append({
                'frame': fn, 'center': pos.tolist(),
                'real_coords': None, 'bbox': None,
                'track_id': None, 'interpolated': True, 'lost': True,
            })
            continue

        predicted = pos + vel
        best, best_dist = None, float('inf')

        for det in frame_index.get(fn, []):
            if target_dorsal is not None and det['dorsal'] is not None:
                if det['dorsal'] != target_dorsal:
                    continue
            elif target_team is not None and det['team'] is not None:
                if det['team'] != target_team:
                    continue
            d = np.linalg.norm([det['cx'] - predicted[0], det['cy'] - predicted[1]])
            if d < best_dist:
                best_dist, best = d, det

        if best is not None and best_dist < max_dist:
            new_pos = np.array([best['cx'], best['cy']])
            vel = (new_pos - pos) * 0.7 + vel * 0.3
            pos = new_pos
            frames_lost = 0
            result.append({
                'frame':        fn,
                'center':       [best['cx'], best['cy']],
                'real_coords':  [best['real_x'], best['real_y']],
                'bbox':         best['bbox'],
                'track_id':     best['track_id'],
                'interpolated': best['interpolated'],
                'lost':         False,
                'team':         best['team'],
                'dorsal':       best['dorsal'],
            })
        else:
            vel *= decel
            pos = pos + vel
            frames_lost += 1
            result.append({
                'frame':        fn,
                'center':       pos.tolist(),
                'real_coords':  None,
                'bbox':         None,
                'track_id':     None,
                'interpolated': True,
                'lost':         frames_lost > max_gap,
            })

    return result


# ── render do vídeo ───────────────────────────────────────────────────────────

def render_validation(follow_result, video_path, output_path, scale=20):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_h  = 400
    panel_w  = int(fw * panel_h / fh)
    court_w  = int(COURT_W * scale)
    court_h  = int(COURT_H * scale)
    pad      = 20
    cpanel_w = court_w + pad * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (panel_w + cpanel_w, panel_h))

    court_base, _, _ = draw_court(scale)
    court_ox = pad
    court_oy = (panel_h - court_h) // 2
    result_by_frame = {r['frame']: r for r in follow_result}
    TRAIL = 50

    print("   A gerar vídeo...")
    for fn in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        vdisp = cv2.resize(frame, (panel_w, panel_h))
        sx, sy = panel_w / fw, panel_h / fh
        r = result_by_frame.get(fn)

        # ── vídeo ────────────────────────────────────────────────────────────
        if r:
            if not r.get('lost') and not r['interpolated'] and r['bbox']:
                bb = r['bbox']
                x1, y1 = int(bb[0]*sx), int(bb[1]*sy)
                x2, y2 = int(bb[2]*sx), int(bb[3]*sy)
                cv2.rectangle(vdisp, (x1, y1), (x2, y2), (0, 255, 255), 3)
                lbl = f"ID {r['track_id']}"
                if r.get('dorsal'):
                    lbl += f" #{r['dorsal']}"
                cv2.putText(vdisp, lbl, (x1, max(y1-8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            elif r.get('lost'):
                cv2.putText(vdisp, "PERDIDO", (panel_w//2 - 60, panel_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                # interpolado mas dentro do gap — ponto laranja
                cx = int(r['center'][0] * sx)
                cy = int(r['center'][1] * sy)
                cv2.circle(vdisp, (cx, cy), 10, (0, 100, 255), 2)

        cv2.rectangle(vdisp, (0, 0), (155, 26), (0, 0, 0), -1)
        cv2.putText(vdisp, f"Frame {fn}/{total-1}",
                    (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── campo 2D ─────────────────────────────────────────────────────────
        cpanel = np.zeros((panel_h, cpanel_w, 3), dtype=np.uint8)
        court  = court_base.copy()

        trail = [x for x in follow_result
                 if fn - TRAIL <= x['frame'] < fn
                 and not x.get('lost') and x['real_coords']]
        for j in range(1, len(trail)):
            alpha = j / len(trail)
            col = (0, int(160*alpha), int(200*alpha))
            p1 = (int(trail[j-1]['real_coords'][0]*scale),
                  int(trail[j-1]['real_coords'][1]*scale))
            p2 = (int(trail[j]['real_coords'][0]*scale),
                  int(trail[j]['real_coords'][1]*scale))
            if (0<=p1[0]<court_w and 0<=p1[1]<court_h and
                    0<=p2[0]<court_w and 0<=p2[1]<court_h):
                cv2.line(court, p1, p2, col, 2)

        if r and not r.get('lost') and r['real_coords']:
            rx, ry = r['real_coords']
            px, py = int(rx*scale), int(ry*scale)
            if 0 <= px < court_w and 0 <= py < court_h:
                color = (0, 255, 255) if not r['interpolated'] else (0, 130, 255)
                cv2.circle(court, (px, py), 8, color, -1)
                cv2.circle(court, (px, py), 10, (255, 255, 255), 2)
                cv2.putText(court, f"({rx:.1f},{ry:.1f})m",
                            (px+12, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
        elif r and r.get('lost'):
            cv2.putText(court, "PERDIDO", (court_w//2-30, court_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cpanel[court_oy:court_oy+court_h, court_ox:court_ox+court_w] = court

        tid_str = r['track_id'] if r and r.get('track_id') else '—'
        cv2.putText(cpanel, f"ID: {tid_str}",
                    (court_ox, court_oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(np.hstack([vdisp, cpanel]))
        if fn % 100 == 0:
            print(f"   {fn}/{total} ({fn*100//total}%)")

    cap.release()
    writer.release()
    print(f"\n✅ Validação guardada: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', default='output/trajectories_2d.json')
    parser.add_argument('--video',        default='videos/video_teste_1.mp4')
    parser.add_argument('--track',        type=str, default=None)
    parser.add_argument('--frame',        type=int, default=30)
    parser.add_argument('--max-dist',     type=int, default=150)
    parser.add_argument('--max-gap',      type=int, default=20,
                        help='Frames sem detecção antes de marcar como PERDIDO (default: 20)')
    parser.add_argument('--output',       default=None)
    parser.add_argument('--scale',        type=int, default=20)
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    frame_index = build_frame_index(tracks)

    # ── ponto de partida ─────────────────────────────────────────────────────
    if args.track is not None:
        t = tracks.get(args.track)
        if not t:
            print(f"❌ Track {args.track} não encontrado.")
            return
        first      = sorted(t['frames'], key=lambda f: f['frame'])[0]
        start_px   = first['center'][0]
        start_py   = first['center'][1]
        start_frame = first['frame']
        print(f"Ponto de partida: track {args.track}  frame {start_frame}")
    else:
        start_px, start_py, start_frame = pick_player(
            args.video, frame_index, ref_frame=args.frame)
        if start_px is None:
            return

    # ── equipa/dorsal do ponto de partida ─────────────────────────────────────
    start_det = min(
        (d for d in frame_index.get(start_frame, [])
         if np.linalg.norm([d['cx']-start_px, d['cy']-start_py]) < 80),
        key=lambda d: np.linalg.norm([d['cx']-start_px, d['cy']-start_py]),
        default=None)

    target_team   = start_det['team']   if start_det else None
    target_dorsal = start_det['dorsal'] if start_det else None

    if target_dorsal:
        print(f"   Dorsal #{target_dorsal} — seguimento por número")
    elif target_team:
        print(f"   Equipa {target_team} — seguimento filtrado por cor")

    # ── seguir ────────────────────────────────────────────────────────────────
    print(f"\n🎯 A seguir a partir do frame {start_frame}...")
    follow_result = greedy_follow(
        frame_index, start_frame, start_px, start_py,
        max_dist=args.max_dist,
        target_team=target_team,
        target_dorsal=target_dorsal,
        max_gap=args.max_gap)

    det  = sum(1 for r in follow_result if not r.get('lost') and not r['interpolated'])
    lost = sum(1 for r in follow_result if r.get('lost'))
    print(f"   Detectado: {det} frames  |  Perdido: {lost} frames")

    label  = args.track or "click"
    output = args.output or f"output/validation_{label}.mp4"
    os.makedirs('output', exist_ok=True)
    render_validation(follow_result, args.video, output, args.scale)


if __name__ == '__main__':
    main()
