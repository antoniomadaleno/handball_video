#!/usr/bin/env python3
"""
Valida a homografia para um jogador específico.

Abre uma janela com o frame do vídeo. Clicas num jogador.
Gera um vídeo lado a lado: vídeo original | campo 2D com posição real.
Quando o tracker perde o jogador, não mostra nada (sem previsões inventadas).

Uso:
    python homography/validate.py                  # clica no jogador
    python homography/validate.py --track 12       # usa track ID directamente
    python homography/validate.py --frame 50       # usa frame 50 para selecção
"""
import cv2
import numpy as np
import json
import argparse
import os
import sys

import matplotlib
for _backend in ('Qt5Agg', 'TkAgg', 'WxAgg', 'Agg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()
        break
    except Exception:
        continue

sys.path.insert(0, os.path.dirname(__file__))
from court import draw_court, COURT_W, COURT_H


# ── listar tracks ─────────────────────────────────────────────────────────────

def list_tracks(tracks):
    players = [(tid, t) for tid, t in tracks.items() if t['class'] == 'player']
    players.sort(key=lambda x: -len(x[1]['frames']))
    print(f"\n{'ID':>4}  {'Frames':>7}  {'Início':>7}  {'Fim':>7}")
    print("─" * 32)
    for tid, t in players:
        frs = t['frames']
        print(f"{tid:>4}  {len(frs):>7}  {frs[0]['frame']:>7}  {frs[-1]['frame']:>7}")
    print()


# ── selecção interactiva ──────────────────────────────────────────────────────

def suggest_frames(tracks, top=10):
    """Mostra os frames com mais jogadores detectados para o utilizador escolher."""
    counts = {}
    for t in tracks.values():
        if t['class'] != 'player':
            continue
        for f in t['frames']:
            if not f.get('interpolated', False):
                counts[f['frame']] = counts.get(f['frame'], 0) + 1
    best = sorted(counts.items(), key=lambda x: -x[1])[:top]
    print(f"\n   Frames com mais jogadores detectados:")
    for fn, n in best:
        print(f"     --frame {fn:>4}   ({n} jogadores)")
    print(f"\n   Corre:  python homography/validate.py --frame <número>\n")


def pick_player(video_path, tracks, ref_frame=30):
    """
    Mostra frame do vídeo com todos os jogadores detectados.
    O utilizador clica num — devolve o track_id mais próximo.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref   = min(ref_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print("❌ Não consegui ler o frame.")
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Recolher posições de todos os jogadores neste frame
    det_positions = {}   # track_id → (cx, cy)
    for tid, t in tracks.items():
        if t['class'] != 'player':
            continue
        for f in t['frames']:
            if f['frame'] == ref and not f.get('interpolated', False):
                det_positions[tid] = f['center']
                break

    print(f"   Frame {ref}: {len(det_positions)} jogadores visíveis")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame_rgb)
    ax.set_title(f"Clica no jogador que queres seguir  (frame {ref})", fontsize=11)
    ax.axis('off')

    for tid, (cx, cy) in det_positions.items():
        ax.plot(cx, cy, 'o', markerfacecolor='none',
                markeredgecolor='yellow', markeredgewidth=2, markersize=14)
        ax.text(cx + 8, cy - 8, str(tid), color='yellow', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.55, pad=1, edgecolor='none'))

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
        return None

    cx_c, cy_c = clicked['x'], clicked['y']
    best_tid, best_dist = None, float('inf')
    for tid, (cx, cy) in det_positions.items():
        d = np.linalg.norm([cx - cx_c, cy - cy_c])
        if d < best_dist:
            best_dist, best_tid = d, tid

    if best_tid is None or best_dist > 200:
        print("❌ Nenhum jogador perto do clique.")
        return None

    print(f"✅ Seleccionado: track {best_tid}  "
          f"({det_positions[best_tid][0]:.0f}, {det_positions[best_tid][1]:.0f})px")
    return best_tid


# ── geração do vídeo de validação ─────────────────────────────────────────────

def render_validation(track_id, track_data, video_path, output_path, scale=20):
    # Índice rápido: frame_num → frame_data
    frame_index = {f['frame']: f for f in track_data['frames']}

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

    TRAIL = 60   # frames de rastro

    print(f"   A gerar vídeo ({total} frames)...")

    for fn in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        vdisp = cv2.resize(frame, (panel_w, panel_h))
        sx, sy = panel_w / fw, panel_h / fh

        fd = frame_index.get(fn)   # detecção neste frame (ou None)

        # ── painel vídeo ─────────────────────────────────────────────────────
        if fd and not fd.get('interpolated', False):
            bb = fd['bbox']
            x1, y1 = int(bb[0]*sx), int(bb[1]*sy)
            x2, y2 = int(bb[2]*sx), int(bb[3]*sy)
            cv2.rectangle(vdisp, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(vdisp, f"ID {track_id}",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.rectangle(vdisp, (0, 0), (155, 26), (0, 0, 0), -1)
        cv2.putText(vdisp, f"Frame {fn}/{total-1}",
                    (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── painel campo 2D ──────────────────────────────────────────────────
        cpanel = np.zeros((panel_h, cpanel_w, 3), dtype=np.uint8)
        court  = court_base.copy()

        # Rastro: últimos TRAIL frames detectados (não interpolados)
        trail = [f for f in track_data['frames']
                 if fn - TRAIL <= f['frame'] < fn
                 and not f.get('interpolated', False)
                 and f['real_coords']
                 and -1 <= f['real_coords'][0] <= 41
                 and -1 <= f['real_coords'][1] <= 21]

        for j in range(1, len(trail)):
            alpha = j / len(trail)
            col = (0, int(160*alpha), int(200*alpha))
            p1 = (int(trail[j-1]['real_coords'][0]*scale),
                  int(trail[j-1]['real_coords'][1]*scale))
            p2 = (int(trail[j]['real_coords'][0]*scale),
                  int(trail[j]['real_coords'][1]*scale))
            if (0 <= p1[0] < court_w and 0 <= p1[1] < court_h and
                    0 <= p2[0] < court_w and 0 <= p2[1] < court_h):
                cv2.line(court, p1, p2, col, 2)

        # Posição actual (só se detectado, sem interpolação)
        if fd and not fd.get('interpolated', False) and fd['real_coords']:
            rx, ry = fd['real_coords']
            if 0 <= rx <= COURT_W and 0 <= ry <= COURT_H:
                px, py = int(rx*scale), int(ry*scale)
                cv2.circle(court, (px, py), 8, (0, 255, 255), -1)
                cv2.circle(court, (px, py), 10, (255, 255, 255), 2)
                cv2.putText(court, f"({rx:.1f},{ry:.1f})m",
                            (px+12, py+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cpanel[court_oy:court_oy+court_h, court_ox:court_ox+court_w] = court
        cv2.putText(cpanel, f"Track {track_id}",
                    (court_ox, court_oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(np.hstack([vdisp, cpanel]))

        if fn % 100 == 0:
            print(f"   {fn}/{total} ({fn*100//total}%)")

    cap.release()
    writer.release()
    print(f"\n✅ Guardado: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Validação da homografia por jogador')
    parser.add_argument('frame', nargs='?', type=int, default=None,
                        help='Frame para o seletor interactivo (ex: 120)')
    parser.add_argument('--trajectories', default='output/trajectories_2d.json')
    parser.add_argument('--video',  default='videos/video_teste_anjinho.mp4')
    parser.add_argument('--track',  type=str, default=None,
                        help='ID do track (omitir = selecção por clique)')
    parser.add_argument('--frame',  dest='frame_flag', type=int, default=None,
                        help='Frame para o seletor interactivo (alternativa ao posicional)')
    parser.add_argument('--output', default=None)
    parser.add_argument('--scale',  type=int, default=20)
    parser.add_argument('--list',   action='store_true',
                        help='Listar todos os tracks disponíveis')
    args = parser.parse_args()

    # Aceita tanto posicional como --frame
    if args.frame is None and args.frame_flag is not None:
        args.frame = args.frame_flag

    with open(args.trajectories) as f:
        tracks = json.load(f)

    if args.list:
        list_tracks(tracks)
        return

    # Determinar track ID
    if args.track is not None:
        track_id = args.track
    elif args.frame is None:
        # Sem frame especificado — mostrar sugestões e sair
        suggest_frames(tracks)
        return
    else:
        track_id = pick_player(args.video, tracks, ref_frame=args.frame)
        if track_id is None:
            return

    if track_id not in tracks:
        print(f"❌ Track '{track_id}' não encontrado. Usa --list para ver os disponíveis.")
        return

    track_data = tracks[track_id]
    n_frames   = len(track_data['frames'])
    first      = track_data['frames'][0]['frame']
    last       = track_data['frames'][-1]['frame']
    print(f"\n🎯 Track {track_id}: {n_frames} frames  [{first} → {last}]")

    output = args.output or f"output/validation_track{track_id}.mp4"
    os.makedirs('output', exist_ok=True)
    render_validation(track_id, track_data, args.video, output, args.scale)


if __name__ == '__main__':
    main()
