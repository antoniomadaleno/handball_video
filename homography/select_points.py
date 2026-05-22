#!/usr/bin/env python3
"""
Seletor de pontos para homografia.
Janela: frame do video (esquerda) | campo IHF desenhado (direita)

Clica num ponto no VIDEO → depois no ponto correspondente no CAMPO.
Repete 6-8 vezes. Não precisas de cobrir o campo todo — usa só os
pontos que são visíveis no vídeo (cantos, cruzamentos de linhas, etc.)

Botão direito ou Z = desfazer último par
S ou fechar a janela = guardar (se >= 4 pares)
"""
import cv2
import numpy as np
import json
import argparse
import os
import sys

import matplotlib
for _backend in ('Qt5Agg', 'TkAgg', 'WxAgg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()
        break
    except Exception:
        continue
else:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from court import draw_court, COURT_W, COURT_H

COURT_SCALE = 20   # px/m para o desenho do campo


def main():
    parser = argparse.ArgumentParser(description='Selecionar pontos de homografia')
    parser.add_argument('--video',  default='videos/video_teste_1.mp4')
    parser.add_argument('--frame',  type=int, default=30,
                        help='Frame do video a usar como referencia')
    parser.add_argument('--output', default='output/homography_points.json')
    parser.add_argument('--load',   default=None,
                        help='Carregar pontos existentes para adicionar mais (ex: output/homography_points.json)')
    args = parser.parse_args()

    # ── carregar frame do vídeo ───────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(args.frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Não consegui ler frame {frame_idx} de {args.video}")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    court_img, sx, sy = draw_court(COURT_SCALE)
    court_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)

    court_w = int(COURT_W * COURT_SCALE)
    court_h = int(COURT_H * COURT_SCALE)

    # ── estado ────────────────────────────────────────────────────────────────
    video_pts = []   # (px, py) em pixels do frame original
    real_pts  = []   # (rx, ry) em metros
    pending   = [None]   # ponto no vídeo ainda sem correspondência no campo

    # carregar pontos existentes se pedido
    if args.load and os.path.exists(args.load):
        with open(args.load) as f:
            existing = json.load(f)
        video_pts.extend([tuple(p) for p in existing.get('image_points', [])])
        real_pts.extend([tuple(p)  for p in existing.get('real_world_points', [])])
        print(f"   Carregados {len(video_pts)} pontos existentes de {args.load}")

    print("\n📍 SELETOR DE PONTOS DE HOMOGRAFIA")
    print("=" * 50)
    print("1. Clica num ponto identificável no VIDEO (esquerda)")
    print("2. Clica no ponto correspondente no CAMPO (direita)")
    print("3. Repete 6-8 vezes")
    print("\nBotão direito = desfazer último  |  fecha a janela = guardar\n")

    fig, (ax_vid, ax_court) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Clica VIDEO → CAMPO  (botão direito = desfazer)", fontsize=11)
    fig.subplots_adjust(bottom=0.1)

    from matplotlib.widgets import Button
    ax_btn = fig.add_axes([0.42, 0.01, 0.16, 0.05])
    btn_save = Button(ax_btn, '💾 Guardar e fechar', color='#2ecc71', hovercolor='#27ae60')

    def redraw():
        ax_vid.clear()
        ax_vid.imshow(frame_rgb)
        ax_vid.set_title(f"VIDEO — frame {frame_idx}", fontsize=9)
        ax_vid.axis('off')

        ax_court.clear()
        ax_court.imshow(court_rgb)
        ax_court.set_title("CAMPO IHF (metros)", fontsize=9)
        ax_court.set_xlim(0, court_w)
        ax_court.set_ylim(court_h, 0)
        ax_court.axis('off')

        for i, (vp, rp) in enumerate(zip(video_pts, real_pts)):
            ax_vid.plot(*vp, 'o', color='lime', markersize=10, markeredgecolor='black')
            ax_vid.text(vp[0]+8, vp[1]-8, str(i+1), color='lime', fontsize=9,
                        fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
            rx_px = rp[0] * sx
            ry_px = rp[1] * sy
            ax_court.plot(rx_px, ry_px, 'o', color='lime', markersize=10,
                          markeredgecolor='black')
            ax_court.text(rx_px+6, ry_px-6, str(i+1), color='lime', fontsize=9,
                          fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

        if pending[0] is not None:
            ax_vid.plot(*pending[0], '*', color='orange', markersize=14,
                        markeredgecolor='black')
            ax_vid.set_title(f"VIDEO — agora clica no CAMPO", fontsize=9, color='orange')

        n = len(video_pts)
        status = f"{n} pares" + (" ✅ pronto para guardar" if n >= 4 else " (mínimo 4)")
        fig.suptitle(f"Clica VIDEO → CAMPO  |  {status}  |  botão direito = desfazer",
                     fontsize=10)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button == 3:   # botão direito = desfazer
            if pending[0] is not None:
                pending[0] = None
            elif video_pts:
                video_pts.pop(); real_pts.pop()
                print(f"   Desfeito. {len(video_pts)} pares.")
            redraw()
            return

        if event.button != 1 or event.inaxes is None:
            return

        if event.inaxes == ax_vid:
            if pending[0] is None:
                pending[0] = (event.xdata, event.ydata)
                print(f"   Ponto vídeo: ({event.xdata:.0f}, {event.ydata:.0f})px "
                      f"— agora clica no CAMPO")
                redraw()

        elif event.inaxes == ax_court:
            if pending[0] is not None:
                # converter px do campo → metros
                rx = event.xdata / sx
                ry = event.ydata / sy
                if 0 <= rx <= COURT_W and 0 <= ry <= COURT_H:
                    video_pts.append(pending[0])
                    real_pts.append((rx, ry))
                    print(f"   Par {len(video_pts)}: "
                          f"pixel=({pending[0][0]:.0f},{pending[0][1]:.0f}) "
                          f"→ ({rx:.2f}m, {ry:.2f}m)")
                    pending[0] = None
                    redraw()

    def on_close(event):
        save()

    def save():
        if len(video_pts) < 4:
            print(f"❌ Precisas de pelo menos 4 pares! Tens {len(video_pts)}.")
            return
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        data = {
            'image_points':      [[float(x), float(y)] for x, y in video_pts],
            'real_world_points': [[float(x), float(y)] for x, y in real_pts],
            'n_points':          len(video_pts),
            'frame_idx':         frame_idx,
        }
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ {len(video_pts)} pares guardados: {args.output}")

    def on_save_btn(event):
        save()
        plt.close(fig)

    btn_save.on_clicked(on_save_btn)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('close_event', on_close)
    redraw()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
