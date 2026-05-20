#!/usr/bin/env python3
"""
Seletor de pontos para homografia.
Janela: frame do video (esquerda) | campo IHF desenhado (direita)

Clica num ponto no VIDEO → depois no ponto correspondente no CAMPO.
Repete 6-8 vezes. Não precisas de cobrir o campo todo — usa só os
pontos que são visíveis no vídeo (cantos, cruzamentos de linhas, etc.)

Teclas:  S = guardar  |  Z = desfazer último  |  Q = sair sem guardar
"""
import cv2
import numpy as np
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from court import draw_court, COURT_W, COURT_H

PANEL_W  = 700
PANEL_H  = 430
COURT_SCALE = 15          # px/m → campo fica 600×300
COURT_DW = int(COURT_W * COURT_SCALE)   # 600
COURT_DH = int(COURT_H * COURT_SCALE)   # 300

state = dict(
    frame=None, frame_w=0, frame_h=0,
    v_scale=1.0, v_ox=0, v_oy=0,
    court=None, court_sx=1.0, court_sy=1.0, court_oy=0,
    video_pts=[], real_pts=[], pending=None,
)


def letterbox(img, tw, th):
    h, w = img.shape[:2]
    sc = min(tw / w, th / h)
    nw, nh = int(w * sc), int(h * sc)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = cv2.resize(img, (nw, nh))
    return canvas, sc, ox, oy


def render():
    s = state

    # ── painel vídeo ────────────────────────────────────────────────────────
    vdisp, sc, ox, oy = letterbox(s['frame'], PANEL_W, PANEL_H)
    s['v_scale'], s['v_ox'], s['v_oy'] = sc, ox, oy

    for i, vp in enumerate(s['video_pts']):
        dp = (int(vp[0]*sc + ox), int(vp[1]*sc + oy))
        cv2.circle(vdisp, dp, 6, (0, 255, 0), -1)
        cv2.putText(vdisp, str(i+1), (dp[0]+7, dp[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if s['pending'] is not None:
        dp = (int(s['pending'][0]*sc + ox), int(s['pending'][1]*sc + oy))
        cv2.circle(vdisp, dp, 7, (0, 165, 255), -1)
        cv2.putText(vdisp, "?", (dp[0]+7, dp[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    n = len(s['video_pts'])
    if s['pending'] is None:
        msg  = f"Passo 1: clica no VIDEO ({n} pares)"
        col  = (255, 255, 255)
    else:
        msg  = "Passo 2: clica no CAMPO (ponto correspondente)"
        col  = (0, 200, 255)
    cv2.rectangle(vdisp, (0, PANEL_H-32), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(vdisp, msg, (6, PANEL_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    # ── painel campo ────────────────────────────────────────────────────────
    court_oy = (PANEL_H - COURT_DH) // 2
    s['court_oy'] = court_oy
    cpanel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    # centrar o campo horizontalmente também
    court_ox = (PANEL_W - COURT_DW) // 2
    cpanel[court_oy:court_oy+COURT_DH, court_ox:court_ox+COURT_DW] = s['court']
    s['court_ox'] = court_ox

    for i, rp in enumerate(s['real_pts']):
        cx = int(rp[0] * s['court_sx']) + court_ox
        cy = int(rp[1] * s['court_sy']) + court_oy
        cv2.circle(cpanel, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(cpanel, str(i+1), (cx+7, cy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    col2 = (0, 200, 0) if n >= 4 else (0, 165, 255)
    hint = "Usa pontos visiveis no video (nao precisas cobrir o campo todo)"
    cv2.rectangle(cpanel, (0, PANEL_H-32), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.putText(cpanel, f"{n} pares | S=guardar  Z=desfazer  Q=sair",
                (6, PANEL_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col2, 1)
    if n == 0:
        cv2.putText(cpanel, hint, (6, PANEL_H-48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("Homografia — VIDEO  |  CAMPO", np.hstack([vdisp, cpanel]))


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    s = state

    if x < PANEL_W:
        # clique no painel de vídeo
        if s['pending'] is None:
            rx = max(0, min(s['frame_w']-1, (x - s['v_ox']) / s['v_scale']))
            ry = max(0, min(s['frame_h']-1, (y - s['v_oy']) / s['v_scale']))
            s['pending'] = (rx, ry)
            render()
    else:
        # clique no painel de campo
        if s['pending'] is not None:
            cx = x - PANEL_W - s.get('court_ox', 0)
            cy = y - s['court_oy']
            if 0 <= cx < COURT_DW and 0 <= cy < COURT_DH:
                real_x = cx / s['court_sx']
                real_y = cy / s['court_sy']
                s['video_pts'].append(s['pending'])
                s['real_pts'].append((real_x, real_y))
                print(f"  Par {len(s['video_pts'])}: "
                      f"pixel=({s['pending'][0]:.0f},{s['pending'][1]:.0f}) "
                      f"→ ({real_x:.2f}m, {real_y:.2f}m)")
                s['pending'] = None
                render()


def main():
    parser = argparse.ArgumentParser(description='Selecionar pontos de homografia')
    parser.add_argument('--video',  default='videos/video_teste_1.mp4')
    parser.add_argument('--frame',  type=int, default=30,
                        help='Frame do video a usar como referencia')
    parser.add_argument('--output', default='output/homography_points.json')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(args.frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ Não consegui ler frame {frame_idx} de {args.video}")
        return

    court, sx, sy = draw_court(COURT_SCALE)
    state.update(frame=frame, frame_w=frame.shape[1], frame_h=frame.shape[0],
                 court=court, court_sx=sx, court_sy=sy)

    cv2.namedWindow("Homografia — VIDEO  |  CAMPO")
    cv2.setMouseCallback("Homografia — VIDEO  |  CAMPO", on_mouse)

    print("\n📍 SELETOR DE PONTOS DE HOMOGRAFIA")
    print("=" * 50)
    print("1. Clica num ponto identificável no VIDEO")
    print("2. Clica no ponto correspondente no CAMPO")
    print("3. Repete 6-8 vezes")
    print()
    print("💡 Dica: usa só pontos visíveis na câmara.")
    print("   Não precisas de cobrir o campo todo.")
    print("   Bons pontos: cantos de linhas, cruzamentos,")
    print("   pontos de penálti, interseções do círculo.")
    print()
    print("S=guardar | Z=desfazer | Q=sair\n")

    render()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("Saído sem guardar.")
            break
        elif key in (ord('z'), ord('Z')):
            if state['pending'] is not None:
                state['pending'] = None
            elif state['video_pts']:
                state['video_pts'].pop()
                state['real_pts'].pop()
                print(f"  Desfeito. {len(state['video_pts'])} pares.")
            render()
        elif key in (ord('s'), ord('S')):
            if len(state['video_pts']) < 4:
                print(f"❌ Precisas de pelo menos 4 pares! Tens {len(state['video_pts'])}.")
            else:
                out_dir = os.path.dirname(args.output)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                data = {
                    'image_points':      [[float(x), float(y)] for x, y in state['video_pts']],
                    'real_world_points': [[float(x), float(y)] for x, y in state['real_pts']],
                    'n_points': len(state['video_pts']),
                }
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\n✅ {len(state['video_pts'])} pares guardados: {args.output}")
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
