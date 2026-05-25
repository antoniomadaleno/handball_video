#!/usr/bin/env python3
"""
Correcção automática das coordenadas reais (pós-homografia).

Estratégia: ao longo do vídeo, os jogadores cobrem aproximadamente
toda a largura do campo (eixo Y, 0-20m) e a parte visível em comprimento
(eixo X, 0-40m). Usamos os percentis das trajectórias para calibrar
linearmente os eixos para os valores reais do campo.

Uso:
    python homography/correct_coords.py
    python homography/correct_coords.py --inset 0.8  (margem maior das linhas)
    python homography/correct_coords.py --x-range 20 40  (limita correção X)
"""
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', default='output/trajectories_2d.json')
    parser.add_argument('--output',       default='output/trajectories_2d.json')
    parser.add_argument('--inset',        type=float, default=0.5,
                        help='Margem em metros das linhas laterais (default: 0.5m)')
    parser.add_argument('--correct-x',    action='store_true',
                        help='Aplicar também correcção ao eixo X')
    parser.add_argument('--x-range',      type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Gama X visível (ex: 20 40 para meio campo direito)')
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    # ── recolher todas as coordenadas ──────────────────────────────────────────
    all_x, all_y = [], []
    for track in tracks.values():
        if track['class'] != 'player':
            continue
        for fr in track['frames']:
            rc = fr.get('real_coords')
            if rc:
                all_x.append(rc[0])
                all_y.append(rc[1])

    if not all_x:
        print("❌ Sem trajectórias de jogadores para analisar.")
        return

    # ── correcção Y ────────────────────────────────────────────────────────────
    # Usamos percentis (2% / 98%) para robustez face a outliers
    y_min = float(np.percentile(all_y, 2))
    y_max = float(np.percentile(all_y, 98))

    target_y_min = args.inset
    target_y_max = 20.0 - args.inset

    if abs(y_max - y_min) < 0.1:
        print("❌ Extensão Y demasiado pequena — não dá para calibrar.")
        return

    ay = (target_y_max - target_y_min) / (y_max - y_min)
    by = target_y_min - ay * y_min

    print(f"📐 CORRECÇÃO AUTOMÁTICA")
    print(f"{'='*50}")
    print(f"\n   Eixo Y (largura do campo):")
    print(f"   ├─ Detectado: [{y_min:6.2f}, {y_max:6.2f}]m")
    print(f"   ├─ Alvo:      [{target_y_min:6.2f}, {target_y_max:6.2f}]m")
    print(f"   └─ Fórmula:   y' = {ay:.4f}·y + ({by:+.4f})")

    # ── correcção X (opcional) ─────────────────────────────────────────────────
    if args.correct_x or args.x_range:
        x_min = float(np.percentile(all_x, 2))
        x_max = float(np.percentile(all_x, 98))

        if args.x_range:
            target_x_min, target_x_max = args.x_range
            target_x_min += args.inset
            target_x_max -= args.inset
        else:
            target_x_min = args.inset
            target_x_max = 40.0 - args.inset

        if abs(x_max - x_min) > 0.1:
            ax = (target_x_max - target_x_min) / (x_max - x_min)
            bx = target_x_min - ax * x_min
            print(f"\n   Eixo X (comprimento do campo):")
            print(f"   ├─ Detectado: [{x_min:6.2f}, {x_max:6.2f}]m")
            print(f"   ├─ Alvo:      [{target_x_min:6.2f}, {target_x_max:6.2f}]m")
            print(f"   └─ Fórmula:   x' = {ax:.4f}·x + ({bx:+.4f})")
        else:
            ax, bx = 1.0, 0.0
    else:
        ax, bx = 1.0, 0.0

    # ── aplicar correcções ────────────────────────────────────────────────────
    n_corrected = 0
    for track in tracks.values():
        for fr in track['frames']:
            rc = fr.get('real_coords')
            if rc:
                rx, ry = rc
                rx_new = ax * rx + bx
                ry_new = ay * ry + by
                # clamp aos limites do campo
                rx_new = max(0.0, min(40.0, rx_new))
                ry_new = max(0.0, min(20.0, ry_new))
                fr['real_coords'] = [float(rx_new), float(ry_new)]
                n_corrected += 1

    with open(args.output, 'w') as f:
        json.dump(tracks, f, indent=2)

    print(f"\n✅ {n_corrected} pontos corrigidos.")
    print(f"💾 Guardado: {args.output}")
    print(f"\n   Corre agora:  python homography/validate.py")


if __name__ == '__main__':
    main()
