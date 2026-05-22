#!/usr/bin/env python3
"""
Calibração da homografia para um novo vídeo.

Uso:
    python run_homography.py                        # usa o vídeo em run.py
    python run_homography.py --video videos/x.mp4  # vídeo específico
    python run_homography.py --frame 60             # frame para selecção (default: 30)

Passos:
    1 - Selecionar pontos  [interactivo — clica video | campo]
    2 - Calcular matriz H
"""
import subprocess
import sys
import os
import argparse

VIDEO  = 'videos/video_teste_anjinho.mp4'
POINTS = 'output/homography_points.json'
OUTPUT = 'output/homography.json'

PY = sys.executable


def run(cmd, step_name):
    result = subprocess.run([PY] + cmd)
    if result.returncode != 0:
        print(f"\n❌ Passo falhou: {step_name}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description='Calibração da homografia')
    parser.add_argument('video', nargs='?', default=VIDEO,
                        help='Caminho para o vídeo (ex: videos/jogo.mp4)')
    parser.add_argument('--frame', type=int, default=30,
                        help='Frame do vídeo para selecção de pontos (default: 30)')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    print("\n📐 CALIBRAÇÃO DA HOMOGRAFIA")
    print(f"   Vídeo: {args.video}")
    print(f"   Frame: {args.frame}\n")

    print("=" * 55)
    print("  PASSO 1/2: Selecionar pontos  [interactivo]")
    print("=" * 55)
    print("\n  Clica num ponto no VIDEO → mesmo ponto no CAMPO.")
    print("  Repete 6-8 vezes. S=guardar  Z=desfazer  Q=sair\n")
    run(['homography/select_points.py',
         '--video',  args.video,
         '--frame',  str(args.frame),
         '--output', POINTS], 'Selecionar pontos')

    print("\n" + "=" * 55)
    print("  PASSO 2/2: Calcular matriz H")
    print("=" * 55)
    run(['homography/compute_homography.py',
         '--points', POINTS,
         '--output', OUTPUT], 'Calcular homografia')

    print("\n" + "=" * 55)
    print("  ✅ CALIBRAÇÃO COMPLETA")
    print("=" * 55)
    print(f"\n  Podes agora correr:  python run.py\n")


if __name__ == '__main__':
    main()
