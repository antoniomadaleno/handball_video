#!/usr/bin/env python3
"""
Pipeline de homografia.
Corre os 4 passos em sequência: pontos → homografia → transformar → visualizar

Uso:
    python homography/pipeline.py                  # corre tudo
    python homography/pipeline.py --from 2         # começa no passo 2
    python homography/pipeline.py --only 3 4       # só esses passos

Passos:
    1 - Seleccionar pontos  [interactivo]
    2 - Calcular homografia
    3 - Transformar trajectórias → metros
    4 - Visualizações 2D do campo
"""
import subprocess
import sys
import os
import argparse

# ── Configuração ──────────────────────────────────────────────────────────────
VIDEO             = 'videos/video_teste_1.mp4'
HOMOGRAPHY_FRAME  = 30
TRAJECTORIES      = 'output/trajectories_video_teste_1.json'
HOMOGRAPHY_POINTS = 'output/homography_points.json'
HOMOGRAPHY_MATRIX = 'output/homography.json'
TRAJECTORIES_2D   = 'output/trajectories_2d.json'
OUTPUT_DIR        = 'output'
# ─────────────────────────────────────────────────────────────────────────────

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    1: "Seleccionar pontos de homografia  [interactivo]",
    2: "Calcular homografia",
    3: "Transformar trajectórias → metros",
    4: "Visualizações 2D do campo",
}


def header(n, title):
    print(f"\n{'='*55}")
    print(f"  PASSO {n}/{len(STEPS)}: {title}")
    print(f"{'='*55}")


def run(cmd, step_name):
    result = subprocess.run([PY] + cmd)
    if result.returncode != 0:
        print(f"\n❌ Passo falhou: {step_name}")
        print("   Usa --from N para recomeçar a partir deste passo.")
        sys.exit(result.returncode)


def passo_1():
    header(1, STEPS[1])
    print(f"\n  Clica 6-8 pares no seletor (VIDEO → CAMPO).")
    print(f"  S = guardar  |  Z = desfazer  |  Q = sair\n")
    run([
        os.path.join(HERE, 'select_points.py'),
        '--video',  VIDEO,
        '--frame',  str(HOMOGRAPHY_FRAME),
        '--output', HOMOGRAPHY_POINTS,
    ], STEPS[1])


def passo_2():
    header(2, STEPS[2])
    if not os.path.exists(HOMOGRAPHY_POINTS):
        print(f"❌ {HOMOGRAPHY_POINTS} não encontrado — corre primeiro o passo 1.")
        sys.exit(1)
    run([
        os.path.join(HERE, 'compute_homography.py'),
        '--points', HOMOGRAPHY_POINTS,
        '--output', HOMOGRAPHY_MATRIX,
    ], STEPS[2])


def passo_3():
    header(3, STEPS[3])
    for f in [TRAJECTORIES, HOMOGRAPHY_MATRIX]:
        if not os.path.exists(f):
            print(f"❌ {f} não encontrado.")
            sys.exit(1)
    run([
        os.path.join(HERE, 'transform.py'),
        '--trajectories', TRAJECTORIES,
        '--homography',   HOMOGRAPHY_MATRIX,
        '--output',       TRAJECTORIES_2D,
    ], STEPS[3])


def passo_4():
    header(4, STEPS[4])
    if not os.path.exists(TRAJECTORIES_2D):
        print(f"❌ {TRAJECTORIES_2D} não encontrado — corre primeiro o passo 3.")
        sys.exit(1)
    run([
        os.path.join(HERE, 'visualize_2d.py'),
        '--trajectories', TRAJECTORIES_2D,
        '--output-dir',   OUTPUT_DIR,
    ], STEPS[4])


PASSOS = {1: passo_1, 2: passo_2, 3: passo_3, 4: passo_4}


def main():
    parser = argparse.ArgumentParser(description='Pipeline de homografia')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_step', type=int, metavar='N',
                       help='Começa a partir do passo N')
    group.add_argument('--only', dest='only', type=int, nargs='+', metavar='N',
                       help='Corre apenas os passos indicados')
    args = parser.parse_args()

    if args.only:
        steps_to_run = sorted(args.only)
    elif args.from_step:
        steps_to_run = list(range(args.from_step, len(PASSOS) + 1))
    else:
        steps_to_run = list(range(1, len(PASSOS) + 1))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📐 HOMOGRAFIA PIPELINE")
    print(f"   Video:        {VIDEO}")
    print(f"   Trajectórias: {TRAJECTORIES}")
    print(f"   Passos:       {steps_to_run}")

    for n in steps_to_run:
        if n not in PASSOS:
            print(f"❌ Passo {n} não existe (1-{len(PASSOS)})")
            sys.exit(1)
        PASSOS[n]()

    print(f"\n{'='*55}")
    print("  ✅ HOMOGRAFIA COMPLETA")
    print(f"{'='*55}")
    print(f"\n  Outputs em {OUTPUT_DIR}/:")
    print(f"    - field_trajectories.png")
    print(f"    - field_heatmap.png")
    print()


if __name__ == '__main__':
    main()
