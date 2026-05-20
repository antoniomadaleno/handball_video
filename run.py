#!/usr/bin/env python3
"""
Pipeline completo de análise de andebol.

Uso:
    python run.py                  # corre tudo do início
    python run.py --from 3         # começa no passo 3
    python run.py --only 1 2       # corre só os passos indicados

Passos:
    1 - Tracking                  → output/trajectories_*.json
    2 - Fundir/interpolar tracks  → melhora IDs e preenche gaps
    3 - Visualizar tracks         → output/output_tracks.mp4
    4 - Seleccionar pontos        [interactivo]
    5 - Calcular homografia
    6 - Transformar trajectórias  → metros
    7 - Visualizações 2D do campo
"""
import subprocess
import sys
import os
import argparse

# ── Configuração ──────────────────────────────────────────────────────────────
VIDEO            = 'videos/video_teste_1.mp4'
MODEL            = 'training/runs/handball/v3_final/weights/best.pt'
HOMOGRAPHY_FRAME = 30

TRAJECTORIES      = 'output/trajectories_video_teste_1.json'
OUTPUT_VIDEO      = 'output/output_tracks.mp4'
HOMOGRAPHY_POINTS = 'output/homography_points.json'
HOMOGRAPHY_MATRIX = 'output/homography.json'
TRAJECTORIES_2D   = 'output/trajectories_2d.json'
# ─────────────────────────────────────────────────────────────────────────────

PY = sys.executable

STEPS = {
    1: "Tracking (detecção + trajectórias)",
    2: "Fundir e interpolar tracks",
    3: "Visualizar tracks no vídeo",
    4: "Seleccionar pontos de homografia  [interactivo]",
    5: "Calcular homografia",
    6: "Transformar trajectórias → metros",
    7: "Visualizações 2D do campo",
}


def header(n, title):
    print(f"\n{'='*62}")
    print(f"  PASSO {n}/{len(STEPS)}: {title}")
    print(f"{'='*62}")


def run(cmd, step_name):
    result = subprocess.run([PY] + cmd)
    if result.returncode != 0:
        print(f"\n❌ Passo falhou: {step_name}")
        print("   Corrige o erro e usa --from N para continuar.")
        sys.exit(result.returncode)


def need(*files):
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Ficheiro não encontrado: {f}")
            sys.exit(1)


def passo_1():
    header(1, STEPS[1])
    run(['src/pipeline.py'], STEPS[1])


def passo_2():
    header(2, STEPS[2])
    need(TRAJECTORIES)
    run(['src/interpolate_tracks.py', '--input', TRAJECTORIES], STEPS[2])


def passo_3():
    header(3, STEPS[3])
    need(TRAJECTORIES)
    run(['src/visualize_tracks.py',
         '--trajectories', TRAJECTORIES,
         '--video',        VIDEO,
         '--output',       OUTPUT_VIDEO], STEPS[3])


def passo_4():
    header(4, STEPS[4])
    print(f"\n  Clica 6-8 pares no seletor (VIDEO → CAMPO).")
    print(f"  S = guardar  |  Z = desfazer  |  Q = sair\n")
    run(['homography/select_points.py',
         '--video',  VIDEO,
         '--frame',  str(HOMOGRAPHY_FRAME),
         '--output', HOMOGRAPHY_POINTS], STEPS[4])


def passo_5():
    header(5, STEPS[5])
    need(HOMOGRAPHY_POINTS)
    run(['homography/compute_homography.py',
         '--points', HOMOGRAPHY_POINTS,
         '--output', HOMOGRAPHY_MATRIX], STEPS[5])


def passo_6():
    header(6, STEPS[6])
    need(TRAJECTORIES, HOMOGRAPHY_MATRIX)
    run(['homography/transform.py',
         '--trajectories', TRAJECTORIES,
         '--homography',   HOMOGRAPHY_MATRIX,
         '--output',       TRAJECTORIES_2D], STEPS[6])


def passo_7():
    header(7, STEPS[7])
    need(TRAJECTORIES_2D)
    run(['homography/visualize_2d.py',
         '--trajectories', TRAJECTORIES_2D,
         '--output-dir',   'output'], STEPS[7])


PASSOS = {1: passo_1, 2: passo_2, 3: passo_3,
          4: passo_4, 5: passo_5, 6: passo_6, 7: passo_7}


def main():
    parser = argparse.ArgumentParser(description='Pipeline de análise de andebol')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_step', type=int, metavar='N')
    group.add_argument('--only', dest='only', type=int, nargs='+', metavar='N')
    args = parser.parse_args()

    if args.only:
        steps_to_run = sorted(args.only)
    elif args.from_step:
        steps_to_run = list(range(args.from_step, len(PASSOS) + 1))
    else:
        steps_to_run = list(range(1, len(PASSOS) + 1))

    os.makedirs('output', exist_ok=True)

    print("\n🏐 HANDBALL ANALYSIS PIPELINE")
    print(f"   Video:  {VIDEO}")
    print(f"   Passos: {steps_to_run}")

    for n in steps_to_run:
        if n not in PASSOS:
            print(f"❌ Passo {n} não existe (1-{len(PASSOS)})")
            sys.exit(1)
        PASSOS[n]()

    print(f"\n{'='*62}")
    print("  ✅ PIPELINE COMPLETO")
    print(f"{'='*62}")
    print(f"\n  Outputs em output/:")
    print(f"    - output_tracks.mp4       (vídeo com tracks)")
    print(f"    - field_trajectories.png  (trajectórias 2D)")
    print(f"    - field_heatmap.png       (heatmap 2D)")
    print()


if __name__ == '__main__':
    main()
