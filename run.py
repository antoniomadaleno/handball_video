#!/usr/bin/env python3
"""
Pipeline de análise de andebol.

Uso:
    python run.py                  # corre tudo do início
    python run.py --from 2         # começa no passo 2
    python run.py --only 1 2       # corre só os passos indicados

Passos:
    1 - Tracking SAM2             → output/trajectories_<nome_video>.json
    2 - Transformar homografia    → output/trajectories_2d.json
    3 - Validar jogador           [interactivo — clica no jogador]

Opcional (correr manualmente se a calibração for muito enviesada):
    python homography/correct_coords.py

Nota: a calibração da homografia (select_points + compute_homography)
é um passo manual único, não incluído aqui. Corre só quando mudas de vídeo:
    python homography/select_points.py
    python homography/compute_homography.py
"""
import subprocess
import sys
import os
import argparse

# ── Configuração ──────────────────────────────────────────────────────────────
VIDEO             = 'videos/video_teste_anjinho.mp4'
HOMOGRAPHY_MATRIX = 'output/homography.json'
TRAJECTORIES_2D   = 'output/trajectories_2d.json'
VALIDATE_FRAME    = 30

# Deriva automaticamente do nome do vídeo
_vname      = os.path.splitext(os.path.basename(VIDEO))[0]
TRAJECTORIES = f'output/trajectories_{_vname}.json'
# ─────────────────────────────────────────────────────────────────────────────

PY = sys.executable

STEPS = {
    1: "Tracking SAM2 (detecção + trajectórias)",
    2: "Transformar trajectórias → metros",
    3: "Validar jogador  [interactivo]",
}


def header(n, title):
    print(f"\n{'='*55}")
    print(f"  PASSO {n}/{len(STEPS)}: {title}")
    print(f"{'='*55}")


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
            print(f"   Corre os passos anteriores primeiro.")
            sys.exit(1)


def passo_1():
    header(1, STEPS[1])
    run(['src/pipeline_sam2.py',
         '--video',  VIDEO,
         '--output', TRAJECTORIES], STEPS[1])


def passo_2():
    header(2, STEPS[2])
    need(TRAJECTORIES, HOMOGRAPHY_MATRIX)
    run(['homography/transform.py',
         '--trajectories',      TRAJECTORIES,
         '--homography',        HOMOGRAPHY_MATRIX,
         '--homography-points', 'output/homography_points.json',
         '--video',             VIDEO,
         '--output',            TRAJECTORIES_2D], STEPS[2])


def passo_3():
    header(3, STEPS[3])
    need(TRAJECTORIES_2D)
    print(f"\n  Abre janela — clica no jogador que queres validar.\n")
    run(['homography/validate.py',
         '--video',        VIDEO,
         '--trajectories', TRAJECTORIES_2D,
         '--frame',        str(VALIDATE_FRAME)], STEPS[3])


PASSOS = {1: passo_1, 2: passo_2, 3: passo_3}


def main():
    global VALIDATE_FRAME
    parser = argparse.ArgumentParser(description='Pipeline de análise de andebol')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--from', dest='from_step', type=int, metavar='N',
                       help='Começa a partir do passo N')
    group.add_argument('--only', dest='only', type=int, nargs='+', metavar='N',
                       help='Corre apenas os passos indicados')
    parser.add_argument('--frame', type=int, default=VALIDATE_FRAME,
                        help=f'Frame para o seletor interactivo no passo 4 (default: {VALIDATE_FRAME})')
    args = parser.parse_args()

    VALIDATE_FRAME = args.frame

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

    print(f"\n{'='*55}")
    print("  ✅ PIPELINE COMPLETO")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
