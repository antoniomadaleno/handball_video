"""
Extrai frames candidatos para anotação focada em bola.

Estratégias:
  1. Frames com baixa confiança na bola (0.1-0.35) — casos difíceis, valiosos
  2. Frames com motion alto (frame difference) — situações de jogo
  3. Amostragem regular de frames sem deteção de bola

Pré-preenche os labels com deteções de jogadores de alta confiança,
para só teres de anotar/corrigir a bola.

Output:
  training/annotation_pool/
    ├── images/<video>_<frame>.jpg
    └── labels/<video>_<frame>.txt  (formato YOLO: cls cx cy w h, normalizado)
"""
import cv2
import numpy as np
import os
import sys
import argparse
import glob
import random
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from detector import HandballDetector

VIDEOS_DIR  = 'training/videos'
MODEL_PATH  = 'training/runs/handball/v7_combined/weights/best.pt'
OUTPUT_DIR  = 'training/annotation_pool'

# Estratégias e quantos frames de cada
N_LOW_CONF = 30    # bola com confiança 0.10-0.35 (casos difíceis)
N_MOTION   = 20    # frames com movimento alto
N_NO_BALL  = 15    # bola não detetada (potenciais falsos negativos)

PLAYER_CONF = 0.5  # threshold para pré-preencher jogadores
BALL_LOW    = 0.10
BALL_HIGH   = 0.35

# Filtros para excluir close-ups e replays — só frames de jogo com campo visível
MIN_PLAYERS_VISIBLE   = 5      # ≥ 5 jogadores confiáveis = vista geral
MAX_AVG_PLAYER_RATIO  = 0.30   # bbox médio < 30% da altura do frame
MAX_PLAYER_RATIO      = 0.45   # NENHUM jogador > 45% (apanha comentadores PIP)


def bbox_to_yolo(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def save_frame(frame_bgr, label_str, video_name, frame_idx, images_dir, labels_dir):
    name = f"{video_name}_{frame_idx:06d}"
    cv2.imwrite(os.path.join(images_dir, f"{name}.jpg"), frame_bgr)
    with open(os.path.join(labels_dir, f"{name}.txt"), 'w') as f:
        f.write(label_str)


def is_game_frame(boxes, scores, classes, frame_h):
    """
    Devolve True se o frame parece ser uma vista geral do jogo (não close-up/replay).
    Critério: pelo menos N jogadores confiáveis e bboxes pequenos relativamente.
    """
    player_heights = []
    for b, s, c in zip(boxes, scores, classes):
        if int(c) == 1 and float(s) >= PLAYER_CONF:
            player_heights.append(b[3] - b[1])

    if len(player_heights) < MIN_PLAYERS_VISIBLE:
        return False

    avg_height_ratio = float(np.mean(player_heights)) / frame_h
    if avg_height_ratio > MAX_AVG_PLAYER_RATIO:
        return False

    # rejeita se ALGUM "jogador" for enorme (comentadores, close-ups parciais)
    max_height_ratio = float(np.max(player_heights)) / frame_h
    if max_height_ratio > MAX_PLAYER_RATIO:
        return False

    return True


def build_label(boxes, scores, classes, w, h):
    """Devolve string YOLO com jogadores de alta confiança + bolas (com baixa confiança incluídas)."""
    lines = []
    for b, s, c in zip(boxes, scores, classes):
        c = int(c)
        if c == 1 and s < PLAYER_CONF:
            continue   # jogador com confiança baixa — não pré-preencher
        cx, cy, bw, bh = bbox_to_yolo(b, w, h)
        lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def process_video(video_path, detector, images_dir, labels_dir):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap  = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n📹 {name}  |  {total} frames")

    candidates_low_conf = []   # (frame_idx, ball_max_conf)
    candidates_no_ball  = []   # frame_idx
    candidates_motion   = []   # (frame_idx, motion_score)
    n_skipped_closeup   = 0    # close-ups / replays filtrados

    prev_gray = None

    # 1ª passagem: analisa todos os frames
    for fi in tqdm(range(total), desc=f"   analise"):
        ret, frame = cap.read()
        if not ret:
            break

        # motion score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4))
        if prev_gray is not None:
            diff = cv2.absdiff(gray_small, prev_gray)
            motion_score = float(diff.mean())
        else:
            motion_score = 0
        prev_gray = gray_small

        # detecção (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)

        # filtro: ignorar close-ups e replays
        if not is_game_frame(boxes, scores, classes, h):
            n_skipped_closeup += 1
            prev_gray = gray_small
            continue

        ball_scores = [float(s) for s, c in zip(scores, classes) if int(c) == 0]

        if not ball_scores:
            candidates_no_ball.append(fi)
        else:
            max_ball = max(ball_scores)
            if BALL_LOW <= max_ball <= BALL_HIGH:
                candidates_low_conf.append((fi, max_ball))

        candidates_motion.append((fi, motion_score))

    cap.release()

    # Selecciona top candidatos
    random.seed(42)
    selected = set()

    # Low confidence (top N pela confiança mais baixa = mais difíceis)
    candidates_low_conf.sort(key=lambda x: x[1])
    for fi, _ in candidates_low_conf[:N_LOW_CONF]:
        selected.add(fi)

    # Motion (top N pelo movimento)
    candidates_motion.sort(key=lambda x: -x[1])
    motion_added = 0
    for fi, _ in candidates_motion:
        if fi not in selected:
            selected.add(fi)
            motion_added += 1
            if motion_added >= N_MOTION:
                break

    # No ball (random sample)
    random.shuffle(candidates_no_ball)
    no_ball_added = 0
    for fi in candidates_no_ball:
        if fi not in selected:
            selected.add(fi)
            no_ball_added += 1
            if no_ball_added >= N_NO_BALL:
                break

    print(f"   Filtrados (close-up/replay): {n_skipped_closeup}")
    print(f"   Seleccionados: {len(selected)} frames")
    print(f"     ├─ baixa confiança bola: {min(N_LOW_CONF, len(candidates_low_conf))}")
    print(f"     ├─ alto movimento:       {motion_added}")
    print(f"     └─ sem bola detectada:    {no_ball_added}")

    # 2ª passagem: salva os frames seleccionados com labels pré-preenchidos
    cap = cv2.VideoCapture(video_path)
    selected_sorted = sorted(selected)
    for fi in tqdm(selected_sorted, desc=f"   salvar"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detector.detect(frame_rgb)
        label_str = build_label(boxes, scores, classes, w, h)
        save_frame(frame, label_str, name, fi, images_dir, labels_dir)
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', default=None,
                        help='Caminho para 1 vídeo específico (ex: training/videos/v3_olympic_games.mp4). '
                             'Sem argumento corre todos os vídeos da pasta.')
    parser.add_argument('--videos-dir', default=VIDEOS_DIR,
                        help='Pasta com vídeos (usado se não passares um vídeo específico)')
    parser.add_argument('--model',      default=MODEL_PATH)
    parser.add_argument('--output',     default=OUTPUT_DIR)
    parser.add_argument('--conf',       type=float, default=BALL_LOW,
                        help='Threshold do detector (default: 0.10)')
    args = parser.parse_args()

    images_dir = os.path.join(args.output, 'images')
    labels_dir = os.path.join(args.output, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Determinar lista de vídeos a processar
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Vídeo não encontrado: {args.video}")
            return
        videos = [args.video]
    else:
        videos = sorted(glob.glob(os.path.join(args.videos_dir, '*.mp4')))
        if not videos:
            print(f"❌ Nenhum vídeo em {args.videos_dir}")
            return

    print(f"\n🎬 {len(videos)} vídeo(s) a processar")

    detector = HandballDetector(args.model, conf_threshold=args.conf)

    for v in videos:
        process_video(v, detector, images_dir, labels_dir)

    n = len(glob.glob(os.path.join(images_dir, '*.jpg')))
    print(f"\n✅ Total no pool: {n} frames em {args.output}")
    print(f"\n   Próximo passo:")
    print(f"   python training/annotate.py")


if __name__ == '__main__':
    main()
