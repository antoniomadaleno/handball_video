#!/usr/bin/env python3
"""
Reconhece números de dorsal usando EasyOCR.

Requer:  pip install easyocr

Para cada track:
  1. Amostra frames com alta confiança
  2. Recorta zona do dorsal (frente e costas do jersey)
  3. Pré-processa para melhorar legibilidade
  4. Corre OCR e filtra resultados numéricos (1-99)
  5. Vota no número mais frequente

Adiciona 'dorsal' a cada track no JSON.

Uso:
    python src/recognize_numbers.py
    python src/recognize_numbers.py --samples 15   # mais amostras = mais lento mas melhor
"""
import cv2
import numpy as np
import json
import argparse
import re
from collections import Counter

VIDEO    = 'videos/video_teste_1.mp4'
TRAJ_IN  = 'output/trajectories_video_teste_1.json'
TRAJ_OUT = 'output/trajectories_video_teste_1.json'
SAMPLES  = 10


def preprocess_jersey(crop):
    """
    Pré-processa recorte do jersey para melhorar OCR:
    - Escala para pelo menos 80px de altura
    - Aumenta contraste
    - Converte para cinza
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return None

    # Escala: mínimo 80px de altura
    if h < 80:
        scale = 80 / h
        crop = cv2.resize(crop, (int(w*scale), 80), interpolation=cv2.INTER_CUBIC)

    # Cinza + contraste adaptativo
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Threshold para isolar dígitos brancos/pretos no jersey
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def jersey_number_crop(frame, bbox):
    """Recorta zona central do jersey onde costuma estar o número."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    w = x2 - x1
    # Dorsal normalmente: 25%-60% da altura, 15%-85% da largura
    top    = y1 + int(h * 0.25)
    bottom = y1 + int(h * 0.60)
    left   = x1 + int(w * 0.15)
    right  = x1 + int(w * 0.85)
    top, bottom = max(0, top), min(frame.shape[0], bottom)
    left, right = max(0, left), min(frame.shape[1], right)
    crop = frame[top:bottom, left:right]
    return crop if crop.size > 0 else None


def is_valid_number(text):
    """Verifica se o texto é um número de dorsal válido (1-99)."""
    text = text.strip()
    if re.fullmatch(r'\d{1,2}', text):
        n = int(text)
        if 1 <= n <= 99:
            return n
    return None


def read_track_number(frames_data, cap, reader, n_samples=SAMPLES):
    """
    Lê o número de dorsal de um track por votação.
    Devolve (número, confiança) ou (None, 0).
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Ordena por confiança e selecciona amostras espaçadas
    candidates = sorted(frames_data, key=lambda f: -f.get('confidence', 0))
    step = max(1, len(candidates) // n_samples)
    sampled = candidates[::step][:n_samples]

    votes = Counter()

    for fd in sampled:
        fn = fd['frame']
        if fn >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        crop = jersey_number_crop(frame, fd['bbox'])
        processed = preprocess_jersey(crop)
        if processed is None:
            continue

        # EasyOCR espera imagem BGR ou RGB
        img_for_ocr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        results = reader.readtext(img_for_ocr, allowlist='0123456789',
                                  detail=1, paragraph=False)

        for (_, text, conf) in results:
            n = is_valid_number(text)
            if n is not None and conf > 0.4:
                votes[n] += 1

    if not votes:
        return None, 0.0

    best_num, best_count = votes.most_common(1)[0]
    confidence = best_count / len(sampled)
    return best_num, confidence


def main():
    parser = argparse.ArgumentParser(description='Reconhecimento de dorsais')
    parser.add_argument('--trajectories', default=TRAJ_IN)
    parser.add_argument('--output',       default=TRAJ_OUT)
    parser.add_argument('--video',        default=VIDEO)
    parser.add_argument('--samples', type=int, default=SAMPLES,
                        help='Frames a amostrar por track (default: 10)')
    parser.add_argument('--min-conf', type=float, default=0.2,
                        help='Confiança mínima para aceitar um número (default: 0.2)')
    args = parser.parse_args()

    # Importa EasyOCR só quando necessário
    try:
        import easyocr
    except ImportError:
        print("❌ EasyOCR não instalado. Corre:  pip install easyocr")
        return

    print("\n🔢 RECONHECIMENTO DE DORSAIS")
    print("   A carregar modelo EasyOCR...")
    reader = easyocr.Reader(['en'], verbose=False)
    print("   Modelo carregado.\n")

    with open(args.trajectories) as f:
        tracks = json.load(f)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Não abriu: {args.video}")
        return

    players = {tid: t for tid, t in tracks.items() if t['class'] == 'player'}
    results = {}

    for i, (tid, t) in enumerate(players.items()):
        num, conf = read_track_number(t['frames'], cap, reader, args.samples)
        results[tid] = num
        status = f"#{num} (conf={conf:.0%})" if num else "não detectado"
        print(f"   [{i+1}/{len(players)}] Track {tid:>3}: {status}")
        t['dorsal'] = num

    cap.release()

    # Detectar conflitos (dois tracks com mesmo número = provável mesmo jogador)
    num_tracks = {}
    for tid, num in results.items():
        if num is not None:
            num_tracks.setdefault(num, []).append(tid)

    print(f"\n📊 RESULTADO:")
    for num in sorted(num_tracks):
        tids = num_tracks[num]
        if len(tids) > 1:
            print(f"   #{num:>2}: tracks {tids}  ⚠️  mesmo dorsal — prováveis fragmentos do mesmo jogador")
        else:
            print(f"   #{num:>2}: track {tids[0]}")

    unknown = [tid for tid, num in results.items() if num is None]
    if unknown:
        print(f"   Sem dorsal: tracks {unknown}")

    with open(args.output, 'w') as f:
        json.dump(tracks, f, indent=2)
    print(f"\n✅ Guardado: {args.output}")
    print(f"\n💡 Tracks com o mesmo dorsal podem ser fundidos com:")
    print(f"   python src/interpolate_tracks.py")


if __name__ == '__main__':
    main()
