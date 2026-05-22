#!/usr/bin/env python3
"""
Transforma trajectórias de pixels para coordenadas reais (metros).

Por frame: estima H via SIFT feature matching contra o frame de referência.
Pós-processamento: filtro de Kalman por track para suavizar ruído do zoom.
"""
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


# ── homografia por frame ───────────────────────────────────────────────────────

def load_reference_frame(video_path, homography_points_path):
    with open(homography_points_path) as f:
        pts_data = json.load(f)
    ref_idx = pts_data.get('frame_idx', 30)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Não consegui ler frame {ref_idx} de {video_path}")
    return frame, ref_idx


def build_sift_reference(ref_frame_bgr):
    sift = cv2.SIFT_create(nfeatures=2000)
    gray = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    return sift, matcher, kp, des


def estimate_H(frame_bgr, sift, matcher, ref_kp, ref_des, H_ref):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)

    if des2 is None or len(kp2) < 10:
        return None

    matches = matcher.knnMatch(ref_des, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return None

    pts_ref   = np.float32([ref_kp[m.queryIdx].pt for m in good])
    pts_frame = np.float32([kp2[m.trainIdx].pt    for m in good])

    H_f2r, mask = cv2.findHomography(pts_frame, pts_ref, cv2.RANSAC, 5.0)

    if H_f2r is None or mask.ravel().sum() < 8:
        return None

    return H_ref @ H_f2r


# ── suavização temporal das matrizes H ────────────────────────────────────────

def smooth_homographies(H_per_frame, sigma=5):
    """
    Suaviza os 9 parâmetros de cada H com um filtro gaussiano temporal.
    Elimina saltos bruscos causados por zoom/pan da câmara.
    sigma: desvio padrão em frames (maior = mais suave).
    """
    frame_indices = sorted(H_per_frame.keys())
    H_stack = np.array([H_per_frame[fi] for fi in frame_indices], dtype=np.float64)

    H_smooth = np.zeros_like(H_stack)
    for i in range(3):
        for j in range(3):
            H_smooth[:, i, j] = gaussian_filter1d(H_stack[:, i, j], sigma=sigma)

    return {fi: H_smooth[k] for k, fi in enumerate(frame_indices)}


# ── filtro de Kalman ───────────────────────────────────────────────────────────

def smooth_track_kalman(frames, process_noise=0.05, measurement_noise=1.5, gate=2.0):
    """
    Suaviza as coordenadas reais de um track com filtro de Kalman.

    Estado: [x, y, vx, vy]  —  modelo de velocidade constante.
    Medição: [x, y].

    process_noise:     confiança no modelo cinemático (menor = mais suave)
    measurement_noise: confiança nas medições da homografia (maior = mais suave)
    gate:              distância máxima em metros entre previsão e medição.
                       Medições além deste limite (ex: artefactos de zoom) são
                       rejeitadas e o filtro usa só a previsão cinemática.
    """
    if not frames:
        return frames

    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
    kf.errorCovPost        = np.eye(4, dtype=np.float32)

    x0, y0 = frames[0]['real_coords']
    kf.statePost = np.array([[x0], [y0], [0.], [0.]], dtype=np.float32)

    smoothed = []
    prev_fn = frames[0]['frame']

    for fr in frames:
        fn = fr['frame']
        steps = max(1, fn - prev_fn)
        for _ in range(steps):
            predicted = kf.predict()

        rc = fr.get('real_coords')
        if rc and -2 <= rc[0] <= 42 and -2 <= rc[1] <= 22:
            meas = np.array([[rc[0]], [rc[1]]], dtype=np.float32)

            # Innovation gate: rejeita medições muito distantes da previsão
            # (artefactos de zoom causam saltos de vários metros)
            pred_xy = predicted[:2].flatten()
            innovation = np.linalg.norm(meas.flatten() - pred_xy)
            if innovation <= gate:
                kf.correct(meas)
            # se fora do gate: mantém a previsão, não actualiza

        state = kf.statePost
        smoothed.append({
            **fr,
            'real_coords': [float(state[0]), float(state[1])],
        })
        prev_fn = fn

    return smoothed


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories',      default='output/trajectories_video_teste_1.json')
    parser.add_argument('--homography',        default='output/homography.json')
    parser.add_argument('--homography-points', default='output/homography_points.json')
    parser.add_argument('--video',             default='videos/video_teste_1.mp4')
    parser.add_argument('--output',            default='output/trajectories_2d.json')
    parser.add_argument('--process-noise',     type=float, default=0.05,
                        help='Ruído do processo Kalman — menor = mais suave (default: 0.05)')
    parser.add_argument('--measurement-noise', type=float, default=1.5,
                        help='Ruído da medição Kalman — maior = mais suave (default: 1.5)')
    parser.add_argument('--gate',              type=float, default=2.0,
                        help='Distância máxima em metros para aceitar medição (default: 2.0)')
    parser.add_argument('--h-sigma',           type=float, default=5.0,
                        help='Suavização gaussiana das matrizes H em frames (default: 5)')
    args = parser.parse_args()

    with open(args.trajectories) as f:
        tracks = json.load(f)

    with open(args.homography) as f:
        H_ref = np.array(json.load(f)['homography_matrix'], dtype=np.float64)

    print(f"🔄 A transformar {len(tracks)} tracks...")

    # ── frame de referência e SIFT ────────────────────────────────────────────
    print("   A carregar frame de referência e calcular SIFT...")
    ref_frame, ref_idx = load_reference_frame(args.video, args.homography_points)
    sift, matcher, ref_kp, ref_des = build_sift_reference(ref_frame)
    print(f"   Frame de referência: {ref_idx}  |  {len(ref_kp)} keypoints SIFT")

    # ── estimar H por frame ───────────────────────────────────────────────────
    frame_indices = sorted({
        fr['frame']
        for t in tracks.values()
        for fr in t['frames']
    })
    print(f"   A estimar H para {len(frame_indices)} frames...")

    cap = cv2.VideoCapture(args.video)
    H_per_frame = {}
    H_last = H_ref
    n_ok = n_fallback = 0

    for fi in tqdm(frame_indices, desc="H por frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            H_per_frame[fi] = H_last
            n_fallback += 1
            continue

        H = estimate_H(frame, sift, matcher, ref_kp, ref_des, H_ref)
        if H is not None:
            H_per_frame[fi] = H
            H_last = H
            n_ok += 1
        else:
            H_per_frame[fi] = H_last
            n_fallback += 1

    cap.release()
    print(f"   SIFT OK: {n_ok}  |  Fallback: {n_fallback}")

    # ── suavizar H ao longo do tempo ──────────────────────────────────────────
    print(f"   A suavizar matrizes H (sigma={args.h_sigma})...")
    H_per_frame = smooth_homographies(H_per_frame, sigma=args.h_sigma)

    # ── aplicar transformação ─────────────────────────────────────────────────
    result = {}
    total = out_of_bounds = 0

    for track_id, track in tracks.items():
        new_frames = []
        for fr in track['frames']:
            H = H_per_frame.get(fr['frame'], H_ref)

            if track['class'] == 'player':
                bb = fr['bbox']
                cx = (bb[0] + bb[2]) / 2
                cy = bb[3]
            else:
                cx, cy = fr['center']

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            rx, ry = cv2.perspectiveTransform(pt, H)[0][0].tolist()
            total += 1
            if not (-2 <= rx <= 42 and -2 <= ry <= 22):
                out_of_bounds += 1
            new_frames.append({**fr, 'real_coords': [float(rx), float(ry)]})

        # ── filtro de Kalman ──────────────────────────────────────────────────
        new_frames = smooth_track_kalman(
            new_frames,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            gate=args.gate,
        )
        result[track_id] = {**track, 'frames': new_frames}

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    pct = out_of_bounds / total * 100 if total else 0
    print(f"✅ Transformação completa!")
    print(f"   Pontos fora dos limites: {out_of_bounds}/{total} ({pct:.1f}%)")
    if pct > 10:
        print("   ⚠️  Muitos pontos fora — verifica a homografia")
    print(f"💾 Guardado: {args.output}")


if __name__ == '__main__':
    main()
