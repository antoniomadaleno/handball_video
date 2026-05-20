#!/usr/bin/env python3
"""Calcula matriz de homografia a partir dos pontos selecionados."""
import cv2
import numpy as np
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', default='output/homography_points.json')
    parser.add_argument('--output', default='output/homography.json')
    args = parser.parse_args()

    with open(args.points) as f:
        data = json.load(f)

    img_pts  = np.array(data['image_points'],      dtype=np.float32)
    real_pts = np.array(data['real_world_points'],  dtype=np.float32)

    print(f"\n🔧 A calcular homografia com {len(img_pts)} pontos...")

    H, mask = cv2.findHomography(img_pts, real_pts, cv2.RANSAC, 3.0)

    if H is None:
        print("❌ Falhou o cálculo! Tenta adicionar mais pontos.")
        return

    inliers = int(mask.sum())
    print(f"✅ Homografia calculada!  Inliers: {inliers}/{len(img_pts)}")

    print("\n📊 Teste de reprojeção:")
    errors = []
    for i, (ip, rp) in enumerate(zip(img_pts, real_pts)):
        pt = np.array([[[ip[0], ip[1]]]], dtype=np.float32)
        t  = cv2.perspectiveTransform(pt, H)[0][0]
        err = float(np.linalg.norm(t - rp))
        errors.append(err)
        status = "✅" if err < 0.5 else "⚠️ "
        print(f"  {status} {i+1}. ({t[0]:.2f}, {t[1]:.2f})m   erro={err:.3f}m")

    print(f"\n  Erro médio: {np.mean(errors):.3f}m  |  máximo: {np.max(errors):.3f}m")
    if np.mean(errors) > 1.0:
        print("  ⚠️  Erro elevado — verifica se clicaste nos pontos certos")

    result = {
        'homography_matrix': H.tolist(),
        'n_points': len(img_pts),
        'inliers':  inliers,
        'mean_error_m': float(np.mean(errors)),
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Homografia guardada: {args.output}")


if __name__ == '__main__':
    main()
