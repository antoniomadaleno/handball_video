#!/usr/bin/env python3
"""
Pós-processamento das trajectórias:
  1. Funde tracks partidos que são provavelmente o mesmo jogador
  2. Interpola gaps dentro de cada track

Corre APÓS pipeline.py e ANTES da homografia.

Uso:
    python src/interpolate_tracks.py
    python src/interpolate_tracks.py --max-gap 60 --max-dist 150

Parâmetros chave:
    --max-gap   frames máximos de interrupção para fundir dois tracks (default: 60)
    --max-dist  distância máxima em pixels para fundir tracks (default: 120)
    --fill-gap  gap máximo para interpolação intra-track (default: 30)
"""
import json
import numpy as np
import argparse


# ── interpolação ──────────────────────────────────────────────────────────────

def _velocity(frames, n=5):
    """Velocidade média (px/frame) calculada a partir dos últimos n frames."""
    recent = sorted(frames, key=lambda f: f['frame'])[-n:]
    if len(recent) < 2:
        return np.array([0.0, 0.0])
    dx = recent[-1]['center'][0] - recent[0]['center'][0]
    dy = recent[-1]['center'][1] - recent[0]['center'][1]
    dt = recent[-1]['frame']     - recent[0]['frame']
    return np.array([dx / dt, dy / dt]) if dt else np.array([0.0, 0.0])


def interp_frames(f_before, f_after, frames_before=None, frames_after=None):
    """
    Gera frames interpolados entre dois frames com Hermite cúbica.
    Usa a velocidade nos extremos para uma curva mais realista.
    frames_before/after: listas de frames para calcular velocidade.
    """
    n_gap = f_after['frame'] - f_before['frame'] - 1
    if n_gap <= 0:
        return []

    # Velocidade nos extremos (px/frame)
    v0 = _velocity(frames_before or [f_before])
    v1 = _velocity(frames_after  or [f_after])

    p0 = np.array(f_before['center'])
    p1 = np.array(f_after['center'])

    # Escalar velocidade para o comprimento do gap (Hermite usa derivadas normalizadas)
    T = n_gap + 1.0
    m0 = v0 * T
    m1 = v1 * T

    bw = ((f_before['bbox'][2] - f_before['bbox'][0]) +
          (f_after['bbox'][2]  - f_after['bbox'][0])) / 2
    bh = ((f_before['bbox'][3] - f_before['bbox'][1]) +
          (f_after['bbox'][3]  - f_after['bbox'][1])) / 2

    result = []
    for i in range(1, n_gap + 1):
        t  = i / T
        # Polinómios de Hermite
        h00 =  2*t**3 - 3*t**2 + 1
        h10 =    t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =    t**3 -   t**2
        pt = h00*p0 + h10*m0 + h01*p1 + h11*m1
        cx, cy = float(pt[0]), float(pt[1])
        result.append({
            'frame':        f_before['frame'] + i,
            'bbox':         [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2],
            'center':       [cx, cy],
            'confidence':   0.0,
            'interpolated': True,
        })
    return result


def fill_intra_gaps(track, max_gap=75):
    """Preenche gaps dentro de um track com interpolação Hermite."""
    frames = sorted(track['frames'], key=lambda f: f['frame'])
    out = [frames[0]]
    for i in range(1, len(frames)):
        gap = frames[i]['frame'] - frames[i-1]['frame'] - 1
        if 0 < gap <= max_gap:
            ctx_before = frames[max(0, i-5):i]
            ctx_after  = frames[i:min(len(frames), i+5)]
            out.extend(interp_frames(frames[i-1], frames[i], ctx_before, ctx_after))
        out.append(frames[i])
    return out


# ── fusão de tracks ───────────────────────────────────────────────────────────

def _track_summary(frames):
    frs = sorted(frames, key=lambda f: f['frame'])
    return {
        'start':     frs[0]['frame'],
        'end':       frs[-1]['frame'],
        'start_pos': frs[0]['center'],
        'end_pos':   frs[-1]['center'],
    }


def merge_pass(tracks, max_gap, max_dist):
    """Uma passagem de fusão greedy. Retorna tracks fundidos e nº de fusões."""
    info = {tid: _track_summary(t['frames']) for tid, t in tracks.items()}
    by_start = sorted(tracks.keys(), key=lambda tid: info[tid]['start'])

    absorbed = set()   # tracks que foram absorvidos por outro

    for i, tid_a in enumerate(by_start):
        if tid_a in absorbed:
            continue
        end_a   = info[tid_a]['end']
        pos_a   = info[tid_a]['end_pos']
        cls_a   = tracks[tid_a]['class']

        best, best_dist = None, float('inf')

        for tid_b in by_start[i+1:]:
            if tid_b in absorbed or tracks[tid_b]['class'] != cls_a:
                continue
            gap  = info[tid_b]['start'] - end_a
            if gap < 0 or gap > max_gap:
                continue
            dist = float(np.linalg.norm(np.array(pos_a) - np.array(info[tid_b]['start_pos'])))
            if dist < max_dist and dist < best_dist:
                best, best_dist = tid_b, dist

        if best is not None:
            # fundir best em tid_a: interpolar gap e juntar frames
            sorted_a = sorted(tracks[tid_a]['frames'], key=lambda f: f['frame'])
            sorted_b = sorted(tracks[best]['frames'],  key=lambda f: f['frame'])
            last_a, first_b = sorted_a[-1], sorted_b[0]
            gap_frames = interp_frames(last_a, first_b, sorted_a[-5:], sorted_b[:5])

            tracks[tid_a]['frames'].extend(gap_frames)
            tracks[tid_a]['frames'].extend(tracks[best]['frames'])
            tracks[tid_a]['frames'].sort(key=lambda f: f['frame'])

            # actualizar info para possíveis fusões subsequentes
            info[tid_a] = _track_summary(tracks[tid_a]['frames'])
            absorbed.add(best)

    new_tracks = {tid: t for tid, t in tracks.items() if tid not in absorbed}
    return new_tracks, len(absorbed)


def merge_tracks(tracks, max_gap=60, max_dist=120):
    """Itera fusões até não haver mais para fazer."""
    total = 0
    for _ in range(10):   # máximo 10 passagens (evita loop infinito)
        tracks, n = merge_pass(tracks, max_gap, max_dist)
        total += n
        if n == 0:
            break
    return tracks, total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fusão e interpolação de tracks')
    parser.add_argument('--input',    default='output/trajectories_video_teste_1.json')
    parser.add_argument('--output',   default=None,
                        help='Ficheiro de saída (default: sobrescreve o input)')
    parser.add_argument('--max-gap',  type=int, default=60,
                        help='Gap máximo (frames) para fundir dois tracks (default: 60)')
    parser.add_argument('--max-dist', type=int, default=120,
                        help='Distância máxima (px) para fundir tracks (default: 120)')
    parser.add_argument('--fill-gap', type=int, default=75,
                        help='Gap máximo para interpolação intra-track (default: 30)')
    args = parser.parse_args()

    output = args.output or args.input   # sobrescreve por defeito

    with open(args.input) as f:
        tracks = json.load(f)

    n_before  = len(tracks)
    n_players = sum(1 for t in tracks.values() if t['class'] == 'player')
    n_frames_before = sum(len(t['frames']) for t in tracks.values())

    print(f"\n🔧 INTERPOLAÇÃO E FUSÃO DE TRACKS")
    print(f"   Tracks antes:  {n_before} ({n_players} jogadores)")
    print(f"   max-gap={args.max_gap}f  max-dist={args.max_dist}px  fill-gap={args.fill_gap}f")

    # 1. Fundir tracks partidos
    tracks, n_merged = merge_tracks(tracks, max_gap=args.max_gap, max_dist=args.max_dist)
    print(f"\n   Fusões feitas:  {n_merged}")
    print(f"   Tracks depois: {len(tracks)}")

    # 2. Interpolação intra-track
    n_interp = 0
    for t in tracks.values():
        before = len(t['frames'])
        t['frames'] = fill_intra_gaps(t, max_gap=args.fill_gap)
        n_interp += len(t['frames']) - before

    n_frames_after = sum(len(t['frames']) for t in tracks.values())
    print(f"   Frames interpolados: {n_interp}")
    print(f"   Total frames: {n_frames_before} → {n_frames_after}")

    with open(output, 'w') as f:
        json.dump(tracks, f, indent=2)

    print(f"\n✅ Guardado: {output}")
    print(f"\n💡 Próximo passo — actualizar coordenadas 2D:")
    print(f"   python homography/pipeline.py --from 3")


if __name__ == '__main__':
    main()
