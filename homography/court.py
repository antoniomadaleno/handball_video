"""
Desenha campo de andebol conforme normas IHF (40m x 20m).
Usado por select_points.py e visualize_2d.py.
"""
import cv2
import numpy as np

COURT_W = 40.0
COURT_H = 20.0


def _dashed_line(img, p1, p2, color, thickness=1, dash=8, gap=5):
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    if length == 0:
        return
    dx, dy = (x2-x1)/length, (y2-y1)/length
    pos = 0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            pt1 = (int(x1 + dx*pos), int(y1 + dy*pos))
            pt2 = (int(x1 + dx*end), int(y1 + dy*end))
            cv2.line(img, pt1, pt2, color, thickness)
        pos = end
        drawing = not drawing


def _dashed_arc(img, center, axes, start_angle, end_angle, color, thickness=1,
                dash_deg=6, gap_deg=4):
    a = start_angle
    drawing = True
    while a < end_angle:
        seg = dash_deg if drawing else gap_deg
        b = min(a + seg, end_angle)
        if drawing:
            cv2.ellipse(img, center, axes, 0, a, b, color, thickness)
        a = b
        drawing = not drawing


def draw_court(scale=20):
    """
    Desenha campo de andebol.
    scale: pixels por metro (mesmo valor para X e Y — campo é 2:1).
    Retorna imagem (H×W×3), scale_x, scale_y.
    """
    W = int(COURT_W * scale)
    H = int(COURT_H * scale)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)

    s = scale

    def p(x, y):
        return (int(round(x * s)), int(round(y * s)))

    W_c  = (255, 255, 255)   # branco — linhas principais
    Dash = (190, 190, 190)   # cinza — linhas tracejadas (9m)
    Cyan = (0,   220, 255)   # ciano — balizas

    # ── Contorno do campo ────────────────────────────────────────────────────
    cv2.rectangle(img, p(0, 0), (W-1, H-1), W_c, 2)

    # ── Linha central ────────────────────────────────────────────────────────
    cv2.line(img, p(20, 0), p(20, 20), W_c, 1)

    # ── Círculo central (r=3m) ───────────────────────────────────────────────
    cv2.circle(img, p(20, 10), int(3*s), W_c, 1)
    cv2.circle(img, p(20, 10), max(2, s//8), W_c, -1)

    # ── Área de 6m (esquerda) ────────────────────────────────────────────────
    # Arco inferior: centro (0, 8.5), r=6m, 270°→360°
    cv2.ellipse(img, p(0, 8.5),  (int(6*s), int(6*s)), 0, 270, 360, W_c, 1)
    # Linha reta: (6, 8.5) → (6, 11.5)
    cv2.line(img, p(6, 8.5), p(6, 11.5), W_c, 1)
    # Arco superior: centro (0, 11.5), r=6m, 0°→90°
    cv2.ellipse(img, p(0, 11.5), (int(6*s), int(6*s)), 0,   0,  90, W_c, 1)

    # ── Área de 6m (direita) ─────────────────────────────────────────────────
    cv2.ellipse(img, p(40, 8.5),  (int(6*s), int(6*s)), 0, 180, 270, W_c, 1)
    cv2.line(img, p(34, 8.5), p(34, 11.5), W_c, 1)
    cv2.ellipse(img, p(40, 11.5), (int(6*s), int(6*s)), 0,  90, 180, W_c, 1)

    # ── Linha de livre de 9m tracejada (esquerda) ────────────────────────────
    _dashed_arc(img, p(0, 8.5),  (int(9*s), int(9*s)), 270, 360, Dash)
    _dashed_line(img, p(9, 8.5), p(9, 11.5), Dash)
    _dashed_arc(img, p(0, 11.5), (int(9*s), int(9*s)),   0,  90, Dash)

    # ── Linha de livre de 9m tracejada (direita) ─────────────────────────────
    _dashed_arc(img, p(40, 8.5),  (int(9*s), int(9*s)), 180, 270, Dash)
    _dashed_line(img, p(31, 8.5), p(31, 11.5), Dash)
    _dashed_arc(img, p(40, 11.5), (int(9*s), int(9*s)),  90, 180, Dash)

    # ── Pontos de penálti (7m) ───────────────────────────────────────────────
    cv2.circle(img, p(7,  10), max(2, s//8), W_c, -1)
    cv2.circle(img, p(33, 10), max(2, s//8), W_c, -1)

    # ── Balizas (postes a y=8.5 e y=11.5) ───────────────────────────────────
    cv2.line(img, p(0,  8.5), p(0,  11.5), Cyan, max(2, s//8))
    cv2.line(img, p(40, 8.5), p(40, 11.5), Cyan, max(2, s//8))

    # ── Arcos dos cantos (r=1m) ───────────────────────────────────────────────
    cv2.ellipse(img, p(0,  0),  (s, s), 0,   0,  90, W_c, 1)
    cv2.ellipse(img, p(40, 0),  (s, s), 0,  90, 180, W_c, 1)
    cv2.ellipse(img, p(40, 20), (s, s), 0, 180, 270, W_c, 1)
    cv2.ellipse(img, p(0,  20), (s, s), 0, 270, 360, W_c, 1)

    # ── Marcas da zona de substituição (±4.45m do centro, 15cm para campo) ───
    for x in (15.55, 24.45):
        cv2.line(img, p(x, 0),    p(x, 0.15),  W_c, 1)
        cv2.line(img, p(x, 20),   p(x, 19.85), W_c, 1)

    return img, float(s), float(s)
