"""
Ferramenta de anotação de bboxes para handball.

Carrega frames de training/annotation_pool/ e permite:
  - Desenhar bboxes com o rato (click + drag)
  - Seleccionar classe com B (bola) ou P (jogador)
  - Apagar bbox com botão direito sobre ela
  - Navegar com setas ou botões
  - Guardar com S ou botão

As anotações ficam em training/annotation_pool/labels/ no formato YOLO.
Quando estiveres satisfeito, corre integrate_annotations.py para
mover para o dataset de treino.

Atalhos:
    B          → classe BOLA
    P          → classe JOGADOR
    ←  /  →    → frame anterior/seguinte
    S          → guardar
    U          → desfazer última bbox
    R          → recarregar do disco (descarta alterações)
"""
import cv2
import numpy as np
import os
import sys
import argparse
import glob
import json
from datetime import datetime

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'   # desativa toolbar — interfere com cliques
for _backend in ('Qt5Agg', 'TkAgg', 'WxAgg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close()
        break
    except Exception:
        continue
else:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons


CLASSES = ['ball', 'player']
COLORS  = {'ball': '#e74c3c', 'player': '#2ecc71'}


class Annotator:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        os.makedirs(labels_dir, exist_ok=True)

        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        if not self.image_paths:
            raise RuntimeError(f"Sem imagens em {images_dir}")

        self.progress_file = os.path.join(images_dir, '..', '.annotate_progress.json')
        self.reviewed = self.load_progress()

        # retoma na primeira imagem não revista (ou no final se estiverem todas)
        self.idx = self.find_next_unreviewed(start=0, default=0)
        self.current_class = 0     # ball por defeito (mais importante)
        self.boxes = []            # (cls, x1, y1, x2, y2)
        self.drag_start = None
        self.drag_rect  = None
        self.pan_start  = None
        self.modified   = False

        # ── layout ────────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Handball Annotator")
        self.ax = self.fig.add_axes([0.03, 0.10, 0.78, 0.85])

        # painel direita: classe + atalhos
        ax_radio = self.fig.add_axes([0.83, 0.70, 0.15, 0.20])
        self.radio = RadioButtons(ax_radio, CLASSES, active=0)
        for label in self.radio.labels:
            label.set_fontsize(11)
            label.set_fontweight('bold')
        self.radio.on_clicked(self.on_class_change)

        ax_help = self.fig.add_axes([0.83, 0.10, 0.15, 0.55])
        ax_help.axis('off')
        help_text = (
            "ATALHOS\n"
            "─────────\n"
            "B    bola (vermelho)\n"
            "P    jogador (verde)\n"
            "←→   frame ant./seg.\n"
            "S    guardar\n"
            "U    desfazer último\n"
            "R    recarregar\n"
            "F    reset zoom\n"
            "\n"
            "RATO\n"
            "─────────\n"
            "Esq  desenhar bbox\n"
            "Dir  apagar bbox\n"
            "Meio drag = pan\n"
            "Scroll = zoom\n"
            "(pinch no touchpad)\n"
        )
        ax_help.text(0, 1, help_text, fontsize=9, family='monospace',
                     verticalalignment='top')

        # botões em baixo
        ax_prev = self.fig.add_axes([0.05, 0.02, 0.10, 0.05])
        ax_next = self.fig.add_axes([0.17, 0.02, 0.10, 0.05])
        ax_save = self.fig.add_axes([0.32, 0.02, 0.13, 0.05])
        ax_undo = self.fig.add_axes([0.47, 0.02, 0.10, 0.05])
        ax_skip = self.fig.add_axes([0.59, 0.02, 0.10, 0.05])

        self.btn_prev = Button(ax_prev, '◀ anterior')
        self.btn_next = Button(ax_next, 'seguinte ▶')
        self.btn_save = Button(ax_save, '💾 guardar + seguinte',
                               color='#27ae60', hovercolor='#2ecc71')
        self.btn_undo = Button(ax_undo, '↶ desfazer')
        self.btn_skip = Button(ax_skip, '⏭ saltar')

        self.btn_prev.on_clicked(lambda e: self.navigate(-1))
        self.btn_next.on_clicked(lambda e: self.navigate(+1))
        self.btn_save.on_clicked(lambda e: self.save_and_next())
        self.btn_undo.on_clicked(lambda e: self.undo())
        self.btn_skip.on_clicked(lambda e: self.navigate(+1))

        self.fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event',      self.on_key)
        self.fig.canvas.mpl_connect('scroll_event',         self.on_scroll)
        self.fig.canvas.mpl_connect('close_event',          self.on_close)

        self.load_frame()
        print(f"\n📍 Retomar em frame {self.idx+1}/{len(self.image_paths)}  "
              f"({len(self.reviewed)} já revistos)\n")
        plt.show()

    # ── persistência do progresso ─────────────────────────────────────────────

    def load_progress(self):
        """Devolve set de nomes de imagens já revistos."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                return set(data.get('reviewed', []))
            except Exception:
                return set()
        return set()

    def save_progress(self):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'reviewed':  sorted(self.reviewed),
                    'total':     len(self.image_paths),
                    'last_seen': os.path.basename(self.image_paths[self.idx]),
                    'updated':   datetime.now().isoformat(timespec='seconds'),
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Falhei a guardar progresso: {e}")

    def find_next_unreviewed(self, start=0, default=None):
        """Devolve idx da primeira imagem não revista a partir de `start`."""
        for i in range(start, len(self.image_paths)):
            if os.path.basename(self.image_paths[i]) not in self.reviewed:
                return i
        return default if default is not None else len(self.image_paths) - 1

    def mark_reviewed(self):
        self.reviewed.add(os.path.basename(self.image_paths[self.idx]))

    def on_close(self, event):
        self.save_progress()
        print(f"\n💾 Progresso guardado: {len(self.reviewed)}/{len(self.image_paths)} revistos")
        print(f"   Da próxima vez retomas de onde parámos.\n")

    # ── load / save ───────────────────────────────────────────────────────────

    def get_label_path(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, f"{name}.txt")

    def load_frame(self):
        img_path = self.image_paths[self.idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Não consegui ler {img_path}")
            return
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = img.shape[:2]

        self.boxes = []
        label_path = self.get_label_path(img_path)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = (cx - bw/2) * self.img_w
                        y1 = (cy - bh/2) * self.img_h
                        x2 = (cx + bw/2) * self.img_w
                        y2 = (cy + bh/2) * self.img_h
                        self.boxes.append((cls, x1, y1, x2, y2))

        self.modified = False
        self.redraw()

    def save(self):
        img_path = self.image_paths[self.idx]
        label_path = self.get_label_path(img_path)
        with open(label_path, 'w') as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                cx = (x1 + x2) / 2 / self.img_w
                cy = (y1 + y2) / 2 / self.img_h
                bw = (x2 - x1) / self.img_w
                bh = (y2 - y1) / self.img_h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.modified = False
        self.mark_reviewed()
        self.save_progress()
        n_balls = sum(1 for b in self.boxes if b[0] == 0)
        print(f"   💾 [{len(self.reviewed)}/{len(self.image_paths)}] "
              f"{os.path.basename(img_path)} → {len(self.boxes)} boxes ({n_balls} bolas)")

    def save_and_next(self):
        self.save()
        self.navigate(+1)

    # ── navegação ────────────────────────────────────────────────────────────

    def navigate(self, delta):
        if self.modified:
            self.save()
        new_idx = self.idx + delta
        if 0 <= new_idx < len(self.image_paths):
            self.idx = new_idx
            self.load_frame()
        else:
            print(f"   (limite — frame {self.idx+1}/{len(self.image_paths)})")

    # ── desenho ──────────────────────────────────────────────────────────────

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.ax.axis('off')

        for cls, x1, y1, x2, y2 in self.boxes:
            cname = CLASSES[cls]
            color = COLORS[cname]
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                             edgecolor=color, linewidth=2)
            self.ax.add_patch(rect)
            self.ax.text(x1, y1-3, cname, color=color, fontsize=9,
                         fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.7, pad=1, edgecolor='none'))

        n_balls   = sum(1 for b in self.boxes if b[0] == 0)
        n_players = sum(1 for b in self.boxes if b[0] == 1)
        cls_name  = CLASSES[self.current_class]

        img_name = os.path.basename(self.image_paths[self.idx])
        status   = "✅" if img_name in self.reviewed else "  "
        title = (f"{status} [{self.idx+1}/{len(self.image_paths)}, "
                 f"{len(self.reviewed)} revistos]  {img_name}   "
                 f"⚽ {n_balls}   👤 {n_players}   |   classe activa: {cls_name.upper()}")
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle()

    # ── mouse ────────────────────────────────────────────────────────────────

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:                       # esquerdo: começa drag
            self.drag_start = (event.xdata, event.ydata)
        elif event.button == 2:                     # meio: pan
            self.pan_start = (event.xdata, event.ydata)
        elif event.button == 3:                     # direito: apaga bbox
            x, y = event.xdata, event.ydata
            for i in range(len(self.boxes) - 1, -1, -1):
                cls, x1, y1, x2, y2 = self.boxes[i]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.boxes.pop(i)
                    self.modified = True
                    self.redraw()
                    break

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        # pan com botão do meio
        if self.pan_start is not None:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.fig.canvas.draw_idle()
            return

        # preview da bbox a ser desenhada
        if self.drag_start is None:
            return
        if self.drag_rect:
            self.drag_rect.remove()
        x1, y1 = self.drag_start
        x2, y2 = event.xdata, event.ydata
        cname = CLASSES[self.current_class]
        self.drag_rect = Rectangle(
            (min(x1, x2), min(y1, y2)),
            abs(x2 - x1), abs(y2 - y1),
            fill=False, edgecolor=COLORS[cname], linewidth=2, linestyle='--')
        self.ax.add_patch(self.drag_rect)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 2:
            self.pan_start = None
            return
        if self.drag_start is None or event.button != 1:
            return
        if event.inaxes == self.ax and event.xdata is not None:
            x1, y1 = self.drag_start
            x2, y2 = event.xdata, event.ydata
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                self.boxes.append((self.current_class, x1, y1, x2, y2))
                self.modified = True
        self.drag_start = None
        if self.drag_rect:
            self.drag_rect.remove()
            self.drag_rect = None
        self.redraw()

    def on_scroll(self, event):
        """Zoom in/out à volta do cursor — funciona com pinch no touchpad."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = 1.2
        if event.button == 'up':
            factor = 1.0 / scale        # zoom in
        elif event.button == 'down':
            factor = scale              # zoom out
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x, y = event.xdata, event.ydata

        new_xlim = (
            x - (x - cur_xlim[0]) * factor,
            x + (cur_xlim[1] - x) * factor,
        )
        new_ylim = (
            y - (y - cur_ylim[0]) * factor,
            y + (cur_ylim[1] - y) * factor,
        )
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def fit_view(self):
        """Reset do zoom para a imagem inteira."""
        self.ax.set_xlim(0, self.img_w)
        self.ax.set_ylim(self.img_h, 0)   # invertido (origem em cima)
        self.fig.canvas.draw_idle()

    # ── teclado ──────────────────────────────────────────────────────────────

    def on_class_change(self, label):
        self.current_class = CLASSES.index(label)
        self.redraw()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.modified = True
            self.redraw()

    def on_key(self, event):
        k = event.key
        if k == 'b':
            self.current_class = 0
            self.radio.set_active(0)
        elif k == 'p':
            self.current_class = 1
            self.radio.set_active(1)
        elif k == 'right':
            self.save_and_next()
        elif k == 'left':
            if self.modified:
                self.save()
            self.navigate(-1)
        elif k == 's':
            self.save()
        elif k == 'u':
            self.undo()
        elif k == 'r':
            self.load_frame()
        elif k == 'f':
            self.fit_view()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='training/annotation_pool/images')
    parser.add_argument('--labels', default='training/annotation_pool/labels')
    args = parser.parse_args()
    Annotator(args.images, args.labels)


if __name__ == '__main__':
    main()
