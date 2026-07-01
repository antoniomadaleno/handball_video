"""
Funde os 3 datasets num só (training/handball-combined/) pronto para o v7.

Datasets fonte:
  1. training/handball-detection-3       — 285 imgs, [ball, player] (já no formato final)
  2. handball.v1i.yolov8                 — 658 imgs, [ball, dribble, goal, player]
  3. handball.v11i.yolov8                — 2546 imgs, [Arbitro, Area, Jugador, Pelota, Porteria, Portero]

Re-mapeamento para o nosso formato (ball=0, player=1):
  - DS1: identity (já está certo)
  - DS2: ball=0→0, player=3→1; dribble e goal eliminados
  - DS3: Pelota=3→0, Jugador=2→1, Portero=5→1; Arbitro, Area, Porteria eliminados

Imagens são copiadas com prefixo (ds1_, ds2_, ds3_) para evitar colisões de nomes.
"""
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm


DST_DIR = 'training/handball-combined'

# (caminho fonte, prefixo, mapa de classes velho→novo, nome split valid no original)
DATASETS = [
    {
        'name':       'ours (handball-detection-3)',
        'src':        'training/handball-detection-3',
        'prefix':     'ds1',
        # classes originais: [ball=0, player=1]
        'class_map':  {0: 0, 1: 1},
        'valid_dir':  'valid',
    },
    {
        'name':       'handball-6hu75 (658 imgs)',
        'src':        'handball.v1i.yolov8',
        'prefix':     'ds2',
        # originais: [ball=0, dribble=1, goal=2, player=3]
        'class_map':  {0: 0, 1: None, 2: None, 3: 1},
        'valid_dir':  'valid',
    },
    {
        'name':       'handball-y6iy6 (2546 imgs)',
        'src':        'handball.v11i.yolov8',
        'prefix':     'ds3',
        # originais: [Arbitro=0, Area=1, Jugador=2, Pelota=3, Porteria=4, Portero=5]
        'class_map':  {0: None, 1: None, 2: 1, 3: 0, 4: None, 5: 1},
        'valid_dir':  'valid',
    },
]


def convert_label(src_path, dst_path, class_map):
    """
    Reescreve o ficheiro de label aplicando class_map.
    Aceita bboxes (5 valores) E polígonos de segmentação (cls + 2N pares x,y).
    Polígonos são convertidos para bbox via min/max.
    Devolve nº de bboxes mantidos.
    """
    lines_out = []
    if os.path.exists(src_path):
        with open(src_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                old_cls = int(parts[0])
                new_cls = class_map.get(old_cls)
                if new_cls is None:
                    continue

                if len(parts) == 5:
                    # já é bbox: cls cx cy w h
                    cx, cy, w, h = map(float, parts[1:])
                else:
                    # polígono: cls x1 y1 x2 y2 ... → converte para bbox
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0 or len(coords) < 6:
                        continue
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    w  = x_max - x_min
                    h  = y_max - y_min
                    if w <= 0 or h <= 0:
                        continue

                lines_out.append(
                    f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")
    return len(lines_out)


def process_split(ds, split_name, dst_split_dir):
    """Processa um split de um dataset. Devolve (n_imgs, n_bboxes)."""
    src_img_dir = os.path.join(ds['src'], split_name, 'images')
    src_lbl_dir = os.path.join(ds['src'], split_name, 'labels')

    if not os.path.exists(src_img_dir):
        return 0, 0

    dst_img_dir = os.path.join(dst_split_dir, 'images')
    dst_lbl_dir = os.path.join(dst_split_dir, 'labels')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(src_img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    n_imgs = 0
    n_bboxes = 0
    for img in tqdm(images, desc=f"   {ds['prefix']}/{split_name}", leave=False):
        base = os.path.splitext(img)[0]
        ext  = os.path.splitext(img)[1]
        new_name = f"{ds['prefix']}_{base}"

        src_lbl = os.path.join(src_lbl_dir, base + '.txt')
        dst_lbl = os.path.join(dst_lbl_dir, new_name + '.txt')
        n_bb = convert_label(src_lbl, dst_lbl, ds['class_map'])

        shutil.copy2(os.path.join(src_img_dir, img),
                     os.path.join(dst_img_dir, new_name + ext))
        n_imgs   += 1
        n_bboxes += n_bb

    return n_imgs, n_bboxes


def main():
    # validação prévia
    for ds in DATASETS:
        if not os.path.exists(ds['src']):
            print(f"❌ Não encontro: {ds['src']}")
            return

    print(f"📦 A fundir 3 datasets → {DST_DIR}\n")

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)

    totals = {'train': [0, 0], 'valid': [0, 0], 'test': [0, 0]}

    for ds in DATASETS:
        print(f"▸ {ds['name']}")
        for split in ['train', 'valid', 'test']:
            # alguns datasets usam 'val' em vez de 'valid'
            split_in_src = split if os.path.exists(os.path.join(ds['src'], split)) else (
                          'val' if split == 'valid' and os.path.exists(os.path.join(ds['src'], 'val')) else split)
            n_i, n_b = process_split(
                {**ds, 'src': ds['src']},
                split_in_src,
                os.path.join(DST_DIR, split),
            )
            totals[split][0] += n_i
            totals[split][1] += n_b
            print(f"     {split:>5}: {n_i:>5} imgs  |  {n_b:>5} bboxes")
        print()

    # data.yaml final
    data_yaml = {
        'train': '../train/images',
        'val':   '../valid/images',
        'test':  '../test/images',
        'nc':    2,
        'names': ['ball', 'player'],
    }
    with open(os.path.join(DST_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"{'─'*55}")
    print(f"✅ TOTAL no dataset combinado:")
    for split, (n_i, n_b) in totals.items():
        print(f"   {split:>5}: {n_i:>5} imagens  |  {n_b:>5} bboxes")
    print(f"\n💾 {DST_DIR}/data.yaml")
    print(f"\n   Próximo passo: python training/train_v7.py")


if __name__ == '__main__':
    main()
