"""
Converte o dataset 'handball.v11i.yolov8' (Roboflow Universe) para o nosso formato:
  - Classes do dataset original (6): Arbitro, Area, Jugador, Pelota, Porteria, Portero
  - Mantemos só: Jugador (2), Pelota (3), Portero (5)
  - Re-mapeamos:  Pelota→0 (ball), Jugador→1 (player), Portero→1 (player)
  - Eliminamos:   Arbitro, Area, Porteria

Output: training/handball-extra-clean/  (formato YOLOv8 pronto para treino)
"""
import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

SRC_DIR  = 'handball.v11i.yolov8'
DST_DIR  = 'training/handball-extra-clean'

# class_id antigo → class_id novo (None = eliminar)
CLASS_MAP = {
    0: None,   # Arbitro
    1: None,   # Area
    2: 1,      # Jugador → player
    3: 0,      # Pelota → ball
    4: None,   # Porteria
    5: 1,      # Portero → player
}


def convert_label_file(src_path, dst_path):
    """Reescreve label YOLO aplicando CLASS_MAP."""
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_cls = int(parts[0])
            new_cls = CLASS_MAP.get(old_cls)
            if new_cls is None:
                continue
            lines_out.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")


def convert_split(split_name):
    src_img_dir = os.path.join(SRC_DIR, split_name, 'images')
    src_lbl_dir = os.path.join(SRC_DIR, split_name, 'labels')
    dst_img_dir = os.path.join(DST_DIR, split_name, 'images')
    dst_lbl_dir = os.path.join(DST_DIR, split_name, 'labels')

    if not os.path.exists(src_img_dir):
        print(f"   (sem split {split_name})")
        return 0, 0

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(src_img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    n_kept_imgs   = 0
    n_kept_labels = 0

    for img in tqdm(images, desc=f"   {split_name}"):
        base = os.path.splitext(img)[0]
        src_lbl = os.path.join(src_lbl_dir, base + '.txt')
        dst_lbl = os.path.join(dst_lbl_dir, base + '.txt')

        # converte label (ou cria vazio se não existia)
        if os.path.exists(src_lbl):
            convert_label_file(src_lbl, dst_lbl)
        else:
            with open(dst_lbl, 'w') as f:
                pass

        # copia imagem (sempre, mesmo se o label ficar vazio — frames de "background"
        # ajudam a reduzir falsos positivos)
        shutil.copy2(os.path.join(src_img_dir, img),
                     os.path.join(dst_img_dir, img))
        n_kept_imgs += 1
        with open(dst_lbl) as f:
            n_kept_labels += sum(1 for _ in f if _.strip())

    return n_kept_imgs, n_kept_labels


def main():
    if not os.path.exists(SRC_DIR):
        print(f"❌ Pasta {SRC_DIR} não encontrada.")
        return

    print(f"📦 A converter '{SRC_DIR}' → '{DST_DIR}'")
    print(f"   Classes mantidas: Jugador, Pelota, Portero (→ ball=0, player=1)\n")

    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)

    totals = {}
    for split in ['train', 'valid', 'test']:
        n_imgs, n_lbls = convert_split(split)
        totals[split] = (n_imgs, n_lbls)

    # cria data.yaml compatível com o nosso pipeline
    data_yaml = {
        'train': '../train/images',
        'val':   '../valid/images',
        'test':  '../test/images',
        'nc':    2,
        'names': ['ball', 'player'],
    }
    with open(os.path.join(DST_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Conversão completa")
    for split, (n_i, n_l) in totals.items():
        print(f"   {split:>5}: {n_i:>5} imagens  |  {n_l:>5} bboxes mantidos")
    print(f"\n💾 Dataset: {DST_DIR}/data.yaml")
    print(f"\n   Próximo passo:")
    print(f"   python training/train_v6.py")


if __name__ == '__main__':
    main()
