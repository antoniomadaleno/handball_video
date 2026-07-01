"""
Handball v5 — YOLOv8l @ imgsz=1280

Objetivo: melhorar Ball mAP. A bola tem ~20px na imagem original e fica
~10px depois do resize para 640. Com imgsz=1280 a bola fica 2-4× maior
na entrada do modelo, permitindo deteções muito melhores.

Dataset igual ao v4 (handball-detection-3).
Hiperparâmetros base iguais ao v3_final/v4_large, com batch reduzido
para acomodar a VRAM extra do imgsz=1280.
"""
from ultralytics import YOLO
import torch


def main():
    print("=" * 70)
    print("🏐 HANDBALL v5 — YOLOv8l @ imgsz=1280  (FOCO NA BOLA)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM: {vram_gb:.1f} GB")

    # batch automático conforme VRAM (imgsz=1280 consome ~4× mais memória)
    # 8GB: batch=2,  12GB: batch=4,  16GB+: batch=6
    if vram_gb < 10:
        batch = 2
    elif vram_gb < 14:
        batch = 4
    else:
        batch = 6
    print(f"   Batch: {batch}\n")

    # parte dos pesos do v4_large (transfer learning) em vez de COCO
    # — acelera convergência e mantém o que já aprendeu sobre jogadores
    model = YOLO('training/runs/handball/v4_large/weights/best.pt')

    results = model.train(
        data='training/handball-detection-3/data.yaml',
        epochs=200,            # menos épocas — partimos de bons pesos
        imgsz=1280,            # ✱ chave para melhorar deteção da bola
        batch=batch,
        device="0",
        project='training/runs/handball',
        name='v5_1280',
        patience=0,
        workers=0,
        save=True,
        save_period=25,
        plots=True,

        # Hiperparâmetros (iguais ao v4, lr0 mais baixo por partir de pesos treinados)
        lr0=0.005,             # metade — fine-tuning, não treino do zero
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,     # menos warmup — pesos já treinados
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=0.8,
        dfl=1.5,
        iou=0.5,

        # Augmentation (igual ao v4)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=0,

        flipud=0.0,
        fliplr=0.0,
    )

    print("\n✅ TREINO COMPLETO!")
    print(f"📁 Modelo: training/runs/handball/v5_1280/weights/best.pt")
    print(f"\nPara usar este modelo, atualiza MODEL_PATH em:")
    print(f"   src/pipeline_sam2.py")
    print(f"   src/analyze_ball.py")


if __name__ == '__main__':
    main()
