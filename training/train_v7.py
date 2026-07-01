"""
Handball v7 — YOLOv8l no dataset COMBINADO (3 fontes, ~3500 imagens).

Estratégia para resolver o problema de overfitting do v6:
  - Mistura imagens dos 3 datasets para forçar generalização
  - Inclui o NOSSO dataset (handball-detection-3) que tem o estilo dos
    teus vídeos de teste → o modelo não pode "esquecer" esse look
  - Parte do v4_large (já familiar com o nosso estilo) e refina com
    a variedade extra

Hiperparâmetros base iguais ao v4_large (que provou estabilidade).
"""
from ultralytics import YOLO
import torch


def main():
    print("=" * 70)
    print("🏐 HANDBALL v7 — YOLOv8l no dataset COMBINADO (~3500 imgs)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM: {vram_gb:.1f} GB\n")

    # parte do v4_large que já generaliza nos nossos vídeos
    model = YOLO('training/runs/handball/v4_large/weights/best.pt')

    results = model.train(
        data='training/handball-combined/data.yaml',
        epochs=150,
        imgsz=640,
        batch=8,
        device="0",
        project='training/runs/handball',
        name='v7_combined',
        patience=25,          # mais tolerante que o v6 (dataset maior, mais oscilação esperada)
        workers=0,
        save=True,
        save_period=25,
        plots=True,

        # Hiperparâmetros (iguais ao v4)
        lr0=0.005,            # fine-tuning a partir do v4
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=0.8,
        dfl=1.5,
        iou=0.5,

        # Augmentation
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
    print(f"📁 Modelo: training/runs/handball/v7_combined/weights/best.pt")
    print(f"\nPara testar:")
    print(f"   python -c \"from ultralytics import YOLO; YOLO('training/runs/handball/v7_combined/weights/best.pt').val(data='training/handball-combined/data.yaml', verbose=True)\"")


if __name__ == '__main__':
    main()
