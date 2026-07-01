"""
Handball v6 — YOLOv8l no dataset extra (2.5k imagens da Roboflow Universe).

Objetivo: testar se um dataset 9× maior resolve o problema da bola.
Não junta com o nosso (handball-detection-3) — é um teste isolado.

Hiperparâmetros iguais ao v4_large (que deu os melhores resultados).
imgsz=640 (não 1280) — o v5 mostrou que 1280 não trouxe ganho.
"""
from ultralytics import YOLO
import torch


def main():
    print("=" * 70)
    print("🏐 HANDBALL v6 — YOLOv8l no dataset extra Roboflow (2.5k imgs)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM: {vram_gb:.1f} GB\n")

    # parte do v4_large que já aprendeu jogadores neste contexto
    model = YOLO('training/runs/handball/v4_large/weights/best.pt')

    results = model.train(
        data='training/handball-extra-clean/data.yaml',
        epochs=150,           # menos épocas — dataset grande, fine-tuning
        imgsz=640,
        batch=8,
        device="0",
        project='training/runs/handball',
        name='v6_extra',
        patience=20,          # early stopping se não melhorar por 20 épocas
        workers=0,
        save=True,
        save_period=25,
        plots=True,

        # Hiperparâmetros (iguais ao v4)
        lr0=0.005,
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
    print(f"📁 Modelo: training/runs/handball/v6_extra/weights/best.pt")
    print(f"\nPara testar:")
    print(f"   1. Atualiza MODEL_PATH em src/analyze_ball.py")
    print(f"   2. python src/analyze_ball.py")


if __name__ == '__main__':
    main()
