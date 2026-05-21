"""
Handball v4 — YOLOv8l (large)
Melhoria de recall: modelo maior detecta melhor jogadores ocluídos e pequenos.
Mesmos hiperparâmetros do v3_final que deu os melhores resultados.
"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("🏐 HANDBALL v4 — YOLOv8l (LARGE)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO('yolov8l.pt')   # descarrega automaticamente se não existir

    results = model.train(
        data='training/handball-detection-3/data.yaml',
        epochs=250,
        imgsz=640,
        batch=8,           # reduzido para YOLOv8l (mais VRAM)
        device="0",
        project='training/runs/handball',
        name='v4_large',
        patience=0,
        workers=0,
        save=True,
        save_period=50,
        plots=True,

        # Hiperparâmetros iguais ao v3_final
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5,
        cls=0.8,
        dfl=1.5,
        iou=0.5,

        # Augmentation contínua (close_mosaic=0 evita spikes)
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

        # Desactivado (handball não é simétrico)
        flipud=0.0,
        fliplr=0.0,
    )

    print("\n✅ TREINO COMPLETO!")
    print(f"📁 Modelo: training/runs/handball/v4_large/weights/best.pt")

if __name__ == '__main__':
    main()
