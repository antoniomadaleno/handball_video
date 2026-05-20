"""
Handball v3 - FINAL OPTIMIZATION
Remove close_mosaic spikes + longer warmup
"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("🏐 HANDBALL v3 - FINAL OPTIMIZATION (NO CLOSE_MOSAIC)")
    print("=" * 70)
    
    # Verifica GPU
    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # YOLOv8m
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data='handball-detection-3/data.yaml',
        epochs=250,  # PARA ANTES DOS SPIKES
        imgsz=640,
        batch=12,
        device="0",
        project='runs/handball',
        name='v3_final',
        patience=0,
        workers=0,
        save=True,
        save_period=50,
        plots=True,
        
        # HYPERPARAMETERS OPTIMIZED:
        lr0=0.01,
        lrf=0.0001,  # AINDA MAIS BAIXO (vs 0.001)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10.0,  # WARMUP MAIS LONGO (vs 5.0)
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # LOSS WEIGHTS:
        box=7.5,
        cls=0.8,
        dfl=1.5,
        iou=0.5,
        
        # AUGMENTATION (mantém sempre ativo):
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=0,  # ← NUNCA DESLIGA! KEY CHANGE
        
        # DISABLE:
        flipud=0.0,
        fliplr=0.0,
    )
    
    print("\n✅ TREINO COMPLETO!")
    print(f"📁 Modelo: runs/handball/v3_final/weights/best.pt")

if __name__ == '__main__':
    main()
