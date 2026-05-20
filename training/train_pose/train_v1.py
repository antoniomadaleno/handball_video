"""
Handball Court Keypoints v2 - FIXED
Mais conservador para dataset pequeno
"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("🏐 HANDBALL KEYPOINTS v2 - CONSERVATIVE TRAINING")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ GPU necessária!")
        exit(1)
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Modelo MENOR para dataset pequeno
    model = YOLO('yolov8s.pt')  # Small em vez de Medium!
    
    results = model.train(
        data='handball-court-pose-1/data.yaml',
        epochs=100,  # MENOS epochs
        imgsz=640,
        batch=8,  # BATCH MENOR
        device="0",
        project='runs/handball_keypoints',
        name='v2_conservative',
        patience=20,  # Patience menor
        workers=0,
        save=True,
        save_period=10,
        plots=True,
        
        # LEARNING RATE MAIS BAIXO (chave!)
        lr0=0.001,  # 10x mais baixo!
        lrf=0.00001,
        momentum=0.9,
        weight_decay=0.001,  # Mais regularização
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        
        # LOSS WEIGHTS NORMAIS:
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # AUGMENTATION REDUZIDA (dataset pequeno):
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=0.0,  # SEM rotação
        translate=0.05,
        scale=0.2,  # Scaling mínimo
        mosaic=0.0,  # SEM mosaic!
        mixup=0.0,  # SEM mixup!
        copy_paste=0.0,  # SEM copy_paste!
        close_mosaic=0,
        
        flipud=0.0,
        fliplr=0.3,  # Só flip horizontal leve
        
        # DROPOUT para evitar overfitting:
        dropout=0.1,
    )
    
    print("\n✅ TREINO COMPLETO!")
    print(f"📁 Modelo: runs/handball_keypoints/v2_conservative/weights/best.pt")

if __name__ == '__main__':
    main()