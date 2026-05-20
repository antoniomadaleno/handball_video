#!/usr/bin/env python3
"""
Verifica se a câmara se mexe durante o vídeo.
Extrai frames e compara visualmente.
"""
import cv2
import argparse
import os

def check_camera_movement(video_path, output_dir='camera_check', num_samples=10):
    """
    Extrai frames ao longo do vídeo para verificar movimento de câmara.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"\n📹 Análise do Vídeo")
    print("="*70)
    print(f"Total de frames: {total_frames}")
    print(f"FPS: {fps:.1f}")
    print(f"Duração: {duration:.1f}s")
    
    # Criar diretório
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair frames uniformemente distribuídos
    frame_indices = [int(i * total_frames / num_samples) for i in range(num_samples)]
    
    print(f"\n🎬 A extrair {num_samples} frames de amostra...")
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            timestamp = frame_idx / fps
            output_path = os.path.join(output_dir, f'frame_{i:02d}_t{timestamp:.1f}s.jpg')
            cv2.imwrite(output_path, frame)
            print(f"  ✅ Frame {frame_idx:5d} (t={timestamp:6.1f}s) → {output_path}")
    
    cap.release()
    
    print(f"\n📊 ANÁLISE:")
    print(f"   • {num_samples} frames guardados em: {output_dir}/")
    print(f"\n🔍 PRÓXIMO PASSO:")
    print(f"   1. Abre a pasta '{output_dir}/'")
    print(f"   2. Compara visualmente os frames")
    print(f"   3. Verifica se:")
    print(f"      ✅ Câmara FIXA → Campo aparece sempre na mesma posição")
    print(f"      ❌ Câmara MÓVEL → Campo muda de posição/ângulo entre frames")
    print(f"\n💡 DICAS para identificar câmara móvel:")
    print(f"   • Linhas do campo aparecem em posições diferentes")
    print(f"   • Zoom in/out (campo maior/menor)")
    print(f"   • Pan (câmara roda horizontalmente)")
    print(f"   • Tilt (câmara roda verticalmente)")
    print(f"\n⚠️  Se a câmara SE MEXE:")
    print(f"   → Precisas de homografia DIFERENTE para cada segmento do vídeo!")
    print(f"   → OU usa apenas frames de um segmento com câmara fixa")

def main():
    parser = argparse.ArgumentParser(
        description='Verifica movimento de câmara no vídeo'
    )
    parser.add_argument('--video', required=True, help='Vídeo a analisar')
    parser.add_argument('--output', default='camera_check', help='Pasta para frames')
    parser.add_argument('--samples', type=int, default=10, help='Número de frames a extrair')
    
    args = parser.parse_args()
    
    check_camera_movement(args.video, args.output, args.samples)

if __name__ == '__main__':
    main()