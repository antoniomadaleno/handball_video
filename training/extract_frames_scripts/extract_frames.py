import cv2
import os
from pathlib import Path

# Configuração
VIDEO_DIR = "videos"
OUTPUT_DIR = "frames"

FRAMES_PER_VIDEO = {
    "v1_ehf_cl_final": 50,
    "v2_ehf_cl_semi_final": 40,
    "v3_olympic_games": 60,
    "v4_PO02_viseu_portugal": 30,
    "v5_PO01_vfc_portugal": 40,
    "v6_PO01_xico_portugal": 40,
    "v7_PO02_aac_portugal": 30,
}

def extract_frames_with_preview(video_path, output_folder, num_frames):
    """
    Extrai frames com preview - tu decides quais guardar
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"\n📹 {Path(video_path).stem}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Duração: {duration:.1f}s")
    print(f"   Frames a extrair: {num_frames}")
    print(f"\n   Controlos:")
    print(f"   - Y ou SPACE = Guardar frame")
    print(f"   - N = Skip (avançar para próximo)")
    print(f"   - Q ou ESC = Sair")
    print(f"   - R = Retry (recuar 2 segundos)")
    
    interval = total_frames // (num_frames * 2)  # Extrai mais candidatos
    
    frame_count = 0
    saved_count = 0
    candidates_shown = 0
    
    while cap.isOpened() and saved_count < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Mostrar frame candidato
        if frame_count % interval == 0 and candidates_shown < num_frames * 2:
            candidates_shown += 1
            
            # Resize para preview (mais rápido)
            preview = cv2.resize(frame, (960, 540))
            
            # Info no frame
            cv2.putText(preview, 
                       f"Frame {saved_count+1}/{num_frames} | Candidato {candidates_shown}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(preview,
                       f"Y/SPACE=Save | N=Skip | R=Retry | Q/ESC=Quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Preview - Frame Candidato', preview)
            
            # Espera input
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                # Y ou Space = Guardar
                if key == ord('y') or key == ord('Y') or key == 32:  # Space
                    frame_filename = f"{Path(video_path).stem}_frame_{saved_count:04d}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                    print(f"   ✓ Frame {saved_count}/{num_frames} guardado")
                    break
                
                # N = Skip
                elif key == ord('n') or key == ord('N'):
                    print(f"   ⊘ Frame {candidates_shown} ignorado")
                    break
                
                # R = Retry (recua 2 segundos)
                elif key == ord('r') or key == ord('R'):
                    new_pos = max(0, frame_count - int(fps * 2))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_count = new_pos
                    candidates_shown -= 1
                    print(f"   ↺ Recuando 2 segundos...")
                    break
                
                # Q ou ESC = Quit
                elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                    print(f"\n   ⚠ Interrompido pelo utilizador")
                    cap.release()
                    cv2.destroyAllWindows()
                    return saved_count
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"   ✅ Completo: {saved_count} frames guardados\n")
    return saved_count


def main():
    print("=" * 70)
    print("🎬 EXTRAÇÃO DE FRAMES COM PREVIEW - DATASET DE ANDEBOL")
    print("=" * 70)
    
    total_extracted = 0
    
    for video_name, num_frames in FRAMES_PER_VIDEO.items():
        # Procurar vídeo
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            test_path = os.path.join(VIDEO_DIR, video_name + ext)
            if os.path.exists(test_path):
                video_path = test_path
                break
        
        if not video_path:
            print(f"❌ Vídeo não encontrado: {video_name}")
            continue
        
        # Extrair com preview
        output_folder = os.path.join(OUTPUT_DIR, video_name)
        extracted = extract_frames_with_preview(video_path, output_folder, num_frames)
        total_extracted += extracted
        
        # Perguntar se quer continuar
        print(f"\n   Continuar para próximo vídeo?")
        print(f"   C = Continuar | Q = Sair")
        
        while True:
            key = input("   Escolha (C/Q): ").lower()
            if key == 'c':
                break
            elif key == 'q':
                print("\n⚠ Processo interrompido pelo utilizador")
                print(f"Total extraído até agora: {total_extracted} frames")
                return
    
    print("=" * 70)
    print(f"✅ COMPLETO!")
    print(f"   Total de frames extraídos: {total_extracted}")
    print(f"   Frames guardados em: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
"""
```

---

## Como Funciona

### 1. Script mostra frame candidato
```
┌─────────────────────────────────────┐
│  Frame 5/50 | Candidato 8           │
│  Y/SPACE=Save | N=Skip | R=Retry    │
│                                     │
│          [Preview do Frame]         │
│                                     │
└─────────────────────────────────────┘
"""