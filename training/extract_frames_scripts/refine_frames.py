import cv2
import os
import random
from pathlib import Path

# Configuração
VIDEO_DIR = "videos"
FRAMES_DIR = "frames"

# Frames para remover
FRAMES_TO_REMOVE = [
    "v1_ehf_cl_final/v1_ehf_cl_final_frame_0030.jpg",
    "v1_ehf_cl_final/v1_ehf_cl_final_frame_0039.jpg",
    "v2_ehf_cl_semi_final/v2_ehf_cl_semi_final_frame_0004.jpg",
    "v4_PO02_viseu_portugal/v4_PO02_viseu_portugal_frame_0006.jpg",
]

# Quantos frames faltam por vídeo
FRAMES_NEEDED = {
    "v1_ehf_cl_final": 0,           # Apenas remover 2
    "v2_ehf_cl_semi_final": 7,      # 6 em falta + 1 removido
    "v3_olympic_games": 2,           # 2 em falta
    "v4_PO02_viseu_portugal": 1,     # 1 removido
    "v5_PO01_vfc_portugal": 2,       # 2 em falta
    "v6_PO01_xico_portugal": 8,      # 8 em falta
    "v7_PO02_aac_portugal": 0,       # Perfeito
}

def remove_bad_frames():
    """Remove frames marcados como maus"""
    print("\n🗑️  REMOVENDO FRAMES MAUS")
    print("=" * 60)
    
    for frame_path in FRAMES_TO_REMOVE:
        full_path = os.path.join(FRAMES_DIR, frame_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"   ✓ Removido: {frame_path}")
        else:
            print(f"   ⚠ Não encontrado: {frame_path}")
    
    print("=" * 60)


def get_next_frame_number(output_folder, video_name):
    """Descobre o próximo número de frame disponível"""
    existing_frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    if not existing_frames:
        return 0
    
    numbers = []
    for f in existing_frames:
        try:
            num = int(f.split('_frame_')[1].split('.')[0])
            numbers.append(num)
        except:
            continue
    
    return max(numbers) + 1 if numbers else 0


def add_frames_random(video_path, output_folder, video_name, num_frames):
    """Mostra frames aleatórios até completar o número necessário"""
    
    if num_frames == 0:
        return 0
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 {video_name}")
    print(f"   Frames a adicionar: {num_frames}")
    print(f"   Total frames no vídeo: {total_frames}")
    print(f"\n   Controlos:")
    print(f"   - Y ou SPACE = Guardar este frame")
    print(f"   - N ou qualquer tecla = Mostrar próximo (aleatório)")
    print(f"   - Q ou ESC = Concluir este vídeo")
    print()
    
    saved_count = 0
    start_number = get_next_frame_number(output_folder, video_name)
    shown_frames = set()  # Para evitar mostrar o mesmo frame 2x
    
    while saved_count < num_frames:
        # Escolhe frame aleatório
        # Evita frames já mostrados (até certo ponto)
        attempts = 0
        while attempts < 50:
            random_frame = random.randint(0, total_frames - 1)
            if random_frame not in shown_frames or len(shown_frames) > total_frames * 0.8:
                break
            attempts += 1
        
        shown_frames.add(random_frame)
        
        # Vai para esse frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        
        if not ret:
            print("   ⚠ Erro ao ler frame, tentando outro...")
            continue
        
        # Preview
        preview = cv2.resize(frame, (1280, 720))
        
        # Timestamp
        timestamp = random_frame / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # Info
        cv2.putText(preview, 
                   f"Frame {saved_count+1}/{num_frames} | Time: {minutes:02d}:{seconds:02d}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(preview,
                   f"Y/SPACE=Save | N/Any=Next Random | Q/ESC=Done",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview,
                   f"Frame #{random_frame} de {total_frames}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(f'Preview - {video_name}', preview)
        
        # Espera input
        key = cv2.waitKey(0) & 0xFF
        
        # Y ou Space = Guardar
        if key == ord('y') or key == ord('Y') or key == 32:
            frame_filename = f"{video_name}_frame_{start_number + saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            print(f"   ✓ Frame {saved_count}/{num_frames} guardado: {frame_filename}")
        
        # Q ou ESC = Quit
        elif key == ord('q') or key == ord('Q') or key == 27:
            print(f"   ⚠ Concluído: {saved_count}/{num_frames} frames adicionados")
            break
        
        # Qualquer outra tecla = próximo aleatório (loop continua)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"   ✅ Total adicionado: {saved_count} frames\n")
    return saved_count


def main():
    print("=" * 70)
    print("🔧 REFINAMENTO DE FRAMES - DATASET DE ANDEBOL")
    print("=" * 70)
    
    # Passo 1: Remover frames maus
    remove_bad_frames()
    
    # Passo 2: Adicionar frames em falta
    print("\n➕ ADICIONANDO FRAMES EM FALTA (ALEATÓRIOS)")
    print("=" * 70)
    
    total_added = 0
    
    for video_name, num_needed in FRAMES_NEEDED.items():
        if num_needed == 0:
            print(f"\n✓ {video_name}: Completo, nada a fazer")
            continue
        
        # Procurar vídeo
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            test_path = os.path.join(VIDEO_DIR, video_name + ext)
            if os.path.exists(test_path):
                video_path = test_path
                break
        
        if not video_path:
            print(f"\n❌ Vídeo não encontrado: {video_name}")
            continue
        
        # Adicionar frames
        output_folder = os.path.join(FRAMES_DIR, video_name)
        added = add_frames_random(video_path, output_folder, video_name, num_needed)
        total_added += added
        
        # Perguntar se quer continuar
        remaining_videos = [v for v, n in FRAMES_NEEDED.items() 
                          if list(FRAMES_NEEDED.keys()).index(v) > list(FRAMES_NEEDED.keys()).index(video_name) 
                          and n > 0]
        
        if remaining_videos:
            print(f"\n   Continuar para próximo vídeo? (C=Sim, Q=Não): ", end='')
            choice = input().lower()
            if choice == 'q':
                print("\n⚠ Processo interrompido")
                break
    
    print("\n" + "=" * 70)
    print(f"✅ REFINAMENTO COMPLETO!")
    print(f"   Frames removidos: {len(FRAMES_TO_REMOVE)}")
    print(f"   Frames adicionados: {total_added}")
    print(f"   Dataset em: {FRAMES_DIR}/")
    print("=" * 70)
    
    # Resumo final
    print("\n📊 RESUMO POR VÍDEO:")
    print("-" * 70)
    
    expected = {
        "v1_ehf_cl_final": 50,
        "v2_ehf_cl_semi_final": 40,
        "v3_olympic_games": 60,
        "v4_PO02_viseu_portugal": 30,
        "v5_PO01_vfc_portugal": 40,
        "v6_PO01_xico_portugal": 40,
        "v7_PO02_aac_portugal": 30,
    }
    
    total = 0
    for video_name, expected_count in expected.items():
        folder = os.path.join(FRAMES_DIR, video_name)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
            status = "✓" if count == expected_count else f"⚠ (esperado: {expected_count})"
            print(f"   {video_name}: {count} frames {status}")
            total += count
        else:
            print(f"   {video_name}: Pasta não encontrada")
    
    print("-" * 70)
    print(f"   TOTAL: {total} frames")
    print("-" * 70)


if __name__ == "__main__":
    main()
