import cv2
import os
import random
import shutil
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

# Frames esperados por vídeo
EXPECTED_FRAMES = {
    "v1_ehf_cl_final": 50,
    "v2_ehf_cl_semi_final": 40,
    "v3_olympic_games": 60,
    "v4_PO02_viseu_portugal": 30,
    "v5_PO01_vfc_portugal": 40,
    "v6_PO01_xico_portugal": 40,
    "v7_PO02_aac_portugal": 30,
}


def remove_bad_frames():
    """Remove frames marcados como maus"""
    print("\n🗑️  REMOVENDO FRAMES MAUS")
    print("=" * 60)
    
    removed = 0
    for frame_path in FRAMES_TO_REMOVE:
        full_path = os.path.join(FRAMES_DIR, frame_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"   ✓ Removido: {frame_path}")
            removed += 1
        else:
            print(f"   ⚠ Não encontrado: {frame_path}")
    
    print(f"\n   Total removido: {removed}")
    print("=" * 60)
    return removed


def renumber_frames(video_name):
    """Renumera frames para sequência contínua 0000, 0001, 0002..."""
    folder = os.path.join(FRAMES_DIR, video_name)
    
    if not os.path.exists(folder):
        return 0
    
    # Lista todos os frames
    frames = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    
    if not frames:
        return 0
    
    print(f"\n🔢 Renumerando: {video_name}")
    print(f"   Frames atuais: {len(frames)}")
    
    # Cria pasta temporária
    temp_folder = folder + "_temp"
    os.makedirs(temp_folder, exist_ok=True)
    
    # Move e renumera
    for i, frame_file in enumerate(frames):
        old_path = os.path.join(folder, frame_file)
        new_name = f"{video_name}_frame_{i:04d}.jpg"
        new_path = os.path.join(temp_folder, new_name)
        shutil.copy2(old_path, new_path)
    
    # Remove originais e move renumerados
    for frame_file in frames:
        os.remove(os.path.join(folder, frame_file))
    
    for frame_file in os.listdir(temp_folder):
        shutil.move(os.path.join(temp_folder, frame_file), 
                   os.path.join(folder, frame_file))
    
    os.rmdir(temp_folder)
    
    print(f"   ✓ Renumerado: 0000 até {len(frames)-1:04d}")
    return len(frames)


def count_frames(video_name):
    """Conta frames existentes"""
    folder = os.path.join(FRAMES_DIR, video_name)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.jpg')])


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
    
    # Conta quantos já existem (depois de renumerar)
    current_count = count_frames(video_name)
    saved_count = 0
    shown_frames = set()
    
    while saved_count < num_frames:
        # Frame aleatório
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
            continue
        
        # Preview
        preview = cv2.resize(frame, (1280, 720))
        
        # Timestamp
        timestamp = random_frame / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # Info
        cv2.putText(preview, 
                   f"Adicionando: {saved_count+1}/{num_frames} | Time: {minutes:02d}:{seconds:02d}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(preview,
                   f"Y/SPACE=Save | Any=Next Random | Q/ESC=Done",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview,
                   f"Frame #{random_frame} de {total_frames}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(f'Preview - {video_name}', preview)
        
        key = cv2.waitKey(0) & 0xFF
        
        # Y ou Space = Guardar
        if key == ord('y') or key == ord('Y') or key == 32:
            # Número sequencial após os existentes
            frame_number = current_count + saved_count
            frame_filename = f"{video_name}_frame_{frame_number:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            print(f"   ✓ Frame {saved_count}/{num_frames} guardado: {frame_filename}")
        
        # Q ou ESC = Quit
        elif key == ord('q') or key == ord('Q') or key == 27:
            print(f"   ⚠ Concluído: {saved_count}/{num_frames} frames adicionados")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"   ✅ Total adicionado: {saved_count} frames\n")
    return saved_count


def main():
    print("=" * 70)
    print("🔧 REFINAMENTO E RENUMERAÇÃO DE FRAMES")
    print("=" * 70)
    
    # Passo 1: Remover frames maus
    removed = remove_bad_frames()
    
    # Passo 2: Renumerar todos os vídeos
    print("\n🔢 RENUMERANDO FRAMES")
    print("=" * 70)
    
    for video_name in EXPECTED_FRAMES.keys():
        renumber_frames(video_name)
    
    # Passo 3: Verificar quantos faltam
    print("\n📊 VERIFICANDO GAPS")
    print("=" * 70)
    
    frames_needed = {}
    for video_name, expected in EXPECTED_FRAMES.items():
        current = count_frames(video_name)
        gap = expected - current
        frames_needed[video_name] = gap
        
        if gap > 0:
            print(f"   {video_name}: {current}/{expected} frames (faltam {gap})")
        else:
            print(f"   {video_name}: {current}/{expected} frames ✓")
    
    # Passo 4: Adicionar frames em falta
    if any(gap > 0 for gap in frames_needed.values()):
        print("\n➕ ADICIONANDO FRAMES EM FALTA")
        print("=" * 70)
        
        total_added = 0
        
        for video_name, num_needed in frames_needed.items():
            if num_needed == 0:
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
            remaining = [v for v, n in frames_needed.items() 
                        if list(frames_needed.keys()).index(v) > list(frames_needed.keys()).index(video_name) 
                        and n > 0]
            
            if remaining:
                print(f"\n   Continuar para próximo vídeo? (C=Sim, Q=Não): ", end='')
                choice = input().lower()
                if choice == 'q':
                    print("\n⚠ Processo interrompido")
                    break
    
    # Resumo final
    print("\n" + "=" * 70)
    print(f"✅ PROCESSO COMPLETO!")
    print("=" * 70)
    
    print("\n📊 RESUMO FINAL:")
    print("-" * 70)
    
    total = 0
    all_complete = True
    
    for video_name, expected in EXPECTED_FRAMES.items():
        folder = os.path.join(FRAMES_DIR, video_name)
        if os.path.exists(folder):
            count = count_frames(video_name)
            
            if count == expected:
                status = "✓"
            else:
                status = f"⚠ (esperado: {expected})"
                all_complete = False
            
            print(f"   {video_name}: {count} frames {status}")
            total += count
        else:
            print(f"   {video_name}: Pasta não encontrada ❌")
            all_complete = False
    
    print("-" * 70)
    print(f"   TOTAL: {total}/290 frames")
    
    if all_complete:
        print(f"   STATUS: ✅ DATASET COMPLETO E RENUMERADO!")
    else:
        print(f"   STATUS: ⚠ Alguns vídeos ainda precisam de frames")
    
    print("-" * 70)


if __name__ == "__main__":
    main()

