"""
Corta vídeo do segundo 27 ao 35
"""
import cv2

def cut_video(input_path, output_path, start_sec=27, end_sec=35):
    """
    Corta vídeo entre start_sec e end_sec
    """
    print(f"📹 A abrir vídeo: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    # Info vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   FPS: {fps}, Resolução: {width}x{height}")
    
    # Frames de início e fim
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = end_frame - start_frame
    
    print(f"✂️  A cortar: {start_sec}s → {end_sec}s ({total_frames} frames)")
    
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Avança para frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Copia frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
        if i % 50 == 0:
            print(f"   Progresso: {i}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"✅ Vídeo cortado guardado: {output_path}")
    print(f"   Duração: {end_sec - start_sec} segundos")

if __name__ == '__main__':
    # CONFIGURAÇÃO:
    # input_video = 'video_completo.mp4'      # ← Muda aqui
    # output_video = 'video_teste_27-35.mp4'  # ← Output
    input_video = 'video_zmerding.mp4'      # ← Muda aqui
    output_video = 'video_teste_zmerding.mp4'  # ← Output


    cut_video(input_video, output_video, start_sec=67, end_sec=75)