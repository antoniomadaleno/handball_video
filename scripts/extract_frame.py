import cv2

cap = cv2.VideoCapture('video_teste_27_35.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
if ret:
    cv2.imwrite('video_ref.jpg', frame)
    print('✅ Frame 100 guardado: video_ref.jpg')
cap.release()