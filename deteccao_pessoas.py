from ultralytics import YOLO
import cv2

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Verificar se a webcam foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()


# Liberar a captura de vídeo e fechar todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
