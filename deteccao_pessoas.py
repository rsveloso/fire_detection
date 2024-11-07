from ultralytics import YOLO
import cv2

# Carregar o modelo YOLO
modelo = YOLO('yolov8n.pt')

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Verificar se a webcam foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Definir o threshold de confiança e de NMS
confidence_threshold = 0.6  # Aumentar para reduzir o número de detecções
nms_threshold = 0.5  # Non-Maximum Suppression threshold

# Classes de interesse (exemplo: 'person', 'dog', 'cat')
classes_of_interest = ['person', 'dog', 'cat']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar quadro da webcam")
        break

    # Realizar a detecção de objetos
    results = modelo(frame, verbose=False)[0]

    # Aplicar filtragem por confiança e classes de interesse
    if results.boxes is not None:
        filtered_boxes = []
        for box in results.boxes:
            if box.conf.item() > confidence_threshold and results.names[box.cls.item()] in classes_of_interest:
                filtered_boxes.append(box)
        
        # Aplicar Non-Maximum Suppression
        nms_boxes = cv2.dnn.NMSBoxes(
            [box.xyxy[0].tolist() for box in filtered_boxes],
            [box.conf.item() for box in filtered_boxes],
            confidence_threshold,
            nms_threshold
        )

        # Manter apenas as caixas após NMS
        final_boxes = [filtered_boxes[i] for i in nms_boxes]
        results.boxes = final_boxes

    # Plotar apenas as detecções filtradas
    frame_with_detections = results.plot()

    # Exibir o quadro com as detecções
    cv2.imshow('Detecções', frame_with_detections)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas abertas
cap.release()
cv2.destroyAllWindows()
