from ultralytics import YOLO
import cv2
import torch
import supervision as sv
import numpy as np
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

def recognise(input_video_path: str, output_video_path: str):
    '''Берет видео по ссылке input_video_path, выделяет людей боксами и сохраняет видео в output_video_path'''


    # Создаём покадровый считыватель видео
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создаём объект для записи видео
    output_path = output_video_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Разбиваем кадр на сегменты со сдвином
        element_crops = MakeCropsDetectThem(
            image=frame,
            model_path="yolo11n.pt",
            segment=False,
            shape_x=640,
            shape_y=640,
            overlap_x=50,
            classes_list=[0],
            overlap_y=50,
            conf=0.5,
            iou=0.7,
        )
        # Находим на сегментах людей и обединяем их обратно в 1 кадр
        result = CombineDetections(element_crops, nms_threshold=0.25)

        #Отрисовываем людей
        confid = result.filtered_confidences
        box = result.filtered_boxes
        if len(box) != 0:
            for i in range(len(box)):
                param = box[i]
                confidence = confid[i]
                cv2.rectangle(frame, (param[0], param[1]), (param[2], param[3]), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, f"Person: {confidence:.{2}f}", (int(param[0]), int(param[1])-10), font, 1, (255, 0, 0), 2)

        out.write(frame)

        #На всякий случай выход из программы
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognise("crowd.mp4", "output.mp4")