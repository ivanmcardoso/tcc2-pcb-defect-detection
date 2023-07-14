import cv2
import os
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

base_dir = os.getcwd()
model_path = f"{base_dir}/ultralitics_models/V3.pt"
model = YOLO(model_path)


def predict_image(img):
    results = model.predict(img, conf=0.4, iou=0.3)

    for r in results:
        print(r.speed)
        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    return annotator.result()


# define a video capture object
vid = cv2.VideoCapture(0)



# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
