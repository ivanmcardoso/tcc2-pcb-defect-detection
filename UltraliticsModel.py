from ultralytics import YOLO
import os
import random
import cv2
from ultralytics.yolo.utils.plotting import Annotator

base_dir = os.getcwd()
model_path = f"{base_dir}/ultralitics_models/V3.pt"
model = YOLO(model_path)

file_name = "04-e1.jpg"
# file_name = "01.jpg"
src_path = f"{base_dir}/test_imgs/{file_name}"
print(src_path)
img = cv2.imread(src_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.predict(img, conf=0.4, iou=0.3)

for r in results:
    print(r.speed)
    annotator = Annotator(img)

    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
#        annotator.box_label(b, model.names[int(c)])
        annotator.box_label(b)


image = annotator.result()
res_plotted = results[0].plot()

filename = f"{file_name}-ultralitcs"
cv2.imwrite(filename, image)

