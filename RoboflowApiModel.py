from roboflow import Roboflow
import os
import datetime

rf = Roboflow(api_key="5wI00FJ867ByNoXtvQaR")
# project = rf.workspace().project("pcb_defect_expanded")
# model = project.version(2).model
project = rf.workspace().project("missing_hole_or_open_circuit")
model = project.version(1).model

base_dir = os.getcwd()
# file_name = "04-e1.jpg"
file_name = "01.jpg"
src_path = f"{base_dir}/test_imgs/{file_name}"

# visualize your prediction
a = datetime.datetime.now()
result = model.predict(src_path, confidence=40, overlap=30)
b = datetime.datetime.now()
delta = b - a
filename = f"{file_name}-roboflow.jpg"
result.save(filename)
print(f"found {len(result.predictions)} detections in {delta.microseconds/1000} ms")
# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
