# from pylabel import importer
# import os
# import shutil
#
# if not os.path.exists('Annotations'):
#     os.mkdir("Annotations")
# if not os.path.exists('images'):
#     os.mkdir("images")
#
#
# def group_annotations(m_origin, target):
#     files = os.listdir(m_origin)
#
#     # Fetching all the files to directory
#     for file_name in files:
#         shutil.copy(m_origin + '/' + file_name, target + '/' + file_name)
#     print("Files are copied successfully")
#
#
# def group_images(m_origin, target):
#     files = os.listdir(m_origin)
#     print(files)
#     # Fetching all the files to directory
#     # for file_name in files:
#     #     shutil.copy(m_origin + file_name, target + file_name)
#     # print("Files are copied successfully")
#
#
# # Providing the folder path
# origin_annotation_dir = '/home/ivan/Documents/projetos/PCB_DATASET/Annotations'
# origin_images_dir = '/home/ivan/Documents/projetos/PCB_DATASET/images'
# target_annotation_dir = './Annotations/'
# target_image_dir = './images/'
#
# annotation_dirs = os.listdir(origin_annotation_dir)
# for a_origin in annotation_dirs:
#     group_annotations(origin_annotation_dir + '/' + a_origin, target_annotation_dir)
#
# image_dirs = os.listdir(origin_images_dir)
# for i_origin in image_dirs:
#     group_annotations(origin_images_dir + '/' + i_origin, target_image_dir)
#
# # Fetching the list of all the files
#
# base_dir = os.getcwd()
# path_to_annotations = f'{base_dir}/Annotations'
# path_to_images = f'{base_dir}/images'
# dataset = importer.ImportVOC(path=path_to_annotations, path_to_images=path_to_images)
# dataset.export.ExportToYoloV5(copy_images=True)
# shutil.rmtree(path_to_annotations)
# shutil.rmtree(path_to_images)



from ultralytics import YOLO
import os
import random
import cv2
from ultralytics.yolo.utils.plotting import Annotator

base_dir = os.getcwd()
model_path = f"{base_dir}/best.pt"
model = YOLO(model_path)
img_base_dir = f"{base_dir}/training/images"
imgs = os.listdir(img_base_dir)
file_name = imgs[random.randint(1, 692)]
src_path = f"{img_base_dir}/{file_name}"
print(src_path)
img = cv2.imread(src_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,640))
print(img.shape)

results = model.predict(img)
for r in results:

    annotator = Annotator(img)

    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])

img = annotator.result()
cv2.imshow(file_name, img)
cv2.waitKey(0)
