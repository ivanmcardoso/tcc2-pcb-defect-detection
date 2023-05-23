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
import albumentations as A
import shutil


def addBbox(image, bboxes):
    for box in bboxes:
        x, y, w, h, _ = box
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 1)


base_dir = os.getcwd()
model_path = f"{base_dir}/agu-best-v5.pt"
model = YOLO(model_path)
img_base_dir = f"{base_dir}/training/images"
label_base_dir = f"{base_dir}/training/labels"
imgs = os.listdir(img_base_dir)
file_name = imgs[random.randint(1, 692)]
src_path = f"{img_base_dir}/{file_name}"
label_src_path = f"{label_base_dir}/{file_name}"
label_src_path = label_src_path.replace(label_src_path[len(label_src_path) - 3:], "txt")
print(src_path)
print(label_src_path)
img = cv2.imread(src_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,640))

dh, dw, _ = img.shape
fl = open(label_src_path, 'r')
data = fl.readlines()
fl.close()
bbox = []

for dt in data:

    # Split string to float
    label, x, y, w, h = map(float, dt.split(' '))
    bbox.append([x, y, w, h, label])
imger_cpoy = img

transform = A.Compose([
    A.ColorJitter(hue=0.5, always_apply=True),
    A.CLAHE()
])
# transformed = transform(image=img, bboxes=bbox)
transformed = transform(image=img)
augmented_image = transformed['image']

results = model.predict(augmented_image)
for r in results:

    annotator = Annotator(augmented_image)

    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])

image = annotator.result()


# augmented_bbox = transformed['bboxes']
filename = 'savedImage.jpg'
filename1 = 'savedImage1.jpg'


# addBbox(augmented_image, augmented_bbox)
cv2.imwrite(file_name, image)
addBbox(imger_cpoy, bbox)
cv2.imwrite(f"{file_name}-pure", imger_cpoy)

# cv2.imwrite(filename1, augmented_image)
# shutil.copy(label_src_path, f"{base_dir}/agu-{file_name[0:-4]}.txt")
