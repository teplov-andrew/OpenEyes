# !git clone https://github.com/ultralytics/yolov5

import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from IPython.display import display
from itertools import product

CKPT_PATH = 'model/eyes_model.pt'  # путь к весам
yolov5 = torch.hub.load('yolov5', 'custom', path=CKPT_PATH, source='local', force_reload=True)  # загрузка модели

print("start")

img_open = "data_for_test/test_2.1.jpg"  # путь к изображению с открытыми глазами нужного человека
img_closed = "data_for_test/test_2.2.jpg"  # путь к изображению с закрытыми глазами нужного человека

predict_image = yolov5([img_open])
predict_image = predict_image.xyxy[0].data.cpu().numpy().tolist()

for i in range(len(predict_image)):
    predict_image[i] = predict_image[i][:4]

dict_bbox = {}
for i in range(len(predict_image)):
    dict_bbox[i] = predict_image[i]

predict_image2 = yolov5([img_closed])
predict_image2 = predict_image2.xyxy[0].data.cpu().numpy().tolist()

for i in range(len(predict_image2)):
    predict_image2[i] = predict_image2[i][:4]

dict_bbox_2 = {}
predict_image2.reverse()
for i in range(len(predict_image2)):
    dict_bbox_2[i] = predict_image2[i]

img1 = read_image(img_open)
img = img1

box = predict_image

labels = [str(i) for i in list(dict_bbox.keys())]
box = torch.tensor(box, dtype=torch.int)
colors = ["blue"] * len(labels)
img = draw_bounding_boxes(img, box, width=2,
                          colors=colors,
                          labels=labels,
                          fill=False, font_size=20)

img = torchvision.transforms.ToPILImage()(img)

display(img)

print("Какие глаза хочешь открыть?")
i = int(input())
img = img1

box = dict_bbox[i]
our_space = box
box = torch.tensor(box)
box = box.unsqueeze(0)

img = draw_bounding_boxes(img, box, width=2,
                          colors=["blue"], labels=str(i),
                          fill=False, font_size=20)

img = torchvision.transforms.ToPILImage()(img)

display(img)
#                                                           АЛГОРИТМ ВЫРАВНИВАНИЯ
# =================================================================================
dict_bbox_2_2 = {}
for c in range(len(dict_bbox_2)):
    dict_bbox_2_2[c] = dict_bbox_2[c][0:2]
# print(dict_bbox_2_2)
ch_lst = list(set().union(*dict_bbox_2_2.values()))
if max(list(dict_bbox.keys())) == i:
    arr = sorted(product(tuple(dict_bbox[i - 1][0:2]), tuple(ch_lst)), key=lambda t: abs(t[0] - t[1]))[0]
else:
    arr = sorted(product(tuple(dict_bbox[i + 1][0:2]), tuple(ch_lst)), key=lambda t: abs(t[0] - t[1]))[0]
print(arr)
for x in range(len(dict_bbox_2)):
    if arr[1] in dict_bbox_2[x]:
        print("x: ", x)
        break

if max(list(dict_bbox.keys())) == i:
    a = dict_bbox[i - 1]
    b = dict_bbox_2[x]
    x = b[0] - a[0]
    y = b[1] - a[1]
else:
    a = dict_bbox[i + 1]
    b = dict_bbox_2[x]
    x = b[0] - a[0]
    y = b[1] - a[1]
# =================================================================================

img = Image.open(img_open)
cut = img.crop((our_space[0], our_space[1], our_space[2], our_space[3]))
img_test = Image.open(img_closed)
# display(img_test)
# img_test.paste(cut,(int(our_space[0]), int(our_space[1]) , int(our_space[2]), int(our_space[3]) ))
img_test.paste(cut, (int(our_space[0]) + int(x), int(our_space[1]) + int(y)))
display(img_test)

img_test.show()
