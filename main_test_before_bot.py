!git clone https://github.com/ultralytics/yolov5 
    
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

CKPT_PATH = '/kaggle/input/eyes-model-v1/eyes_model.pt'
yolov5 = torch.hub.load('/kaggle/working/yolov5','custom',path=CKPT_PATH,source='local',force_reload=True)

img_open = input() # /kaggle/input/test-img-papa/photo_2022-12-10_18-13-56.jpg
img_closed = input() # /kaggle/input/test-img-papa/photo_2022-12-10_18-13-51.jpg

predict_image = yolov5([img_open])
predict_image = predict_image.xyxy[0].data.cpu().numpy().tolist()

for i in range(len(predict_image)):
    predict_image[i] = predict_image[i][:4]
    
dict_bbox = {}
for i in range(len(predict_image)):
    dict_bbox[i] = predict_image[i]

img1 = read_image(img_open)

img = img1
  
box = predict_image

labels = [str(i) for i in list(dict_bbox.keys())]
box = torch.tensor(box , dtype=torch.int)
colors = ["blue"] * len(labels)
img = draw_bounding_boxes(img, box, width=2, 
                          colors=colors,
                          labels = labels,
                          fill=False, font_size=20)
  
img = torchvision.transforms.ToPILImage()(img)

display(img)

i = int(input())
img = img1
  
box = dict_bbox[i]
our_space = box
# print(our_space)
box = torch.tensor(box)
box = box.unsqueeze(0)

img = draw_bounding_boxes(img, box, width=2, 
                          colors=["blue"],labels = str(i),
                          fill=False, font_size=20)
  
img = torchvision.transforms.ToPILImage()(img)

display(img)

img = Image.open(img_open)
cut = img.crop((our_space[0], our_space[1], our_space[2], our_space[3]))

# display(cut)

img_test = Image.open(img_closed)
# display(img_test)
img_test.paste(cut,(int(our_space[0]), int(our_space[1]) , int(our_space[2]), int(our_space[3]) ))
display(img_test)