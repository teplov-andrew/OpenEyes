import telebot
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from IPython.display import display
import os
import time
import glob
from itertools import product

files = glob.glob('photos/*')
for f in files:
    os.remove(f)

client = telebot.TeleBot("5552391553:AAFowoNUMfbxhqkFuFYE-jypOxZidPGGNeQ") # token
my_img = []
i = 0
params = []
cl_data = {}


@client.message_handler(func=lambda c: c.text == '/start')
def start(message):
    mess = f'Привет, для того, чтобы открыть глаза, пришли мне две фотографии:\n1-ая: глаза открыты\n2-ая: глаза закрыты'
    client.send_message(message.chat.id, mess, parse_mode='html')
    cl_data[message.chat.id] = []


@client.message_handler(content_types=['photo'])
def photo(message):
    chat_id = message.chat.id
    fileID = message.photo[-1].file_id
    file_info = client.get_file(fileID)
    downloaded_file = client.download_file(file_info.file_path)

    with open(f"{file_info.file_path}", 'wb') as new_file:
        my_img.append(str(file_info.file_path))
        new_file.write(downloaded_file)
    if len(my_img) == 2:
        img1, dict_bbox, dict_bbox_2, img_open, img_closed = do(my_img, chat_id, message)
        params.clear()
        params.append(img1)
        params.append(dict_bbox)
        params.append(dict_bbox_2)
        params.append(img_open)
        params.append(img_closed)
        params.append(my_img)


@client.message_handler(content_types=["text"])
def part2(message):
    chat_id = message.chat.id
    if message.text != -1:
        i = message.text
        print(i)
        do2(chat_id, int(i), params[0], params[1], params[2], params[3], params[4], params[5])
        my_img = []


def do(my_img, chat_id, message):
    print("зашел")
    print(my_img)
    cl_data[message.chat.id] = my_img
    print(cl_data)
    CKPT_PATH = 'model/eyes_model.pt' # путь к весам
    yolov5 = torch.hub.load('yolov5', 'custom', path=CKPT_PATH, source='local', force_reload=True) # загрузка модели

    img_open = cl_data[message.chat.id][0]
    img_closed = cl_data[message.chat.id][1]

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

    # display(img)

    client.send_photo(chat_id, img, caption="Посмотри на фото и отправь номер нужного человека")
    return img1, dict_bbox, dict_bbox_2, img_open, img_closed


def do2(chat_id, i, img1, dict_bbox, dict_bbox_2, img_open, img_closed, my_img):
    print("зашел2")
    img = img1
    box = dict_bbox[i]
    our_space = box
    box = torch.tensor(box)
    box = box.unsqueeze(0)

    img = draw_bounding_boxes(img, box, width=2,
                              colors=["blue"], labels=str(i),
                              fill=False, font_size=20)

    img = torchvision.transforms.ToPILImage()(img)
    client.send_photo(chat_id, img)
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
        # print("b ", b)
        x = b[0] - a[0]
        y = b[1] - a[1]

    print(x, y)
    img = Image.open(img_open)
    cut = img.crop((our_space[0], our_space[1], our_space[2], our_space[3]))

    img_test = Image.open(img_closed)
    # display(img_test)
    # img_test.paste(cut,(int(our_space[0]), int(our_space[1]) , int(our_space[2]), int(our_space[3]) ))
    img_test.paste(cut, (int(our_space[0]) + int(x), int(our_space[1]) + int(y)))
    # display(img_test)

    client.send_photo(chat_id, img_test, caption="Готово!")
    print(os.listdir("photos"))
    for f in os.listdir("photos"):
        os.remove("photos/" + f)
        print(1)
    my_img.clear()


print('start')
client.polling(none_stop=True)
