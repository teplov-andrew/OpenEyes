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

files = glob.glob('photos/*')
for f in files:
    os.remove(f)

client = telebot.TeleBot("5552391553:AAFowoNUMfbxhqkFuFYE-jypOxZidPGGNeQ")
my_img = []
i = 0
params = []
@client.message_handler(func=lambda c: c.text =='/start')
def start(message):
	mess = f'Привет, для того, чтобы открыть глаза, пришли мне две фотографии:\n1-ая: глаза открыты\n2-ая: глаза закрыты'
	client.send_message(message.chat.id, mess, parse_mode='html')

@client.message_handler(content_types=['photo'])
def photo(message):
	chat_id = message.chat.id
	print('message.photo =', message.photo)
	fileID = message.photo[-1].file_id
	print('fileID =', fileID)
	file_info = client.get_file(fileID)
	print('file.file_path =', file_info.file_path)
	# print(file_info)
	downloaded_file = client.download_file(file_info.file_path)


	with open(f"{file_info.file_path}", 'wb') as new_file:
		my_img.append(str(file_info.file_path))
		new_file.write(downloaded_file)
	print(my_img)
	if len(my_img)==2:
		img1,dict_bbox, dict_bbox_2, img_open,img_closed = do(my_img,chat_id,message)
		params.clear()
		params.append(img1)
		params.append(dict_bbox)
		params.append(dict_bbox_2)
		params.append(img_open)
		params.append(img_closed)
		params.append(my_img)
		# time.sleep(10)
		# if message.text!=-1:
		# 	i = message.text
		# 	do2(my_img,chat_id,i,img1, dict_bbox, dict_bbox_2)
		# return img1, dict_bbox, dict_bbox_2


@client.message_handler(content_types=["text"])
def part2(message):
	chat_id = message.chat.id
	if message.text != -1:
		i = message.text
		print(i)
		do2(chat_id, int(i), params[0], params[1], params[2],params[3],params[4], params[5])
		my_img=[]



def do(my_img,chat_id,message):
	print("зашел")
	print(my_img)
	CKPT_PATH = 'model/eyes_model.pt'
	yolov5 = torch.hub.load('yolov5','custom',path=CKPT_PATH,source='local',force_reload=True)

	img_open = my_img[0]
	img_closed = my_img[1]

	predict_image = yolov5([img_open])
	predict_image = predict_image.xyxy[0].data.cpu().numpy().tolist()

	for i in range(len(predict_image)):
		predict_image[i] = predict_image[i][:4]

	dict_bbox = {}
	for i in range(len(predict_image)):
		dict_bbox[i] = predict_image[i]
	# ================================================================================

	# closed image

	predict_image2 = yolov5([img_closed])
	predict_image2 = predict_image2.xyxy[0].data.cpu().numpy().tolist()

	for i in range(len(predict_image2)):
		predict_image2[i] = predict_image2[i][:4]

	dict_bbox_2 = {}
	predict_image2.reverse()
	for i in range(len(predict_image2)):
		dict_bbox_2[i] = predict_image2[i]

	# ================================================================================
	# print(dict_bbox)
	# print(dict_bbox_2)


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

	client.send_photo(chat_id, img, caption = "Выбери человека")
	# print("Какие глаза хочешь открыть?")
	# i = int(input())
	return img1, dict_bbox, dict_bbox_2,img_open,img_closed




def do2(chat_id,i,img1, dict_bbox, dict_bbox_2,img_open,img_closed, my_img):
	print("зашел2")
	img = img1
	box = dict_bbox[i]
	our_space = box
	# print("our_space: ",our_space)
	# print("anouther_box: ",dict_bbox[1])
	box = torch.tensor(box)
	box = box.unsqueeze(0)

	img = draw_bounding_boxes(img, box, width=2,
	                          colors=["blue"], labels=str(i),
	                          fill=False, font_size=20)

	img = torchvision.transforms.ToPILImage()(img)

	# display(img)
	client.send_photo(chat_id, img)
	# =================================================================================
	try:
		a = dict_bbox[i + 1]
		b = dict_bbox_2[i + 1]
		x = b[0] - a[0]
		y = b[1] - a[1]
	except:
		x = 0
		y = 0
	# print(x, y)
	# =================================================================================
	img = Image.open(img_open)
	cut = img.crop((our_space[0], our_space[1], our_space[2], our_space[3]))

	img_test = Image.open(img_closed)
	# display(img_test)
	# img_test.paste(cut,(int(our_space[0]), int(our_space[1]) , int(our_space[2]), int(our_space[3]) ))
	img_test.paste(cut, (int(our_space[0]) + int(x), int(our_space[1]) + int(y)))
	# display(img_test)

	client.send_photo(chat_id, img_test)
	print(os.listdir("photos"))
	for f in os.listdir("photos"):
		os.remove("photos/"+f)
		print(1)
	my_img.clear()




print('start')
client.polling(none_stop=True)

