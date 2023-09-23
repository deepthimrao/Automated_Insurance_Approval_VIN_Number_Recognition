from turtle import mode
import tensorflow as tf
import keras
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageFilter
from difflib import SequenceMatcher
import random
from random import shuffle

random.seed(12345)

def img_preproc(img, height, width):
	h1 = 0
	w1 = 0
	dim = img.shape
	h1=dim[0]
	w1=dim[1]
	orig_ar = w1/h1
	inp_ar = width/height
	if(width == height):
		OrigSize = 0
		if h1 > w1 :
			bottomBorder = 0
			rightBorder = h1 - w1
			OrigSize = h1
		else :
			rightBorder = 0
			bottomBorder = w1 - h1
			OrigSize = w1
		
		Scaling_x = float(OrigSize) / float(width)
		Scaling_y = float(OrigSize) / float(height)
		ImgWithBorder = cv2.copyMakeBorder(img, 0, bottomBorder, 0, rightBorder, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
		imsz1 = cv2.resize(ImgWithBorder, (width,height))
		return imsz1, Scaling_x, Scaling_y
	else:
		if(round(inp_ar,1) == round(orig_ar,1)):
			Scaling_x = float(w1)/float(width)
			Scaling_y = float(h1)/float(height)
			imsz1 = cv2.resize(img, (width,height))
			return imsz1, Scaling_x, Scaling_y
		else:
			if(round(orig_ar,1) < round(inp_ar,1)):
				rightBorder = int(h1*inp_ar) - w1
				bottomBorder = 0
			else:
				if(h1<w1):
					bottomBorder = int(w1/inp_ar) - h1
					rightBorder = 0
				else:
					rightBorder = int(h1/inp_ar) - w1
					bottomBorder = 0
			# print(bottomBorder, rightBorder)
			ImgWithBorder = cv2.copyMakeBorder(img, 0, bottomBorder, 0, rightBorder, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
			imsz1 = cv2.resize(ImgWithBorder, (width,height))
			Scaling_y = float(ImgWithBorder.shape[0])/float(height)
			Scaling_x = float(ImgWithBorder.shape[1])/float(width)
			return imsz1, Scaling_x, Scaling_y

def decode_batch_predictions(pred):
	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	# Use greedy search. For complex tasks, you can use beam search
	results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
		:, :17
	]
	# Iterate over the results and get back the text
	output_text = []
	for res in results:
		res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
		output_text.append(res)
	return output_text

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#type of model
# model_edge_cnn, model_edge_lstm, model_edge_lstm_synthetic
model_type = "model_edge_lstm_synthetic"
TEST_IMAGES_PATH = "./test_images/"

prediction_model = keras.models.load_model("./models/model_1_synth_32_500_50_edge_lstm") #path to model file
if("lstm" in model_type):
	prediction_model = keras.models.Model(prediction_model.get_layer(name="image").input, prediction_model.get_layer(name="dense2").output)
	prediction_model.load_weights("./models/ckpt_1_synth_32_500_50_edge_lstm/") #path to model checkpoint file

images_names =  [i for i in os.listdir(TEST_IMAGES_PATH) if i.endswith(".PNG") or i.endswith(".jpg") or i.endswith(".png")]

if("synthetic" not in model_type or "synthetic" not in TEST_IMAGES_PATH):
	labels = [l.split(".jpg")[0] for l in images_names]
else:
	labels = [l.split(".png")[0] for l in images_names]
	labels = [l.split("_")[0] for l in labels]
idx_shuf=list(range(len(images_names)))
shuffle(idx_shuf)
images_names = [images_names[i] for i in idx_shuf]
labels = [labels[i] for i in idx_shuf]

if("synthetic" in model_type):
	characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
if("lstm" in model_type):
	characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'M', 'O', 'Z']
if(model_type == "model_edge_cnn"):
	characters = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(
	vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
	vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

car_model = "nissan"

tot_sim_bef=0
tot_sim_aft=0

c=0
for img in images_names:
	# if(c==100):
	# 	break
	img_path = TEST_IMAGES_PATH+img
	
	if("rgb" in model_type):
		image_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	# cv2.imshow("test",image_raw)
	if("edge" in model_type):
		image_raw = np.array(Image.open(img_path).convert("L").filter(ImageFilter.FIND_EDGES))
	
	if("synthetic" in model_type):
		image_rsz,sx,sy = img_preproc(image_raw, 50, 500)
	else:
		image_rsz,sx,sy = img_preproc(image_raw, 100, 900)
	image_inp = image_rsz[...,np.newaxis]
	image_inp = tf.convert_to_tensor(image_inp, tf.float32)
	image_inp = tf.transpose(image_inp, perm=(1,0,2))
	image_inp = image_inp[np.newaxis,...]
	
	pred=prediction_model(image_inp)

	if("lstm" in model_type):
		output_text=decode_batch_predictions(pred)
		ot=output_text[0].replace("[UNK]","_")
	if("cnn" in model_type):
		output_text="".join([characters[np.argmax(a.numpy())-1] for a in pred])
		ot=output_text.replace("[UNK]","_")
	print(c,ot,labels[c],similar(ot,labels[c])) #before prior
	tot_sim_bef+=similar(ot,labels[c])
	
	if(car_model == "nissan"):
		ol=list(ot)
		ol[0:4]=['M','D','H','F']
		fo="".join(ol)
		print(c,fo,labels[c]) #after prior
		tot_sim_aft+=similar(fo,labels[c])
	c+=1

print(tot_sim_bef/len(images_names))
print(tot_sim_aft/len(images_names))

# print(tot_sim_bef/100)
# print(tot_sim_aft/100)