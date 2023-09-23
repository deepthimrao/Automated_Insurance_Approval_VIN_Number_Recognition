import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image, ImageFilter

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

input_images_path = "./niss_vin_imgs_rgb_final/" #path to rgb input images
output_images_path = "./niss_vin_imgs_edge_preprocess_900_100/" #output path

width = 900
height = 100

images_names = [i for i in os.listdir(input_images_path) if i.endswith(".png") or i.endswith(".jpg")]

tw=0
th=0
edge_d = 1 #if edge detection has to be performed
for img_name in images_names:
	img_name_path = input_images_path+img_name

	image_raw = cv2.imread(img_name_path)
	
	h,w,_=image_raw.shape
	tw+=w
	th+=h

	# image_raw = cv2.fastNlMeansDenoising(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB),None,4,7,21)
	if(edge_d):
		image_raw = np.array(Image.fromarray(image_raw).convert("L").filter(ImageFilter.FIND_EDGES))
	# image_raw=cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
	# image_raw=cv2.Canny(image_raw, 30, 90)
	# cv2.imshow("test",image_raw)
	# cv2.waitKey(0)

	image_rsz,sx,sy = img_preproc(image_raw, height, width)
	# cv2.imshow("test1",image_rsz)
	# cv2.waitKey(0)
	output_img_name_path = output_images_path+img_name
	cv2.imwrite(output_img_name_path, image_rsz)

print(tw/len(images_names))
print(th/len(images_names))