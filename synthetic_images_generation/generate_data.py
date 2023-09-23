from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from PIL import ImageFilter
import re
import cv2


SEED=847685
np.random.seed(SEED)

def img_preproc(img, height, width):
	h1 = 0
	w1 = 0
	h1, w1 = img.shape
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



f=open("./test.txt","r") # give path to corpus text file
corpus=f.readlines()

corpus=[c.replace("\n","") for c in corpus]
corpus=[c for c in corpus if len(c)==17]
corpus=[c for c in corpus if bool(re.match('^[A-Z0-9]*$', c))]

corpus=np.array(corpus)
np.random.shuffle(corpus)

# text colors - number of images
#847685 - 10000
#817382 - 5000
#827586 - 5000
#1F2022 - 5000
#030303 - 5000
text_color = "#847685"
generator = GeneratorFromStrings(
    corpus,
    count=20,
    fonts=["./fonts/ProFontWindows.ttf"], # path to font, must be in ttf format
    skewing_angle=10,
    random_skew=True,
    fit=True,
    size=50,
    # blur=2,
    # random_blur=True
    background_type=3,
    image_dir="./synthetic_images_test/images_car_bg/", # path to background images
    text_color=text_color,
    character_spacing=1,
    # width=400
    
)


def emboss_image(image, is_img_arr=True):
    if(is_img_arr == False):
        image=np.array(image)
    aug = iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))
    embossed_img = aug(images=image)

    return embossed_img

def etch_image(image):
    imageEtch = image.filter(ImageFilter.EMBOSS)

    return imageEtch

def save_pil_image(image, path, is_img_arr=False):
    if(is_img_arr):
        image=Image.fromarray(image)
    image.save(path)  

OUTPUT_PATH = "./test_synthetic_images/"
OUTPUT_PATH_EDGES_PREPROCESS = "./final_synthetic_images_edges_preprocess_profont_800/"

count = 0
edge_preprocess = 0 # set to 1 if you want to preprocess and save the images.

for img,lbl in generator:
    save_pil_image(img, OUTPUT_PATH+lbl+"_{}.png".format(text_color))

    if(edge_preprocess):
        img_g=img.convert("L") #converts to grayscale
        img_e=img_g.filter(ImageFilter.FIND_EDGES)
        img_rsz,sx,sy=img_preproc(np.array(img_e),50,500)
        save_pil_image(img_rsz, OUTPUT_PATH_EDGES_PREPROCESS+lbl+"_{}.png".format(text_color), True)

   