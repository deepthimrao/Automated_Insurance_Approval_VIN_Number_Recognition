import cv2
import numpy as np
import os

INPUT_PATH = "./niss_vin_imgs_4/" #path to original images
BBOX_PATH = "./niss_vin_imgs_4_text_detection_results/" #path to the result folder obtained after CRAFT text detection
OUTPUT_PATH = "./niss_vin_imgs_4_cropped_final/" #path to save cropped images

images_names = [i for i in os.listdir(INPUT_PATH) if i.endswith(".jpg") or i.endswith(".png")]
bboxes_results = [i for i in os.listdir(BBOX_PATH) if i.endswith(".txt")]

box_cnt=0
for img in images_names:
    image_path_name = INPUT_PATH+img
    image_raw = cv2.imread(image_path_name)
    if(img.endswith(".jpg")):
        actual_image_name = img.split(".jpg")[0]
        bbox_result_file = "res_"+img.split(".jpg")[0]+".txt"
    else:
        actual_image_name = img.split(".png")[0]
        bbox_result_file = "res_"+img.split(".png")[0]+".txt"

    if(bbox_result_file in bboxes_results):
        f=open(BBOX_PATH+bbox_result_file,"r")
        data=f.readlines()
        print(img)
        # print(data)
        for line in data:
            if(line!="\n"):
                l=list(map(int,line.replace("\n","").split(",")))
                if(len(l) % 8 == 0):
                    l=np.array(l).reshape([len(l)//8,8])
                    for pts in l:
                        p=pts.reshape((4,2))
                        # print(p)
                        x,y,w,h = box = cv2.boundingRect(p)
                        image_cropped=image_raw[y:y+h, x:x+w]
                        image_cropped_name = actual_image_name+"_{}.jpg".format(box_cnt)
                        box_cnt+=1
                        cv2.imwrite(OUTPUT_PATH+image_cropped_name, image_cropped)
                else:
                    l=np.array(l).reshape([len(l)//2,2])
                    x,y,w,h = box = cv2.boundingRect(l)
                    image_cropped=image_raw[y:y+h, x:x+w]
                    image_cropped_name = actual_image_name+"_{}.jpg".format(box_cnt)
                    box_cnt+=1
                    cv2.imwrite(OUTPUT_PATH+image_cropped_name, image_cropped)

                