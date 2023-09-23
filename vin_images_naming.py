import os
import pickle

INPUT_PATH = "./niss_vin_imgs_4_vin_crops_renamed/"
DATA_PATH = "./nissan_vin_gt.pkl"

images_names = [i for i in os.listdir(INPUT_PATH)]

with open(DATA_PATH, "rb") as f:
    gt_data = pickle.load(f)

cnt=0
for img in images_names:
    # print(img)
    img_ = img.split("_")
    image_name = "_".join(img_[:len(img_)-1])+"."+img_[-1].split(".")[1]
    # print(image_name)
    if(image_name in gt_data):
        print(image_name)
        print(gt_data[image_name])
        if(os.path.exists(INPUT_PATH+gt_data[image_name][0]+".jpg") == False):
            os.rename(INPUT_PATH+img,INPUT_PATH+gt_data[image_name][0]+".jpg")
        else:
            os.rename(INPUT_PATH+img,INPUT_PATH+gt_data[image_name][0]+"_{}.jpg".format(cnt))
            cnt+=1
