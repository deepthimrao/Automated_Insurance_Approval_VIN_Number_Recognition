import os
import cv2
import shutil

INPUT_PATH = "./niss_vin_imgs_4_cropped_final/"
OUTPUT_PATH = "./niss_vin_imgs_4_vin_crops/"

images_names = [i for i in os.listdir(INPUT_PATH) if i.endswith(".jpg") or i.endswith(".png")]

tot_imgs = len(images_names)

img_idx=0
while(img_idx<tot_imgs):
    image_name = images_names[img_idx]
    try:
        image_raw = cv2.imread(INPUT_PATH+image_name)

        cv2.namedWindow("test",cv2.WINDOW_NORMAL)
        cv2.imshow("test", image_raw)
        k=cv2.waitKey(0)
        if(k==ord('v')):
            shutil.copy(INPUT_PATH+image_name, OUTPUT_PATH)
            img_idx+=1
        elif(k==ord('a')):
            if(img_idx-1>0):
                img_idx-=1
        else:
            img_idx+=1
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        img_idx+=1