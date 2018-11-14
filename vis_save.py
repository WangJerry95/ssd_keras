import os
import cv2
import numpy as np

label_pth = r"/home/jerry/Projects/ssd_keras/results/ssd7/2.5/gray/large_results.txt"
img_dir = r"/home/jerry/Datasets/sea_ship/new_train_1109/Test/GSD=2.5"
save_dir = r"/home/jerry/Projects/ssd_keras/results/ssd7/2.5/gray/"

with open(label_pth,"r") as f:
    for line in f.readlines():
        splited = line.strip().split("\t")
        img_name = splited[0]
        img_pth = os.path.join(img_dir, img_name)
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, ]*3, -1)
        save_pth = os.path.join(save_dir,img_name)
        for i in range(int((len(splited)-1)/6)):
            xmin = int(splited[6*i+1])
            ymin = int(splited[6*i+2])
            xmax = int(splited[6*i+3])
            ymax = int(splited[6*i+4])
            label = splited[6*i+5]
            # print(xmin,ymin,xmax,ymax,label)

            if label == "1.0":
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color = [0,255,0],thickness = 2)
            elif label == "2.0":
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color = [255,0,0],thickness = 2)
            elif label == "3.0":
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color = [0,0,255],thickness = 2)
        
        cv2.imwrite(save_pth, img)
