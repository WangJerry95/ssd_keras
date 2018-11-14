import os
import cv2
import re

'''
convert small patches label to large image
label_pth : small patches predict results .txt  (img_name xmin ymin xmax ymax class confidence xmin ymin ...)
img_dir : large image root (include 50 images)
save_pth : large image label save path .txt (img_name xmin ymin xmax ymax class confidence xmin ymin ...)
'''
label_pth = r"/home/jerry/Projects/ssd_keras/results/ssd7/2.5/gray/results.txt"
img_dir = r"/home/jerry/Datasets/sea_ship/new_train_1109/Test/GSD=2.5"
save_pth = r"/home/jerry/Projects/ssd_keras/results/ssd7/2.5/gray/large_results.txt"

def compute_iou(box1,box2):
    x0 = max(int(box1[0]),int(box2[0]))
    y0 = max(int(box1[1]),int(box2[1]))
    x1 = min(int(box1[2]),int(box2[2]))
    y1 = min(int(box1[3]),int(box2[3]))

    if (x1-x0) >=0 and (y1-y0) >=0:
        return True
    else: 
        return False

def large_label(save_name):
    content_txt = []
    with open(label_pth,"r") as f:
        for line in f.readlines():
            splited = line.strip().split("\t")
            img_name = splited[0]
            # split_name = re.split("_",img_name)
            start_Y = int(float(img_name.split("|")[1].split("_")[0]))
            start_X = int(float(img_name.split("|")[1].split("_")[1]))
            # '''
            # debug
            # '''
            # start_Y = int(float(img_name.split("_")[5]))
            # start_X = int(float(img_name.split("_")[6].split(".")[0]))

            # large_name = str()
            # for i in range(4):
            #     large_name += split_name[i] +"_"
            # large_name += split_name[4]  
            large_name = img_name.split("|")[0]

            if large_name == save_name:
                object_num = (len(splited)-1)//6
                for i in range(object_num):
                    xmin = str(start_X + round(float(splited[6*i+1])))
                    ymin = str(start_Y + round(float(splited[6*i+2])))
                    xmax = str(start_X + round(float(splited[6*i+3])))
                    ymax = str(start_Y + round(float(splited[6*i+4])))
                    c = splited[6*i+5]
                    conf = str(splited[6*i+6])
                    content_txt.append([xmin,ymin,xmax,ymax,c,conf])

    '''
    merge overlapping boxes select the biggest as class information 
    '''
    tmp_result = []
    for i in range(len(content_txt)):
        objects = [content_txt[i]]
        for j in range(len(content_txt)):
            if compute_iou(content_txt[i],content_txt[j]):
                objects.append(content_txt[j])

        xmin = content_txt[i][0]
        ymin = content_txt[i][1]
        xmax = content_txt[i][2]
        ymax = content_txt[i][3]
        c = content_txt[i][4]
        conf = content_txt[i][5]
        area = (int(content_txt[i][2])-int(content_txt[i][0]))*(int(content_txt[i][3])-int(content_txt[i][1]))
        for j in range(len(objects)):
            xmin = objects[j][0] if objects[j][0] < xmin else xmin
            ymin = objects[j][1] if objects[j][1] < ymin else ymin
            xmax = objects[j][2] if objects[j][2] > xmax else xmax
            ymax = objects[j][3] if objects[j][3] > ymax else ymax
            c = content_txt[j][4] if (int(content_txt[j][2])-int(content_txt[j][0]))*(int(content_txt[j][3])-int(content_txt[j][1])) > area else c
            conf = content_txt[j][5] if (int(content_txt[j][2])-int(content_txt[j][0]))*(int(content_txt[j][3])-int(content_txt[j][1])) > area else conf
        tmp_result.append([xmin,ymin,xmax,ymax,c,conf])
    
    '''
    remove repeated boxes
    '''
    result = []
    for i in range(len(tmp_result)):
        tag = True
        for j in range(len(result)):
            if (tmp_result[i][0] == result[j][0]) and (tmp_result[i][1] == result[j][1]) and (tmp_result[i][2] == result[j][2]) and (tmp_result[i][3] == result[j][3]):
                tag =False
        if tag:
            result.append(tmp_result[i])
    

    with open(save_pth,"a+") as fn:
        save_name = save_name + ".jpg"
        fn.write(save_name)
        fn.write("\t")
        '''
        debug
        '''
        for i in range(len(result)):
            for j in range(6):
                fn.write(result[i][j])
                fn.write("\t")
        fn.write("\n")



def test():
    for img_name in os.listdir(img_dir):
        save_name = img_name.split(".")[0]
        large_label(save_name)

test()