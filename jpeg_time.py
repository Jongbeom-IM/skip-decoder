import os
import time
import cv2

img_list = os.listdir('data/coco/images/val2017')
print(len(img_list))
times = 0.
i=0
for imgname in img_list:
    img = cv2.imread('data/coco/images/val2017/'+imgname)
    t = time.time()
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg=cv2.imencode('.jpg', img, encode_param)    
    times += time.time()-t
    i+=1
    if i%1000==0:
        print(i)
print(times/len(img_list)*1000)
