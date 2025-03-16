import os
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm
IMGDIR = './images/'
LABELDIR = './labels/'
TARGET = 'val2017'
#coco = COCO(annotation_file='annotations/annotations/instances_'+TARGET+'.json')
#img_size = coco.loadImgs(ids=9)
#print(img_size)

img = [file for file in os.listdir(IMGDIR+TARGET) if file.endswith('jpg')]
label = [file for file in os.listdir(LABELDIR+'old_'+TARGET) if file.endswith('txt')]
image_dict = {f.split('.jpg')[0]: os.path.join(f) for f in img}
label_dict = {f.split('.txt')[0]: os.path.join(f) for f in label}
dataset = []
for name, paths in image_dict.items():
    label_path = label_dict.get(name)
    if label_path:  # 해당 이름의 라벨 파일이 있다면
        dataset.append((paths, label_path))
        
for (img,label) in tqdm(dataset):
    with open(LABELDIR+'old_'+TARGET+'/'+label,'r') as f:
        img = cv2.imread(IMGDIR+TARGET+'/'+img)
        h, w, c = img.shape
        lines = f.readlines()
        lines = [s.replace('\n', '').split(' ') for s in lines]
        line_copy = lines
        for idx, line in enumerate(line_copy):
            lines[idx][1] = str(round(((float(line[1])) + (float(line[3]))/2) / w, 6))
            lines[idx][2] = str(round(((float(line[2])) + (float(line[4]))/2) / h, 6))
            lines[idx][3] = str(round((float(line[3]) / w),6))
            lines[idx][4] = str(round((float(line[4]) / h),6)) + '\n'
    with open(LABELDIR+TARGET+'/'+label,'w') as f:
        result = ''.join([' '.join(item) for item in lines])
        f.write(result)
            
            
            
        



