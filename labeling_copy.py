from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import math

IMGID = 472
ANNOTATION = './annotations/annotations/instances_train2017.json'
coco = COCO(annotation_file=ANNOTATION)
annotation = coco.loadAnns(coco.getAnnIds(coco.getImgIds(IMGID)))
bbox_origin = []
bbox_store = []
bbox = []
img = cv2.imread(f'./images/train2017/{str(IMGID).zfill(12)}.jpg')
h, w, ch = img.shape

with open(f'../coco128/labels/train2017/{str(IMGID).zfill(12)}.txt', 'r') as f:
    bbox_store.append(f.read())
for b in bbox_store:
    b = str(b).split(' ')[1:]
    b = [bi.strip('\n') for bi in b]
    bbox.append(b)
print('bbox: ',bbox)    
for idx,ann in enumerate(annotation):
    bbox_origin.append(ann['bbox'])
    print(f'anno{idx}: ',ann['bbox'],ann['category_id'])
result = [round(float(s)/w,6) if i%2==0 else round(float(s)/h,6) for i, s in enumerate(ann['bbox'])]
print('result: ',result)

scale_factor = 640 / max(h, w)
if scale_factor < 1:
    h = math.ceil(scale_factor * h)
    w = math.ceil(scale_factor * w)
print('shape: ',img.shape)
img = cv2.resize(img, (w, h))
mosaic = np.full((int(h), int(w), 3), 255,dtype=np.uint8)
img.transpose(1,2,0)




updated_rect = []
for i, box in enumerate(bbox):
    # 좌표 계산
    x = int(float(box[0]) * w)  # Convert to integer
    y = int(float(box[1]) * h)  # Convert to integer
    w = int(float(box[2]) * w)  # Convert to integer
    h = int(float(box[3]) * h)  # Convert to integer
    # 업데이트된 좌표로 새 튜플 생성
    rect = (x, y, w, h) 
    # bbox 리스트 내의 해당 박스를 새 튜플로 업데이트
    updated_rect.append(rect)
    
for rect in updated_rect:
    cv2.rectangle(img,rect,(255,0,0))
cv2.imwrite('sample.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2

## 이미지 로딩 성공 여부 확인
#if img is None:
#    print("이미지를 로드하지 못했습니다. 파일 경로를 확인하세요.")
#else:
#    # 이미지에 선 그리기
#    cv2.rectangle(img,(1,187),(612,98),(255,0,0))
#    cv2.rectangle(img,(311,4),(631,224),(255,0,0))
#    cv2.rectangle(img,(249,229),(565,15),(0,255,0))
#    cv2.rectangle(img,(0,13),(434,361),(0,255,0))
#    cv2.rectangle(img,(376,40),(451,6),(0,0,255))
#    cv2.rectangle(img,(465,38),(523,7),(0,0,255))
#    cv2.rectangle(img,(364,2),(458,68),(0,0,255))
#

#for ann in coco.loadAnns(coco.getAnnIds(coco.getImgIds(imgIds=9))):
#    print(ann)

"""
for current_img in imgIds:
    print(str(current_img).zfill(12))
    with open(f'{FILEDIR}/{str(current_img).zfill(12)}.txt','w') as f:
        for current_anno in coco.getAnnIds(imgIds=current_img):
            print(current_anno)
            print(coco.loadAnns(current_anno))
            input()
            f.write()
"""