from pycocotools.coco import COCO

FILEDIR = './labels/old_train2017'

coco = COCO(annotation_file='annotations/instances_train2017.json')

imgIds = coco.getImgIds()


for current_img in imgIds:
    print(str(current_img).zfill(12))
    with open(f'{FILEDIR}/{str(current_img).zfill(12)}.txt','w') as f:
        for current_anno in coco.loadAnns(coco.getAnnIds(imgIds=current_img)):
            catId = current_anno['category_id']
            bbox = current_anno['bbox']
            bbox_str = ' '.join(str(coord) for coord in bbox)
            # category_id와 bbox 문자열을 결합하여 파일에 쓰기
            f.write(str(catId) + ' ' + bbox_str + '\n')
