import json

ANNO_DIR = 'annotations/annotations/instances_train2017.json'
with open(ANNO_DIR, 'r') as f:
    data = json.load(f)


print(type(data['images']))

for image in data['images']:
    if '00000081.jpg' in image['file_name']:
        print(image)

"""
for idx, categ in enumerate(Cate):
    for dat in data['annotations']:
        if dat['category_id'] == categ['id']:
            dat['category_id'] = idx

for idx, category in enumerate(data['categories']):
    category['id'] = idx

with open(ANNO_DIR, 'w') as f:
    json.dump(data, f, indent=4)
"""    

