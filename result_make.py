import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--testdir", type=str)
opt = parser.parse_args()

is_dir = os.path.isdir(opt.testdir)
epoch, gpu_mem, box, obj, cls, total, labels, img_size, Class, Images, Labels, Precision, Recall, mAP50, mAP5095 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
data_list = [epoch, gpu_mem, box, obj, cls, total, labels, img_size, Class, Images, Labels, Precision, Recall, mAP50, mAP5095]
if is_dir:
    
    file = opt.testdir + "results.txt"
    with open(file, 'r') as f:
        data = f.readlines()
    for i, d in enumerate(data):
        d = d.strip('\n').split()
        for idx, current_data in enumerate(data_list):
            if current_data == epoch:
                d[idx] = d[idx].split('/')[0]
            current_data.append(d[idx])
        
    df = pd.DataFrame({'Epoch': epoch,
                       'gpu_mem': gpu_mem,
                       'box': box,
                       'obj': obj,
                       'cls': cls,
                       'total': total,
                       'labels': labels,
                       'img_size': img_size,
                       'Class': Class,
                       'Images': Images,
                       'Precision': Precision,
                       'Recall': Recall,
                       'mAP50': mAP50,
                       'mAP5095': mAP5095
                       })
    filename = ''.join(opt.testdir[:])+f"{opt.testdir.split('/')[-2]}_result.xlsx"
    df.to_excel(filename)
        
    