import yaml

with open('./models/sru_int_cus_yolov5s.yaml') as f:
    data_dict = yaml.safe_load(f)
print(data_dict)