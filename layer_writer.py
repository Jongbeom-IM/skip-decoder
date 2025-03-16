import yaml
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str)
opt = parser.parse_args()

FILE_DIR = os.path.abspath(opt.cfg)
print(FILE_DIR)
new_line = []
i= 0
try:
    with open(FILE_DIR, 'r') as f:
        lines = f.readlines()
        if not lines:
            print("파일이 비어있습니다.")
        else:
            for idx, line in enumerate(lines):
                if idx > 14:
                    if line == '\n' or line == '  ]\n' or 'head' in line: 
                        new_line.append(line)
                    else:   
                        data = line[:56] + f"# {i}\n"
                        new_line.append(data)
                        i = i+1
                    if idx == 133: print([line])
except:
    None

with open('./new.yaml', 'w') as f:
    f.writelines(new_line)