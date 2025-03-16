import os
import torch
import h5py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = self._get_image_files()
        self.labels = self._get_labels() if label_dir else None

    def _get_image_files(self):
        image_files = []
        for dirpath, _, filenames in os.walk(self.image_dir):
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(dirpath, filename))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        img_id = os.path.splitext(os.path.basename(img_path))[0]

        if self.transform:
            image = self.transform(image)

        return image, img_id

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_image_dir = '/home/syslab-4090/jongbeom/Dataset/coco/images/train2017'
# val_image_dir = '/home/syslab-4090/jongbeom/Dataset/coco/images/val2017'

train_dataset = COCODataset(image_dir=train_image_dir, transform=transform)
# val_dataset = COCODataset(image_dir=val_image_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# Save tensor outputs to HDF5 files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.experimental import attempt_load
model = attempt_load('./runs/train/e6e640/weights/best.pt', map_location=device)
# tensor_idx = [17, 13, 9]#, 36, 29, 22, 53, 48, 43]
input_idx = [9]#, 43, 58]
model.save = model.save + input_idx#(tensor_idx + input_idx)
model.save = list(set(model.save))
train_path = '/home/syslab-4090/jongbeom/Dataset/coco/images/yolov7/train'
os.makedirs(train_path, exist_ok=True)
for img, path in tqdm(train_loader):
    img = img.to(device)
    y = model(img)
    # h5_file_path = os.path.join(train_path, 'x', f'{os.path.basename(path[0])}.h5')
    h5_file_input = os.path.join(train_path, f'{os.path.basename(path[0])}.h5')
    # with h5py.File(h5_file_path, 'w') as f:
    #     for idx in tensor_idx:
    #         f.create_dataset(str(idx), data=y[idx].to('cpu').numpy())
    with h5py.File(h5_file_input, 'w') as f:
        for idx in input_idx:
            f.create_dataset(str(idx), data=y[idx].to('cpu').numpy())

# for img, path in tqdm(val_loader):
#     img = img.to(device)
#     y = model(img)
#     h5_file_path = os.path.join(output_path, 'val2017', f'{os.path.basename(path[0])}.h5')
#     h5_file_input = os.path.join(input_path, 'val2017', f'{os.path.basename(path[0])}.h5')
#     with h5py.File(h5_file_path, 'w') as f:
#         for y_, idx in zip(y[:3], tensor_idx):
#             f.create_dataset(str(idx), data=y_.to('cpu').numpy())
#     with h5py.File(h5_file_input, 'w') as f:
#         f.create_dataset('input', data=y[3].to('cpu').numpy())