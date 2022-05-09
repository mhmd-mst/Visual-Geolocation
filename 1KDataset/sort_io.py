# sort indoor outdoor images
from ast import arg
import os
import torch
import shutil
import argparse
from tqdm import tqdm
from glob import glob
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from google_drive_downloader import GoogleDriveDownloader as gdd
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, getpass
from os.path import join, getsize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_folder", type=int, default=70)
parser.add_argument("--min_lat", type=float, default=37.70)
parser.add_argument("--max_lat", type=float, default=37.82)
parser.add_argument("--min_lon", type=float, default=-122.53)
parser.add_argument("--max_lon", type=float, default=-122.35)
parser.add_argument("--num_workers", type=int, default=12, help="_")
parser.add_argument("--batch_size", type=int, default=16, help="_")
parser.add_argument("--output_folder", type=str, default="outputs",
                    help="Folder where to save logs and maps")

args = parser.parse_args()

# TODO: Change input folder > 1KDataset/
input_folder = (f"outputs/Unsorted/{args.input_folder}")
folder_counter = sum([len(folder) for _, _, folder in os.walk(input_folder)])

# args.output_folder = (f"{args.output_folder}/Sorted/{args.input_folder}")



# flickr_folder = f"{args.output_folder}/flickr"
# images_folder = f"{args.output_folder}/flickr/images"
# sorted_images_folder = f"{args.output_folder}/flickr/sorted_images"
sorted_images_folder = (f"{args.output_folder}/Sorted/{args.input_folder}")

os.makedirs("1KDataset/data", exist_ok=True)
os.makedirs(f"{sorted_images_folder}/panos", exist_ok=True)
for i in range(10):
    os.makedirs(f"{sorted_images_folder}/{i}", exist_ok=True)

# Just normalization for validation
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
# gdd.download_file_from_google_drive(file_id="1XHGEY234JVvf-UKrxw-u4kCcxFmFx-Xo",
#                                     dest_path="1KDataset/data/sd_e0_in_out.torch")

model.load_state_dict(torch.load("1KDataset/data/sd_e0_in_out.torch", map_location="cpu"))
model = torch.nn.DataParallel(model.to(device).eval())

print(f"Sorted Images Folder {sorted_images_folder}")
print(f"#Images {folder_counter}")
for i in range(3):
    print(f"Folder {i}")
    images_folder = (f"{input_folder}/{i}")
    print(f"Input to the Sorted Images Folder {images_folder}")
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root=images_folder, transform=data_transforms):
            super().__init__()
            self.paths = sorted(glob(f"{root}/*"))
            import random
            random.shuffle(self.paths)
            self.transform = transform
        def __getitem__(self, index):
            path = self.paths[index]
            pil_image = Image.open(path).convert("RGB")
            return path, self.transform(pil_image), pil_image.size
        def __len__(self): return len(self.paths)

    image_dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)


    with torch.no_grad():
        for images_paths, images, images_sizes in tqdm(dataloader, ncols=80):
            outputs = model(images.to(device))
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            list_is_indoor = outputs[:, 0].tolist()
            for image_path, image, image_size, is_indoor in \
                    zip(images_paths, images, torch.stack(images_sizes).T, list_is_indoor):
                assert f"{is_indoor:f}"[0] == "0"
                indoorness = f"{is_indoor:f}"[2]
                image_name = os.path.basename(image_path)
                if max(image_size) / min(image_size) > 2:  # It's a pano, remove it
                    _ = shutil.move(image_path, f"{sorted_images_folder}/panos/{image_name}")
                else:
                    _ = shutil.move(image_path, f"{sorted_images_folder}/{indoorness}/{image_name}")


