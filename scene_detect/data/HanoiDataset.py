from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import glob
import torch
import albumentations as A

AUGMENTERS = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.GaussNoise(p=0.5),
        A.Blur(p=0.5),
        A.CLAHE(p=0.5),
    ]
)

label_names = [
    "guom_lake",
    "west_lake",
    "turtle_tower",
    "thehuc_bridge",
    "post_office",
    "flower_garden",
    "tran_quoc_church",
    "quan_thanh_temple",
    "hospital",
    "water_park",
]


class HanoiDataset(Dataset):
    def __init__(
        self, image_root, annotation_file, img_size, transforms=None, is_training=True
    ):
        self.image_root = image_root
        self.img_size = img_size
        self.transforms = transforms
        self.is_training = is_training
        self.image_names, self.targets = self.read_annotation_file(annotation_file)

    def __len__(self):
        return len(self.image_names)

    def read_annotation_file(self, annotation_file):
        ann = pd.read_excel(annotation_file).dropna()
        img_names = []
        img_targets = []
        for _, row in ann.iterrows():
            img_name = str(row["id"]) + ".jpg"
            img_names.append(img_name)
            target = row[label_names].to_list()
            img_targets.append(target)

        return img_names, img_targets

    def check_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return False
        return True

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_root, img_name)
        img = cv2.imread(img_path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_name)
        img = cv2.resize(img, self.img_size)
        if self.is_training:
            img = AUGMENTERS(image=img)["image"]
        target = self.targets[idx]
        target = torch.tensor(target, dtype=torch.float32)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


if __name__ == "__main__":
    ds = HanoiDataset(
        "/home/luantranthanh/hust/dl_course/dataset",
        "/home/luantranthanh/hust/dl_course/dataset/label.xlsx",
    )
    print(ds[0])
