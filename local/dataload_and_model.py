import albumentations as A
import torch.nn as nn
import numpy as np
import torch
import timm
import cv2

from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


class train_preprocess(object):
    def __init__(self, args):
        self.args = args

        transform_bottle = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.7),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_cable = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.GaussNoise(p=0.7),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.ColorJitter(brightness=0.5, contrast=0.8, hue=[0.2, 0.5]),
            ToTensor()
        ])
        transform_capsule = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_carpet = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            ToTensor()
        ])
        transform_grid = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.ShiftScaleRotate(shift_limit=0.125,scale_limit=0.1, rotate_limit=30,p=0.5),
            ToTensor()
        ])
        transform_hazelnut = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.GaussNoise(p=0.7), 
            A.Rotate(p=0.5),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_leather = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.ShiftScaleRotate(shift_limit=0.125,scale_limit=0.1, rotate_limit=30,p=0.5),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_metal_nut = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.5),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_pill = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.5),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_screw = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_tile = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=10, max_w_size=10),
            ToTensor()
        ])
        transform_toothbrush = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            ToTensor()
        ])
        transform_transistor = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_wood = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.OpticalDistortion(always_apply=False, p=0.5, distort_limit=(-0.2, 0.2), interpolation=2),
            A.Rotate(p=0.9),
            A.Cutout(always_apply=False, p=0.5, num_holes=70, max_h_size=20, max_w_size=20),
            ToTensor()
        ])
        transform_zipper = A.Compose([
            A.RandomCrop(self.args.input_height, self.args.input_width),
            A.HorizontalFlip(p=0.5), # Same with transforms.RandomHorizontalFlip()
            A.VerticalFlip(p=0.5),
            A.Rotate(p=0.9),
            ToTensor()
        ])
        self.transforms = {"bottle": transform_bottle,
                    "cable":transform_cable,
                    "capsule":transform_capsule,
                    "carpet":transform_carpet,
                    "grid": transform_grid,
                    "hazelnut":transform_hazelnut,
                    "leather":transform_leather,
                    "metal_nut":transform_metal_nut,
                    "pill":transform_pill,
                    "screw":transform_screw,
                    "tile":transform_tile,
                    "toothbrush": transform_toothbrush,
                    "transistor":transform_transistor,
                    "wood":transform_wood,
                    "zipper":transform_zipper}

        self.transform_pred = A.Compose([
            ToTensor()
        ])


def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (700, 700))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


class Custom_dataset(Dataset):
    def __init__(self, args, img_arr_list, labels, classes = None, mode='train', transforms=None):
        self.args = args
        self.img_arr_list = img_arr_list
        self.labels = labels
        self.classes = classes
        self.labels_arr = np.array(labels)
        self.mode=mode
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_arr_list)

    def __getitem__(self, idx):
        img = self.img_arr_list[idx]
        
        if self.mode == 'train':
            clas = self.classes[idx]
            img = self.transforms[clas](image=img)['image']

        elif self.mode == 'test':
            img = self.transforms(image=img)['image']

        label = self.labels[idx]

        return img, label

    def get_labels(self):
        return self.labels


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x