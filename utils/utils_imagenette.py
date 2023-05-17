import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
# if "charles" in current_dir:
#     parent_dir = os.path.dirname(current_dir)
#     sys.path.append(parent_dir)
#     sys.path.append(current_dir)
import numpy as np
from math import floor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models.resnet import ResNet18_Weights
from utils.utils_architectures import ResNet18_pl
import utils.utils_shift_functions as usf


net = ResNet18_pl()
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

class Imagenette(Dataset):
    def __init__(self, root, split, transform=None, shift=None, shift_args=None, delta=0.0):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.shift = shift
        self.shift_args = shift_args

        assert(split in ['val', 'train'])
        assert 0.0 <= delta <= 1.0
        self.delta = delta

        if self.shift:
            if self.shift == 'gaussian_noise':
                self.shift_function = usf.gaussian_noise
            elif self.shift == 'gaussian_blur':
                self.shift_args.update({'unsqueeze': False})
                self.shift_function = usf.gaussian_blur
            elif self.shift == 'image_shift':
                self.shift_args.update({'not_batched': True})
                self.shift_function = usf.image_shift
            else:
                NotImplemented(
                    'Please choose "image_shift", "ko_shift", "gaussian_noise" or "gaussian_blur" as a shift function.'
                )

        simpler_classes_names = {
            "n01440764": 0,
            "n02102040": 1,
            "n02979186": 2,
            "n03000684": 3,
            "n03028079": 4,
            "n03394916": 5,
            "n03417042": 6,
            "n03425413": 7,
            "n03445777": 8,
            "n03888257": 9,
        }
        samples_dir = os.path.join(root, split)
        for label in os.listdir(samples_dir):
            class_samples_dir = os.path.join(samples_dir, label)
            for sample in os.listdir(class_samples_dir):
                target = simpler_classes_names[label]
                sample_path = os.path.join(class_samples_dir, sample)
                self.samples.append(sample_path)
                self.targets.append(target)

        self.shifted_indices = None
        if self.delta != 0.0:
            num_shifted_samples = floor(len(self.samples)*self.delta)
            perm = np.random.permutation(len(self.samples))
            self.shifted_indices = perm[:num_shifted_samples]


    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            ber = np.random.binomial(1, self.delta)
            if self.transform:
                x = self.transform(x)
            if self.shift and idx in self.shifted_indices :
                x = self.shift_function(x, **self.shift_args)
            # if self.shift and self.delta != 0.0 and ber == 1:
            #     x = self.shift_function(x, **self.shift_args)
            return x, self.targets[idx]

