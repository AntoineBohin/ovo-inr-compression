import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
import matplotlib.colors as colors
import math

######################## UNIQUE IMAGE SET ###########################

class ImageFile(Dataset):
    """
    Dataset class for a single image file.
    """
    def __init__(self, filename: str):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Image.Image:
        return self.img


################################ DATASETS ################################

class DIV2K(Dataset):
    """
    Dataset class for the DIV2K dataset.
    """
    def __init__(self, split: str, data_root: str, downsampled: bool = True, max_len: int = None):
        super().__init__()
        assert split in ['train', 'val'], "Unknown split"

        self.root = os.path.join(data_root, 'DIV2K')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.png'
        self.size = (768, 512)
        # Define transform
        self.transform = Compose([
            Resize(self.size, interpolation=Image.LANCZOS),
            ToTensor()
        ])
        # Load filenames based on the split
        if split == 'train':
            for i in range(0, 800):
                self.fnames.append("DIV2K_train_HR/{:04d}.png".format(i + 1))
        elif split == 'val':
            for i in range(800, 805):
                self.fnames.append("DIV2K_valid_HR/{:04d}.png".format(i + 1))
        self.downsampled = downsampled

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Image.Image:
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions
            if height > width: 
                img = img.rotate(90, expand=1)
            img = img.resize(self.size, Image.LANCZOS)
        return img


class CelebA(Dataset):
    """
    Dataset class for the CelebA dataset.
    """
    def __init__(self, split: str, data_root: str, downsampled: bool = False, max_len: int = None):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = os.path.join(data_root, 'CelebA', 'img_align_celeba/img_align_celeba')
        self.img_channels = 3
        self.fnames = []
        self.file_type = '.jpg'
        self.size = (178, 218)

        # Load filenames based on the split
        with open(os.path.join(data_root, 'CelebA', 'list_eval_partition.csv'), newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = 0
            for row in rowreader:
                if max_len and i >= max_len: break
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0].split('.')[0])
                    i += 1
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0].split('.')[0])
                    i += 1
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0].split('.')[0])

        self.downsampled = downsampled

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Image.Image:
        path = os.path.join(self.root, self.fnames[idx] + self.file_type)
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions
            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))
        return img


################# PIXEL COORDINATES WRAPPER #####################

class Implicit2DWrapper(Dataset):
    """
    Wrapper class to convert a dataset into a 2D implicit representation.
    """
    def __init__(self, dataset: Dataset, sidelength: int = None):
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        img = self.transform(self.dataset[idx])
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)
        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}
        return in_dict, gt_dict

    def get_item_small(self, idx: int) -> tuple:
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        ground_truth_dict = {'img': img}
        return spatial_img, img, ground_truth_dict

################################# UTILS #################################

def model_l1_dictdiff(ref_model_dict, model_dict, l1_lambda):
    "Computes the L1 norm of the difference between the parameters of two model state dictionaries. Used to regulaize models during training."
    l1_norm = sum((p.squeeze() - ref_p.squeeze()).abs().sum() for (p, ref_p) in zip(ref_model_dict.values(), model_dict.values()))
    return {'l1_loss': l1_lambda * l1_norm}

def model_l1(model, l1_lambda):
    "Computes the L1 norm of the models parameters and weights it with l1_lambda"
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return {'l1_loss': l1_lambda * l1_norm}

def get_mgrid(sidelen: int, dim: int = 2) -> torch.Tensor:
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    """
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
    pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def grads2img(gradients: torch.Tensor) -> torch.Tensor:
    """
    Converts gradients to an image representation using HSV color space.
    """
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.
    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)
    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)



