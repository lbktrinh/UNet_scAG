import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class SEMDataTrain(Dataset):  # input 3 channels

    def __init__(self, image_path, mask_path):

        self.image_arr = os.listdir(image_path)

        self.image_path = image_path
        self.mask_path = mask_path

        # Calculate len
        self.data_len = len(self.image_arr)

        self.transform = transforms.Compose(
            [transforms.ToTensor()])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        # GET IMAGE
        image = Image.open(os.path.join(self.image_path, single_image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_as_tensor = self.transform(image)

        # GET MASK
        mask = Image.open(os.path.join(self.mask_path, single_image_name[:-4] + '.tif'))  # cvc,
        # mask = Image.open(os.path.join(self.mask_path,single_image_name))  # vipcup, brain
        msk_as_np = np.asarray(mask)

        # Normalize mask to only 0 and 1
        msk_as_np = (msk_as_np / 255).astype(np.uint8)
        msk_as_tensor = torch.from_numpy(msk_as_np).float()  # Convert numpy array to tensor
        # msk_as_tensor= (msk_as_tensor > 0.5).float()

        return (img_as_tensor, msk_as_tensor)


class SEMDataVal(Dataset):  # input 3 channels

    def __init__(self, image_path, mask_path):
        self.image_arr = os.listdir(image_path)

        self.image_path = image_path
        self.mask_path = mask_path

        # Calculate len
        self.data_len = len(self.image_arr)

        self.transform = transforms.Compose(
            [transforms.ToTensor()])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]

        # GET IMAGE
        image = Image.open(os.path.join(self.image_path, single_image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_as_tensor = self.transform(image)

        # GET MASK
        mask = Image.open(os.path.join(self.mask_path, single_image_name[:-4] + '.tif'))  # cvc
        # mask = Image.open(os.path.join(self.mask_path,single_image_name))  # vipcup,brain
        msk_as_np = np.asarray(mask)
        # Normalize mask to only 0 and 1
        msk_as_np = (msk_as_np / 255).astype(np.uint8)
        msk_as_tensor = torch.from_numpy(msk_as_np).float()  # Convert numpy array to tensor
        # msk_as_tensor= (msk_as_tensor > 0.5).float()

        # return (img_as_tensor, msk_as_tensor)
        return (img_as_tensor, msk_as_tensor, single_image_name)
