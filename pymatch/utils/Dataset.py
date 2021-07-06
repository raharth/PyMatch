import json
import os
from PIL import Image

import pandas as pd
import torch
from torch.utils import data
from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __getitem__(self, index, return_path=False):
        # @todo cannot be used in that way
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        if return_path:
            path = self.imgs[index][0]
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path
        return original_tuple


class Dataset(data.Dataset):

    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        return X, y

    @staticmethod
    def read_csv(path, target_index=-1, **read_args):
        data_frame = pd.read_csv(path, **read_args)
        # data_frame['species'] = pd.Categorical(data_frame['species']).codes
        data_frame = data_frame.sample(frac=1)
        label = torch.tensor(data_frame.values[:, target_index]).type(torch.FloatTensor)
        data = torch.tensor(data_frame.drop(index=target_index).values).type(torch.FloatTensor)
        return Dataset(data=data, labels=label)


class HardDriveDataset(data.Dataset):

    def __init__(self, list_IDs, labels):
        """
        Dataset for large data. It loads data on demand.
        Also see: "https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel"

        Example:
            Let ID be the Python string that identifies a given sample of the dataset.
            It assumes:
                partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
                labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

            And can be used by:
                training_set = Dataset(partition['train'], labels)
                training_generator = data.DataLoader(training_set, **params)

        Args:
            list_IDs:   file names for each data point
            labels:     labels according to the file names
        """
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y


class ObjectDetectionDataset(data.Dataset):
    def __init__(self, img_root, label_root, transforms, class_mapping):
        self.root = img_root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_root)))

        data = {}
        for file in os.listdir(label_root):
            if file[-4:] == 'json':
                with open(f'{label_root}/{file}', 'r') as f:
                    values = json.load(f)
                boxes = torch.FloatTensor(
                    [list(region['points'][0].values()) + list(region['points'][-2].values()) for region in
                     values['regions']])
                img_data = {
                    'boxes': boxes,
                    'labels': torch.tensor([class_mapping[region['tags'][0]] for region in values['regions']],
                                           dtype=torch.int64),
                    'image_id': None,
                    'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                    'iscrowd': torch.zeros((len(values['regions']),), dtype=torch.int64)
                }
                data[values['asset']['name']] = img_data
        self.data = data

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        target = self.data[self.imgs[idx]]
        target['image_id'] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ExtendedImageDataset(data.Dataset):

    def __init__(self, img_root, data_file, transforms=None):
        self.root = img_root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_root)))

        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        target = self.data[self.imgs[idx]]['y']
        # target['image_id'] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)