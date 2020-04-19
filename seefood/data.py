import pickle

from PIL import Image
import torch
import lmdb
import numpy as np

class LMDBImageItem:
    def __init__(self, img_path, img_features, target):
        self._shape = img_features.shape
        self._img_features = img_features.numpy().tobytes()
        self._img_path = img_path
        self._target = target.round().item()

    @property
    def features(self):
        features = np.frombuffer(self._img_features, dtype=np.float32)
        return torch.from_numpy(features.reshape(self._shape))

    @property
    def path(self):
        return self._img_path

    @property
    def target(self):
        return self_target


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_filename, transform_target = None):
        self._env = lmdb.open(
            lmdb_filename,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._transform_target = transform_target

        with self._env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

    def __getitem__(self, index):
        with self._env.begin(write=False) as txn:
            key = f"{index:08}".encode("ascii")
            buf = txn.get(key)

        item = pickle.loads(buf)

        if self._transform_target:
            target = self._transform_target(item.target)
        else:
            target = item.target

        return item.path, item.features, item.target

    def __len__(self):
        return self.length


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self._images = df["image_path"].reset_index(drop=True)
        self._targets = df["target"].reset_index(drop=True)
        self._transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self._images.iloc[idx]
        with Image.open(img_path) as f:
            image = f.convert("RGB")
        if self._transform:
            image = self._transform(image)
        return img_path, image, self._targets.iloc[idx]
