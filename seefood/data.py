import pickle

from PIL import Image
import torch
from torchvision import transforms
import lmdb
import numpy as np
from tqdm.notebook import tqdm

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
        return self._target


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

        return item.path, item.features, target

    def __len__(self):
        return self.length


def get_default_transform(image_size):
    return transforms.Compose(
       [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # mean and std of imagenet
        ]
    )


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self._images = df["image_path"].reset_index(drop=True)
        self._targets = df["target"].reset_index(drop=True)
        self._transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img_path = self._images.iloc[idx]
        with Image.open(img_path) as f:
            image = f.convert("RGB")
        if self._transform:
            image = self._transform(image)
        return img_path, image, self._targets.iloc[idx]


class LMDBEmbeddingWriter:

    def __init__(self, feature_extractor, device):
        self._feature_extractor = feature_extractor
        self._device = device

    def write(self, lmdb_filename, dataloader, map_size):
        index = 0
        with lmdb.open(lmdb_filename, map_size=map_size) as env:
            for batch in tqdm(dataloader):
                index += self._write_to_env(env, batch, index)

    def _write_to_env(self, env, batch, index):
        image_paths, images, targets = batch
        assert len(image_paths) == len(images) == len(targets)

        written_count = 0
        cpu = torch.device("cpu")
        with env.begin(write=True) as txn:
            images = images.to(self._device)
            features = self._feature_extractor(images).to(cpu)
            for p, f, t in zip(image_paths, features, targets):
                key = f"{index + written_count:08}".encode("ascii")
                value = LMDBImageItem(p, f, t)
                txn.put(key, pickle.dumps(value))
                written_count += 1

        return written_count

