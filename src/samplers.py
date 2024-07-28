import os
import random
from math import ceil, sqrt

import numpy as np
import sklearn
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from src.tools import get_random_colored_images, h5py_to_dataset


class ZeroImageDataset(Dataset):
    def __init__(self, n_channels, h, w, n_samples, transform=None):
        self.n_channels = n_channels
        self.h = h
        self.w = w
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.ones(self.n_channels, self.h, self.w), torch.zeros(
            self.n_channels, self.h, self.w
        )


# ============================================ #
class Sampler:
    def __init__(self, device="cuda"):
        self.device = device

    def sample(self, size=5):
        pass


class LoaderSampler(Sampler):
    def __init__(self, loader, device="cuda"):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)


def get_loader_sampler(
    name,
    path,
    img_size=64,
    batch_size=64,
    device="cuda",
    load_ambient=False,
    num_workers=8,
):
    if name in ["CelebA_low", "CelebA_high"]:
        res = name.split("_")[1]

        if res == "high":
            transform = Compose(
                [
                    CenterCrop(140),
                    Resize((64, 64)),
                    ToTensor(),
                    Lambda(lambda x: 2 * x - 1),
                ]
            )
        elif res == "low":
            transform = Compose(
                [
                    CenterCrop(140),
                    Resize((32, 32)),
                    Resize((64, 64)),
                    ToTensor(),
                    Lambda(lambda x: 2 * x - 1),
                ]
            )

        dataset = ImageFolder(path, transform=transform)

        train_ratio = 0.45
        test_ratio = 0.1

        train_size = int(len(dataset) * train_ratio)
        test_size = int(len(dataset) * test_ratio)
        idx = np.arange(len(dataset))

        np.random.seed(0x000000)
        np.random.shuffle(idx)

        if res == "low":
            train_idx = idx[:train_size]
        elif res == "high":
            train_idx = idx[train_size:-test_size]
        test_idx = idx[-test_size:]

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        train_sampler = LoaderSampler(
            DataLoader(
                train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
            ),
            device,
        )
        test_sampler = LoaderSampler(
            DataLoader(
                test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
            ),
            device,
        )
        return train_sampler, test_sampler
    if name.startswith("MNIST"):
        # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
        transform = Compose(
            [
                Resize((32, 32)),
                ToTensor(),
                Lambda(lambda x: 2 * x - 1),
            ]
        )

        dataset_name = name.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"

        classes = [int(number) for number in name.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]

        train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

        train_test = []

        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(
                    torch.stack(
                        [
                            dataset[i][0]
                            for i in range(len(dataset.targets))
                            if dataset.targets[i] == classes[k]
                        ],
                        dim=0,
                    )
                )
                labels += [k] * data[-1].shape[0]
            data = torch.cat(data, dim=0)
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)

            if is_colored:
                data = get_random_colored_images(data)

            train_test.append(TensorDataset(data, labels))

        train_set, test_set = train_test
        train_sampler = LoaderSampler(
            DataLoader(
                train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
            ),
            device,
        )
        test_sampler = LoaderSampler(
            DataLoader(
                test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
            ),
            device,
        )
        return train_sampler, test_sampler
    # ===============================================================================================
    # load dataset
    if name in ["shoes", "handbag", "outdoor", "church"]:
        dataset = h5py_to_dataset(path, img_size)
    elif name in [
        "celeba_female",
        "celeba_male",
        "aligned_anime_faces",
        "comics",
        "faces",
    ]:
        transform = Compose(
            [
                Resize((img_size, img_size)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageFolder(path, transform=transform)
    elif name in ["dtd"]:
        transform = Compose(
            [
                Resize(300),
                RandomResizedCrop(
                    (img_size, img_size), scale=(128.0 / 300, 1.0), ratio=(1.0, 1.0)
                ),
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageFolder(path, transform=transform)
    elif name in ["cartoon_faces"]:
        transform = Compose(
            [
                CenterCrop(420),
                Resize((img_size, img_size)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageFolder(path, transform=transform)
    elif name in ["fruit360"]:
        transform = Compose(
            [
                Pad(14, fill=(255, 255, 255)),
                Resize((img_size, img_size)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ConcatDataset(
            (
                ImageFolder(os.path.join(path, "Training"), transform=transform),
                ImageFolder(os.path.join(path, "Test"), transform=transform),
            )
        )
    elif name in ["summer", "winter", "vangogh", "photo"]:
        if load_ambient:
            transform = Compose(
                [
                    Resize((img_size, img_size)),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = Compose(
                [
                    RandomCrop(128),
                    RandomHorizontalFlip(0.5),
                    Resize((img_size, img_size)),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception("Unknown dataset")

    # generate index list
    if name in ["celeba_female", "celeba_male"]:
        with open("../datasets/list_attr_celeba.txt", "r") as f:
            lines = f.readlines()[2:]
        if name == "celeba_female":
            idx = [
                i
                for i in list(range(len(lines)))
                if lines[i].replace("  ", " ").split(" ")[21] == "-1"
            ]
        else:
            idx = [
                i
                for i in list(range(len(lines)))
                if lines[i].replace("  ", " ").split(" ")[21] != "-1"
            ]
    elif name in ["comics", "faces"]:
        idx = list(range(len(dataset)))
        if name == "faces":
            idx = np.array(idx)[np.array(dataset.targets) == 1]
        else:
            idx = np.array(idx)[np.array(dataset.targets) == 0]
    else:
        idx = list(range(len(dataset)))
    # splite train test dataset
    test_ratio = 0.1
    test_size = int(len(idx) * test_ratio)
    if name in ["summer", "vangogh"]:
        train_idx = np.array(idx)[np.array(dataset.targets) == 2]
        test_idx = np.array(idx)[np.array(dataset.targets) == 0]
    elif name in ["winter", "photo"]:
        train_idx = np.array(idx)[np.array(dataset.targets) == 3]
        test_idx = np.array(idx)[np.array(dataset.targets) == 1]
    elif name == "fruit360":
        train_idx = idx[: len(dataset.datasets[0])]
        test_idx = idx[len(dataset.datasets[0]) :]
    elif name == "dtd":
        np.random.seed(0x000000)
        np.random.shuffle(idx)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    else:
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]

    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    #     print(len(train_idx), len(test_idx))

    # generate LoaderSampler
    train_sampler = LoaderSampler(
        DataLoader(
            train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    test_sampler = LoaderSampler(
        DataLoader(
            test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    return train_sampler, test_sampler


# ====================== Paired Guided ====================== #
def paired_random_hflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
        im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
    return im1, im2


def paired_random_vflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_TOP_BOTTOM)
        im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)
    return im1, im2


def paired_random_rotate(im1, im2):
    angle = np.random.rand() * 360
    im1 = im1.rotate(angle, fillcolor=(255, 255, 255))
    im2 = im2.rotate(angle, fillcolor=(255, 255, 255))
    return im1, im2


def paired_random_crop(im1, im2, size):
    assert im1.size == im2.size, "Images must have exactly the same size"
    assert size[0] <= im1.size[0]
    assert size[1] <= im1.size[1]

    x1 = np.random.randint(im1.size[0] - size[0])
    y1 = np.random.randint(im1.size[1] - size[1])

    im1 = im1.crop((x1, y1, x1 + size[0], y1 + size[1]))
    im2 = im2.crop((x1, y1, x1 + size[0], y1 + size[1]))

    return im1, im2


class PairedDataset(Dataset):
    def __init__(
        self,
        data_folder,
        labels_folder,
        transform=None,
        reverse=False,
        hflip=False,
        vflip=False,
        crop=None,
    ):
        self.transform = transform
        self.data_paths = sorted(
            [
                os.path.join(data_folder, file)
                for file in os.listdir(data_folder)
                if (
                    os.path.isfile(os.path.join(data_folder, file))
                    and file[-4:] in [".png", ".jpg"]
                )
            ]
        )
        self.labels_paths = sorted(
            [
                os.path.join(labels_folder, file)
                for file in os.listdir(labels_folder)
                if (
                    os.path.isfile(os.path.join(labels_folder, file))
                    and file[-4:] in [".png", ".jpg"]
                )
            ]
        )
        assert len(self.data_paths) == len(self.labels_paths)
        self.reverse = reverse
        self.hflip = hflip
        self.vflip = vflip
        self.crop = crop

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index]).convert("RGB")
        y = Image.open(self.labels_paths[index]).convert("RGB")
        if self.crop is not None:
            x, y = paired_random_crop(x, y, size=self.crop)
        if self.hflip:
            x, y = paired_random_hflip(x, y)
        if self.vflip:
            x, y = paired_random_vflip(x, y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return (
            (
                x,
                y,
            )
            if not self.reverse
            else (
                y,
                x,
            )
        )

    def __len__(self):
        return len(self.data_paths)


def split_glued_image(im):
    w, h = im.size
    im_l, im_r = im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))
    return im_l, im_r


class GluedDataset(Dataset):
    def __init__(
        self,
        path,
        transform=None,
        reverse=False,
        hflip=False,
        vflip=False,
        crop=None,
        rotate=False,
    ):
        self.path = path
        self.transform = transform
        self.data_paths = [
            os.path.join(path, file)
            for file in os.listdir(path)
            if (
                os.path.isfile(os.path.join(path, file))
                and file[-4:] in [".png", ".jpg"]
            )
        ]
        self.reverse = reverse
        self.hflip = hflip
        self.vflip = vflip
        self.crop = crop
        self.rotate = rotate

    def __getitem__(self, index):
        xy = Image.open(self.data_paths[index])
        x, y = split_glued_image(xy)
        if self.reverse:
            x, y = y, x
        if self.crop is not None:
            x, y = paired_random_crop(x, y, size=self.crop)
        if self.hflip:
            x, y = paired_random_hflip(x, y)
        if self.vflip:
            x, y = paired_random_vflip(x, y)
        if self.rotate:
            x, y = paired_random_rotate(x, y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.data_paths)


class PairedLoaderSampler(Sampler):
    def __init__(self, loader, device="cuda"):
        super(PairedLoaderSampler, self).__init__(device)
        # print("[Debug] PairedLoaderSampler: init")
        self.loader = loader
        # print("[Debug] PairedLoaderSampler: build iter")
        self.it = iter(self.loader)
        # print("[Debug] PairedLoaderSampler: init, OK")

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch_X, batch_Y = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch_X) < size:
            return self.sample(size)

        return batch_X[:size].to(self.device), batch_Y[:size].to(self.device)


def get_paired_sampler(
    name,
    path,
    img_size=64,
    batch_size=64,
    device="cuda",
    reverse=False,
    load_ambient=False,
    num_workers=8,
):
    transform = Compose(
        [
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if name in ["edges2shoes", "edges2handbags", "anime-sketch"]:
        hflip = True if (name != "edges2shoes") else False
        train_set = GluedDataset(
            os.path.join(path, "train"),
            transform=transform,
            reverse=reverse,
            hflip=hflip,
        )
        test_set = GluedDataset(
            os.path.join(path, "val"), transform=transform, reverse=reverse, hflip=hflip
        )
    elif name in ["facades", "maps", "cityscapes"]:
        crop = (300, 300) if name == "maps" else None
        vflip = False if name in ["facades", "cityscapes"] else True
        train_set = GluedDataset(
            os.path.join(path, "train"),
            transform=transform,
            reverse=reverse,
            hflip=True,
            vflip=vflip,
            crop=crop,
        )
        test_set = GluedDataset(
            os.path.join(path, "val"),
            transform=transform,
            reverse=reverse,
            hflip=True,
            vflip=vflip,
            crop=crop,
        )
    elif name == "gta5_legend_map":
        input, target = name.split("_")[1:]
        train_set = PairedDataset(
            os.path.join(path, input, "train"),
            os.path.join(path, target, "train"),
            transform=transform,
            reverse=reverse,
            hflip=True,
            vflip=True,
        )
        test_set = PairedDataset(
            os.path.join(path, input, "test"),
            os.path.join(path, target, "test"),
            transform=transform,
            reverse=reverse,
            hflip=True,
            vflip=True,
        )
    elif name == "FS2K":
        source_folder, target_folder = "sketch", "photo"
        train_set = PairedDataset(
            os.path.join(path, "train", source_folder),
            os.path.join(path, "train", target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
        test_set = PairedDataset(
            os.path.join(path, "test", source_folder),
            os.path.join(path, "test", target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
    elif name in [
        "comic_faces",
        "comic_faces_v1",
        "celeba_mask",
        "aligned_anime_faces_sketch",
        "safebooru_sketch",
    ]:
        if name == "comic_faces":
            source_folder, target_folder = "faces", "comics"
        elif name == "comic_faces_v1":
            source_folder, target_folder = "face", "comics"
        elif name == "celeba_mask":
            source_folder, target_folder = "CelebAMask-HQ-mask-color", "CelebA-HQ-img"
        elif name == "safebooru_sketch":
            source_folder, target_folder = "safebooru_sketch", "safebooru_jpeg"
        else:
            source_folder, target_folder = "sketch", "image"
        # print("[Debug] data path, OK")
        dataset = PairedDataset(
            os.path.join(path, source_folder),
            os.path.join(path, target_folder),
            transform=transform,
            reverse=reverse,
            hflip=True,
        )
        # print("[Debug] dataset build, OK")
        idx = list(range(len(dataset)))
        test_ratio = 0.1
        test_size = int(len(idx) * test_ratio)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
        train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
        # print("[Debug] train/test partition, OK")
    else:
        raise Exception("Unknown dataset")

    train_sampler = PairedLoaderSampler(
        DataLoader(
            train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    # print("[Debug] train sampler build, OK")
    test_sampler = PairedLoaderSampler(
        DataLoader(
            test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    # print("[Debug] test sampler build, OK")
    return train_sampler, test_sampler


# ====================== Subset(Class) Guided ====================== #
def get_indicies_subset(dataset, new_labels={}, classes=4, subset_classes=None):
    labels_subset = []
    dataset_subset = []
    class_indicies = [[] for _ in range(classes)]
    i = 0
    for x, y in dataset:
        if y in subset_classes:
            if isinstance(y, int):
                class_indicies[new_labels[y]].append(i)
                labels_subset.append(new_labels[y])
            else:
                class_indicies[new_labels[y.item()]].append(i)
                labels_subset.append(new_labels[y.item()])
            dataset_subset.append(x)
            i += 1
    return dataset_subset, labels_subset, class_indicies


class SubsetGuidedDataset(Dataset):
    def __init__(
        self,
        dataset_in,
        dataset_out,
        num_labeled="all",
        in_indicies=None,
        out_indicies=None,
    ):
        super(SubsetGuidedDataset, self).__init__()
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        assert len(in_indicies) == len(
            out_indicies
        )  # make sure in and out have same num of classes
        self.num_classes = len(in_indicies)
        self.subsets_in = in_indicies
        self.subsets_out = out_indicies
        if (
            num_labeled != "all"
        ):  # semi-supervision training: just using a less number of labeled samplers in each class
            assert isinstance(num_labeled, int)
            tmp_list = [
                np.random.choice(subset, num_labeled) for subset in self.subsets_out
            ]
            self.subsets_out = tmp_list
            # self.subset_out now is a list of index list, [[...],[...],...,[...]] , lenght is num_classes * num_labeled

    def get(self, class_, subsetsize):
        x_subset = []
        y_subset = []
        in_indexis = random.sample(list(self.subsets_in[class_]), subsetsize)
        out_indexis = random.sample(list(self.subsets_out[class_]), subsetsize)
        for x_i, y_i in zip(in_indexis, out_indexis):
            x, c1 = self.dataset_in[x_i]
            y, c2 = self.dataset_out[y_i]
            assert c1 == c2
            x_subset.append(x)
            y_subset.append(y)
        # shape=(subsetsize, sample/label)
        # sample.shape=(channels, height, width), label.shape=(1,)
        return torch.stack(x_subset), torch.stack(y_subset)

    def __len__(self):
        return len(self.dataset_in)


class SubsetGuidedSampler(Sampler):
    def __init__(self, loader, subsetsize=8, weight=None, device="cuda"):
        super(SubsetGuidedSampler, self).__init__(device)
        self.loader = loader
        self.subsetsize = subsetsize
        if weight is None:  # if no weight given, uniform probability for each class
            self.weight = [
                1 / self.loader.num_classes for _ in range(self.loader.num_classes)
            ]
        else:
            self.weight = weight

    def sample(self, num_selected_classes=5):
        classes = np.random.choice(
            self.loader.num_classes, num_selected_classes, p=self.weight
        )
        batch_X = []
        batch_Y = []
        batch_label = []
        with torch.no_grad():
            for class_ in classes:
                X, Y = self.loader.get(class_, self.subsetsize)
                batch_X.append(X.clone().to(self.device).float())
                batch_Y.append(Y.clone().to(self.device).float())
                for _ in range(self.subsetsize):
                    batch_label.append(class_)

        # shape=(num_selected_classes, subsetsize, sample/label)
        # sample.shape=(channels, height, width), label.shape=(1,)
        return torch.stack(batch_X).to(self.device), torch.stack(batch_Y).to(
            self.device
        )


# !TODO: support other subset guided dataset
# def get_subset_guided_sampler(
#     name,
#     path,
#     img_size=64,
#     num_labeled="all",
#     batch_size=64,
#     device="cuda",
#     reverse=False,
#     num_workers=8,
# ):
#     if name in ["fmnist2mnist"]:
#         transform = Compose(
#             [
#                 Resize((img_size, img_size)),
#                 ToTensor(),
#                 Normalize((0.5), (0.5)),
#             ]
#         )

#         mnist_train = datasets.MNIST(
#             root=path, train=True, download=True, transform=transform
#         )
#         fashion_train = datasets.FashionMNIST(
#             root=path, train=True, download=True, transform=transform
#         )
#         mnist_test = datasets.MNIST(
#             root=path, train=False, download=True, transform=transform
#         )
#         fashion_test = datasets.FashionMNIST(
#             root=path, train=False, download=True, transform=transform
#         )

#         if reverse:
#             train_set = SubsetGuidedDataset(
#                 mnist_train, fashion_train, num_labeled=num_labeled
#             )
#             test_set = SubsetGuidedDataset(mnist_test, fashion_test)
#         else:
#             train_set = SubsetGuidedDataset(
#                 fashion_train, mnist_train, num_labeled=num_labeled
#             )
#             test_set = SubsetGuidedDataset(fashion_test, mnist_test)
#     elif name == "usps2mnist":
#         transform = Compose(
#             [
#                 Resize((img_size, img_size)),
#                 ToTensor(),
#                 Normalize((0.5), (0.5)),
#             ]
#         )

#         usps_train = datasets.USPS(
#             root=path, train=True, download=True, transform=transform
#         )
#         mnist_train = datasets.MNIST(
#             root=path, train=True, download=True, transform=transform
#         )
#         usps_test = datasets.USPS(
#             root=path, train=False, download=True, transform=transform
#         )
#         mnist_test = datasets.MNIST(
#             root=path, train=False, download=True, transform=transform
#         )

#         if reverse:
#             train_set = SubsetGuidedDataset(
#                 mnist_train, usps_train, num_labeled=num_labeled
#             )
#             test_set = SubsetGuidedDataset(mnist_test, usps_test)
#         else:
#             train_set = SubsetGuidedDataset(
#                 usps_train, mnist_train, num_labeled=num_labeled
#             )
#             test_set = SubsetGuidedDataset(usps_test, mnist_test)
#     else:
#         raise Exception("Unknown dataset")

#     train_sampler = PairedLoaderSampler(
#         DataLoader(
#             train_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
#         ),
#         device,
#     )
#     test_sampler = PairedLoaderSampler(
#         DataLoader(
#             test_set, shuffle=True, num_workers=num_workers, batch_size=batch_size
#         ),
#         device,
#     )
#     return train_sampler, test_sampler


# ====================== distributions ====================== #
class StandardNormalSampler0(Sampler):
    def __init__(self, dim=1, device="cuda"):
        super(StandardNormalSampler0, self).__init__(device=device)
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)


class StandardNormalSampler1(Sampler):
    def __init__(self, dim=1, device="cuda", dtype=torch.float, requires_grad=False):
        super(StandardNormalSampler1, self).__init__(device=device)
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(
            batch_size,
            self.dim,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )


class StandardNormalSampler(Sampler):
    def __init__(self, dim=2, device="cuda"):
        super(StandardNormalSampler, self).__init__(device)
        self.dim, self.shape = dim, (dim,)
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var, self.cov = float(dim), np.eye(dim, dtype=np.float32)

    def sample(self, size=10):
        return torch.randn(size, self.dim, device=self.device)


class Mix8GaussiansSampler(Sampler):
    def __init__(self, with_central=False, std=1, r=12, dim=2, device="cuda"):
        super(Mix8GaussiansSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r

        self.with_central = with_central
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch


# ================================================================================= #
class SwissRollSampler(Sampler):
    def __init__(self, noise=0.0, dim=3, device="cuda"):
        super(SwissRollSampler, self).__init__(device=device)
        self.dim = dim
        self.noise = noise

    def sample(self, batch_size=100):
        data, index = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size, noise=self.noise
        )
        data = data.astype("float32") / 7.5
        if self.dim == 3:
            batch = data[:, [0, 1, 2]]
        elif self.dim == 2:
            batch = data[:, [0, 2]]
        else:
            raise Exception("Unsupport swiss roll dim, please use 2 or 3")
        return torch.tensor(batch, device=self.device)


class MobiusStripSampler(Sampler):
    def __init__(self, dim=3, noise=0.1, device="cuda"):
        super(MobiusStripSampler, self).__init__(device=device)
        assert dim == 3
        self.dim = dim
        self.noise = noise
        self.colors = None

    def sample(self, batch_size=1000):
        self.colors = [(i / batch_size) for i in range(batch_size)]
        uv_size = ceil(sqrt(batch_size))
        # u 的范围从 0 到 2π
        u = np.linspace(0, 2 * np.pi, uv_size)
        # v 的范围从 0 到 1
        v = np.linspace(-1, 1, uv_size)
        u, v = np.meshgrid(u, v)

        # 计算 x, y, z 坐标
        x, y, z = self._mobius(u, v, batch_size)
        # 将数据重塑为 batch_size 行，3列
        data = np.vstack((x, y, z)).T

        return torch.tensor(data, dtype=torch.float, device=self.device)

    def _mobius(self, u, v, batch_size):
        x = (1 + 0.5 * v * np.cos(u / 2)) * np.cos(u)
        y = (1 + 0.5 * v * np.cos(u / 2)) * np.sin(u)
        z = 0.5 * v * np.sin(u / 2)
        x, y, z = (
            x.flatten()[:batch_size],
            y.flatten()[:batch_size],
            z.flatten()[:batch_size],
        )
        x += self.noise * (np.random.rand(*x.shape) - 0.5)
        y += self.noise * (np.random.rand(*y.shape) - 0.5)
        z += self.noise * (np.random.rand(*z.shape) - 0.5)
        return x, y, z


class DoubleMoonSampler(Sampler):
    def __init__(self, noise=0.0, device="cuda"):
        super(DoubleMoonSampler, self).__init__(device=device)
        self.noise = noise

    def sample(self, batch_size=100):
        data, label = sklearn.datasets.make_moons(batch_size, self.noise)
        batch = data.astype("float32")
        return torch.tensor(batch, device=self.device)


class CubeUniformSampler(Sampler):
    def __init__(self, dim=1, centered=False, normalized=False, device="cuda"):
        super(CubeUniformSampler, self).__init__(
            device=device,
        )
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = (
            np.eye(self.dim, dtype=np.float32)
            if self.normalized
            else np.eye(self.dim, dtype=np.float32) / 12
        )
        self.mean = (
            np.zeros(self.dim, dtype=np.float32)
            if self.centered
            else 0.5 * np.ones(self.dim, dtype=np.float32)
        )

        self.bias = torch.tensor(self.mean, device=self.device)

    def sample(self, batch_size=10):
        return (
            np.sqrt(self.var)
            * (torch.rand(batch_size, self.dim, device=self.device) - 0.5)
            / np.sqrt(self.dim / 12)
            + self.bias
        )


# ================================================================================= #


class Transformer(object):
    def __init__(self, device="cuda"):
        self.device = device


class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device="cuda"):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()

        mean, cov = np.mean(batch, axis=0), np.cov(batch.T)

        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)

        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier), device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(
                self.base_sampler.sample(batch_size), device=self.device
            )
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch


class LinearTransformer0(Transformer):
    def __init__(self, base_sampler, weight, bias=None, device="cuda"):
        super(LinearTransformer0, self).__init__(device=device)
        self.base_sampler = base_sampler

        self.weight = torch.tensor(weight, device=device, dtype=torch.float32)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32)
        else:
            self.bias = torch.zeros(
                self.weight.size(0), device=device, dtype=torch.float32
            )

    def sample(self, size=4):
        batch = torch.tensor(self.base_sampler.sample(size), device=self.device)
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch


class LinearTransformer(Transformer):
    def __init__(self, weight, bias=None, base_sampler=None, device="cuda"):
        super(LinearTransformer, self).__init__(device=device)

        self.fitted = False
        self.dim = weight.shape[0]
        self.weight = torch.tensor(
            weight, device=device, dtype=torch.float32, requires_grad=False
        )
        if bias is not None:
            self.bias = torch.tensor(
                bias, device=device, dtype=torch.float32, requires_grad=False
            )
        else:
            self.bias = torch.zeros(
                self.dim, device=device, dtype=torch.float32, requires_grad=False
            )

        if base_sampler is not None:
            self.fit(base_sampler)

    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        weight, bias = self.weight.cpu().numpy(), self.bias.cpu().numpy()

        self.mean = weight @ self.base_sampler.mean + bias
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)

        self.fitted = True
        return self

    def sample(self, batch_size=4):
        assert self.fitted is True

        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device,
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch = batch.detach()
        return batch


def symmetrize(X):
    return np.real((X + X.T) / 2)


class NormalSampler(Sampler):
    def __init__(self, mean, cov=None, weight=None, device="cuda"):
        super(NormalSampler, self).__init__(device=device)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]

        if weight is not None:
            weight = np.array(weight, dtype=np.float32)

        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)

        if weight is None:
            weight = symmetrize(sqrtm(self.cov))

        self.var = np.trace(self.cov)

        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, batch_size=4):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch
