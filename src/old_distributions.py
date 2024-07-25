import random
from math import ceil, sqrt

import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn import datasets


class Sampler:
    def __init__(
        self,
        device="cuda",
    ):
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


class SwissRollSampler(Sampler):
    def __init__(self, dim=2, device="cuda"):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2

    def sample(self, batch_size=10):
        batch = (
            datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[0].astype(
                "float32"
            )[:, [0, 2]]
            / 7.5
        )
        return torch.tensor(batch, device=self.device)


class SwissRoll3DSampler(Sampler):
    def __init__(self, dim=3, noise=0.5, device="cuda"):
        super(SwissRoll3DSampler, self).__init__(device=device)
        assert dim == 3
        self.dim = dim
        self.noise = noise
        self.colors = None

    def sample(self, batch_size=10):
        data, colors = datasets.make_swiss_roll(n_samples=batch_size, noise=self.noise)
        data = data.astype("float32") / 7.5
        self.colors = colors
        if self.dim == 3:
            batch = data[:, [0, 1, 2]]
        elif self.dim == 2:
            batch = data[:, [0, 2]]
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


class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device="cuda"):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)


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


class LinearTransformer(Transformer):
    def __init__(self, base_sampler, weight, bias=None, device="cuda"):
        super(LinearTransformer, self).__init__(device=device)
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


class StandartNormalSampler(Sampler):
    def __init__(self, dim=1, device="cuda", dtype=torch.float, requires_grad=False):
        super(StandartNormalSampler, self).__init__(device=device)
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


# class ZeroImageSampler(Sampler):
#     def __init__(
#         self, n_channels, h, w, device='cuda',
#         dtype=torch.float, requires_grad=False
#     ):
#         super(StandartNormalSampler, self).__init__(
#             device=device
#         )
#         self.requires_grad = requires_grad
#         self.dtype = dtype
#         self.n_channels = n_channels
#         self.h = h
#         self.w = w

#     def sample(self, batch_size=10):
#         return torch.zeros(batch_size, self.n_channels, self.h, self.w)
