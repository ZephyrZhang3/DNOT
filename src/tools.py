import gc
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from IPython.display import display, update_display
from PIL import Image
from torch.utils.data import TensorDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from src.inception import InceptionV3


def ema_update(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.data.copy_(beta * p_tgt.data + (1.0 - beta) * p_src.data)


def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def get_random_colored_images(images, seed=0x000000):
    np.random.seed(seed)

    images = 0.5 * (images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360 * np.random.rand(size)

    for V, H in zip(images, hues):
        V_min = 0

        a = (V - V_min) * (H % 60) / 60
        V_inc = a
        V_dec = V - a

        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H / 60) % 6

        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec

        colored_images.append(colored_image)

    colored_images = torch.stack(colored_images, dim=0)
    colored_images = 2 * colored_images - 1

    return colored_images


def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = (
            2
            * (torch.tensor(np.array(data), dtype=torch.float32) / 255.0).permute(
                0, 3, 1, 2
            )
            - 1
        )
        dataset = F.interpolate(dataset, img_size, mode="bilinear")

    return TensorDataset(dataset, torch.zeros(len(dataset)))


def ewma(x, span=200):
    return pd.DataFrame({"x": x}).ewm(span=span).mean().values[:, 0]


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))

    return buf


def fig2tensor(fig):
    rgb_buf = fig2data(fig)
    img_tensor = torch.from_numpy(rgb_buf)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.float().div(255)
    return img_tensor


def fig2img(fig):
    buf = fig2data(fig)
    w, h, c = buf.shape
    return Image.frombytes("RGB", (w, h), buf.tostring())


# ========== Energy Distance ============== #
def energy_distance(X1, X2, X3, Y1, Y2, Y3):
    assert X1.shape == X2.shape
    assert X3.shape == Y3.shape
    assert Y1.shape == Y2.shape
    assert len(X1.shape) == 2
    assert len(X3.shape) == 2
    assert len(Y1.shape) == 2
    ED = (
        np.linalg.norm(X3 - Y3, axis=1).mean()
        - 0.5 * np.linalg.norm(X1 - X2, axis=1).mean()
        - 0.5 * np.linalg.norm(Y1 - Y2, axis=1).mean()
    )
    return ED


def EnergyDistances(T, XY_sampler, size=1048, batch_size=8, device="cuda"):
    assert size % batch_size == 0
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    freeze(model)
    freeze(T)

    pred_arr = []
    pixels = []

    with torch.no_grad():
        num_batches = size // batch_size
        for j in range(6 * num_batches):
            X, Y = XY_sampler.sample(batch_size)
            batch = T(X) if j < 3 * num_batches else Y
            img_size = batch.shape[2] * batch.shape[3]

            # inception stats
            pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(batch_size, -1))

            # color stats
            pixel_idx = np.random.randint(0, img_size, size)
            for k in range(batch_size):
                pixels.append(
                    batch[k].flatten(start_dim=1)[:, pixel_idx[k]].cpu().numpy()
                )

    # EID
    pred_arr = np.vstack(pred_arr)
    X1, X2, X3 = (
        pred_arr[:size],
        pred_arr[size : 2 * size],
        pred_arr[2 * size : 3 * size],
    )
    Y1, Y2, Y3 = (
        pred_arr[-3 * size : -2 * size],
        pred_arr[-2 * size : -size],
        pred_arr[-size:],
    )
    EID = energy_distance(X1, X2, X3, Y1, Y2, Y3)

    # ECD
    pixels = np.array(pixels)
    X1, X2, X3 = pixels[:size], pixels[size : 2 * size], pixels[2 * size : 3 * size]
    Y1, Y2, Y3 = (
        pixels[-3 * size : -2 * size],
        pixels[-2 * size : -size],
        pixels[-size:],
    )
    ECD = energy_distance(X1, X2, X3, Y1, Y2, Y3)

    gc.collect()
    torch.cuda.empty_cache()
    return EID, ECD


def EnergyColorDistance(T, XY_sampler, size=2048, batch_size=8, device="cuda"):
    assert size % batch_size == 0

    pred_arr = []
    with torch.no_grad():
        num_batches = size // batch_size
        for j in range(6 * num_batches):
            X, Y = XY_sampler.sample(batch_size)
            batch = T(X) if j < 3 * num_batches else Y
            img_size = batch.shape[2] * batch.shape[3]
            batch = batch.reshape(batch_size, 3, -1)
            pixel_idx = np.random.randint(0, img_size, size)
            for k in range(batch_size):
                pred_arr.append(batch[k, :, pixel_idx[k]].cpu().numpy())

    pred_arr = np.array(pred_arr)
    X1, X2, X3 = (
        pred_arr[:size],
        pred_arr[size : 2 * size],
        pred_arr[2 * size : 3 * size],
    )
    Y1, Y2, Y3 = (
        pred_arr[-3 * size : -2 * size],
        pred_arr[-2 * size : -size],
        pred_arr[-size:],
    )
    ECD = energy_distance(X1, X2, X3, Y1, Y2, Y3)

    gc.collect()
    torch.cuda.empty_cache()
    return ECD


# ========== Mapping ================= #
@torch.no_grad()
def linked_push(Ts, X: torch.Tensor, return_type="T_X"):  # T_X, trajectory
    assert return_type in ["T_X", "trajectory"]
    tr_list = [X.clone().detach()]
    for T in Ts:
        T_X = T(tr_list[-1])
        tr_list.append(T_X)
    if return_type == "trajectory":
        return tr_list[1:]  # not contain X, list[tensor(batch, c, h, w)]
    if return_type == "T_X":
        XN = tr_list[-1].clone().detach()  # XN is tensor(batch, c, h, w)
        del tr_list
        gc.collect()
        torch.cuda.empty_cache()
        return XN


@torch.no_grad()
def sde_push(SDE, X0, return_type="XN"):  # XN, trajectory, all
    assert return_type in ["XN", "trajectory", "all"]
    freeze(SDE)
    trajectory, shifts, times = SDE(X0)
    if return_type == "all":
        return trajectory, shifts, times

    if return_type == "trajectory":
        del shifts, times
        return trajectory  # trajectory is tensor(batch, tr, c, h, w)

    if return_type == "XN":
        XN = trajectory[:, -1].clone().detach()  # (batch, c, h, w)
        del trajectory, shifts, times
        return XN


@torch.no_grad()
def linked_sde_push(SDEs, X0, return_type="XN"):  # XN, trajectory
    assert return_type in ["XN", "trajectory"]
    tr_list = [X0.clone().detach()]
    for sde in SDEs:
        XN = sde_push(sde, tr_list[-1], "XN")
        tr_list.append(XN)

    if return_type == "trajectory":
        return tr_list[1:]  # not contain X0, list[tensor(batch, c, h, w)]
    if return_type == "XN":
        XN = tr_list[-1].clone().detach()  # tensor(batch, c, h, w)
        del tr_list
        gc.collect()
        torch.cuda.empty_cache()
        return XN


# ========== dataloader FID =========== #
@torch.no_grad()
def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False, use_Y=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if use_Y:
                        batch = ((Y[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    else:
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()

                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


@torch.no_grad()
def get_pushed_loader_stats(
    T,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)
    freeze(T)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = (
                        T(X[start:end].type(torch.FloatTensor).to(device))
                        .add(1)
                        .mul(0.5)
                    )
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


@torch.no_grad()
def get_linked_pushed_loader_stats(
    Ts,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    X0 = X[start:end].type(torch.FloatTensor).to(device)
                    batch = linked_push(Ts, X0, return_type="T_X").add(1).mul(0.5)
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


@torch.no_grad()
def get_sde_pushed_loader_stats(
    SDE,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    X0 = X[start:end].type(torch.FloatTensor).to(device)
                    batch = sde_push(SDE, X0, return_type="XN").add(1).mul(0.5)
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


@torch.no_grad()
def get_linked_sde_pushed_loader_stats(
    SDEs,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    X0 = X[start:end].type(torch.FloatTensor).to(device)
                    batch = linked_sde_push(SDEs, X0, return_type="XN").add(1).mul(0.5)
                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


# ========== Some Metrics =========== #
@torch.no_grad()
def get_pushed_loader_metrics(
    T,
    loader,
    n_epochs=1,
    verbose=False,
    device="cuda",
    log_metrics=["LPIPS", "PSNR", "SSIM", "MSE", "MAE"],
):
    loss_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="sum"
    ).to(device)
    loss_psnr = PeakSignalNoiseRatio(data_range=(-1, 1), reduction="sum").to(device)
    loss_ssim = StructuralSimilarityIndexMeasure(
        data_range=(-1, 1), reduction="sum"
    ).to(device)
    loss_mse = MeanSquaredError().to(device)
    loss_mae = MeanAbsoluteError().to(device)

    metrics = dict(
        MSE=loss_mse, MAE=loss_mae, LPIPS=loss_lpips, PSNR=loss_psnr, SSIM=loss_ssim
    )
    results = dict({metric: 0.0 for metric in metrics.keys()})

    size = 0

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                X, Y = X.to(device), Y.to(device)
                size += X.size(0)
                TX = T(X)
                for metric, loss in metrics.items():
                    if metric in log_metrics:
                        results[metric] += loss(TX, Y)
    for metric, loss in metrics.items():
        results[metric] /= size

    gc.collect()
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def get_linked_pushed_loader_metrics(
    Ts,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    log_metrics=["LPIPS", "PSNR", "SSIM", "MSE", "MAE"],
):
    loss_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="sum"
    ).to(device)
    loss_psnr = PeakSignalNoiseRatio(data_range=(-1, 1), reduction="sum").to(device)
    loss_ssim = StructuralSimilarityIndexMeasure(
        data_range=(-1, 1), reduction="sum"
    ).to(device)
    loss_mse = MeanSquaredError().to(device)
    loss_mae = MeanAbsoluteError().to(device)

    metrics = dict(
        MSE=loss_mse, MAE=loss_mae, LPIPS=loss_lpips, PSNR=loss_psnr, SSIM=loss_ssim
    )
    results = dict({metric: 0.0 for metric in metrics.keys()})

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    size = 0
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                X, Y = X.to(device), Y.to(device)
                size += X.size(0)
                TX = linked_push(Ts, X, return_type="T_X")
                for metric, loss in metrics.items():
                    if metric in log_metrics:
                        results[metric] += loss(TX, Y)

    for metric, loss in metrics.items():
        results[metric] /= size

    gc.collect()
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def get_sde_pushed_loader_metrics(
    SDE,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    log_metrics=["LPIPS", "PSNR", "SSIM", "MSE", "MAE"],
):
    loss_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="sum"
    ).to(device)
    loss_psnr = PeakSignalNoiseRatio(data_range=(-1, 1), reduction="sum").to(device)
    loss_ssim = StructuralSimilarityIndexMeasure(
        data_range=(-1, 1), reduction="sum"
    ).to(device)
    loss_mse = MeanSquaredError().to(device)
    loss_mae = MeanAbsoluteError().to(device)

    metrics = dict(
        MSE=loss_mse, MAE=loss_mae, LPIPS=loss_lpips, PSNR=loss_psnr, SSIM=loss_ssim
    )
    results = dict({metric: 0.0 for metric in metrics.keys()})

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    size = 0
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                X, Y = X.to(device), Y.to(device)
                size += X.size(0)
                TX = sde_push(SDE, X, return_type="XN")
                for metric, loss in metrics.items():
                    if metric in log_metrics:
                        results[metric] += loss(TX, Y)

    for metric, loss in metrics.items():
        results[metric] /= size

    gc.collect()
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def get_linked_sde_pushed_loader_metrics(
    SDEs,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    log_metrics=["LPIPS", "PSNR", "SSIM", "MSE", "MAE"],
):
    loss_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="sum"
    ).to(device)
    loss_psnr = PeakSignalNoiseRatio(data_range=(-1, 1), reduction="sum").to(device)
    loss_ssim = StructuralSimilarityIndexMeasure(
        data_range=(-1, 1), reduction="sum"
    ).to(device)
    loss_mse = MeanSquaredError().to(device)
    loss_mae = MeanAbsoluteError().to(device)

    metrics = dict(
        MSE=loss_mse, MAE=loss_mae, LPIPS=loss_lpips, PSNR=loss_psnr, SSIM=loss_ssim
    )
    results = dict({metric: 0.0 for metric in metrics.keys()})

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )

    size = 0
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                X, Y = X.to(device), Y.to(device)
                size += X.size(0)
                TX = linked_sde_push(SDEs, X, return_type="XN")
                for metric, loss in metrics.items():
                    if metric in log_metrics:
                        results[metric] += loss(TX, Y)

    for metric, loss in metrics.items():
        results[metric] /= size

    gc.collect()
    torch.cuda.empty_cache()
    return results


# ================== Accuracy =================
@torch.no_grad()
def get_pushed_loader_accuracy(
    T,
    X_test_loader,
    classifier,
    batch_size=64,
    num_workers=0,
    device="cuda",
):
    correct = 0
    total = 0
    classifier.to(device)

    transport_results = []
    real_labels = []

    for X, labels in X_test_loader:
        X = X.to(device)
        real_labels.append(labels)
        XN = T(X)
        transport_results.append(XN)

    flat_transport_results = [item for sublist in transport_results for item in sublist]
    flat_real_labels = [y for sublist in real_labels for y in sublist]

    flat_transport_results = torch.stack(
        [torch.tensor(item) for item in flat_transport_results]
    )
    flat_real_labels = torch.tensor(flat_real_labels, dtype=torch.long)

    transport_dataset = torch.utils.data.TensorDataset(
        flat_transport_results, flat_real_labels
    )
    transport_loader = torch.utils.data.DataLoader(
        transport_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
    )

    for x, y in transport_loader:
        x, y = x.to(device), y.to(device)
        outputs = classifier(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network: {accuracy:.3f} %")
    return accuracy


@torch.no_grad()
def get_linked_pushed_loader_accuracy(
    Ts, X_test_loader, classifier, batch_size=64, num_workers=0, device="cuda"
):
    correct = 0
    total = 0
    classifier.to(device)

    transport_results = []
    real_labels = []

    for X, labels in X_test_loader:
        X = X.to(device)
        real_labels.append(labels)
        XN = linked_push(Ts, X, "X_T")
        transport_results.append(XN)

    flat_transport_results = [item for sublist in transport_results for item in sublist]
    flat_real_labels = [y for sublist in real_labels for y in sublist]

    flat_transport_results = torch.stack(
        [torch.tensor(item) for item in flat_transport_results]
    )
    flat_real_labels = torch.tensor(flat_real_labels, dtype=torch.long)

    transport_dataset = torch.utils.data.TensorDataset(
        flat_transport_results, flat_real_labels
    )
    transport_loader = torch.utils.data.DataLoader(
        transport_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for x, y in transport_loader:
        x, y = x.to(device), y.to(device)
        outputs = classifier(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network: {accuracy:.3f} %")
    return accuracy


@torch.no_grad()
def get_sde_pushed_loader_accuracy(
    SDE, X_test_loader, classifier, batch_size=64, num_workers=0, device="cuda"
):
    correct = 0
    total = 0
    classifier.to(device)

    transport_results = []
    real_labels = []

    for X, labels in X_test_loader:
        X = X.to(device)
        real_labels.append(labels)
        XN = sde_push(SDE, X, "XN")
        transport_results.append(XN)

    flat_transport_results = [item for sublist in transport_results for item in sublist]
    flat_real_labels = [y for sublist in real_labels for y in sublist]

    flat_transport_results = torch.stack(
        [torch.tensor(item) for item in flat_transport_results]
    )
    flat_real_labels = torch.tensor(flat_real_labels, dtype=torch.long)

    transport_dataset = torch.utils.data.TensorDataset(
        flat_transport_results, flat_real_labels
    )
    transport_loader = torch.utils.data.DataLoader(
        transport_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for x, y in transport_loader:
        x, y = x.to(device), y.to(device)
        outputs = classifier(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network: {accuracy:.3f} %")
    return accuracy


@torch.no_grad()
def get_linked_sde_pushed_loader_accuracy(
    SDEs, X_test_loader, classifier, batch_size=64, num_workers=0, device="cuda"
):
    correct = 0
    total = 0
    classifier.to(device)

    transport_results = []
    real_labels = []

    for X, labels in X_test_loader:
        X = X.to(device)
        real_labels.append(labels)
        XN = linked_sde_push(SDEs, X, "XN")
        transport_results.append(XN)

    flat_transport_results = [item for sublist in transport_results for item in sublist]
    flat_real_labels = [y for sublist in real_labels for y in sublist]

    flat_transport_results = torch.stack(
        [torch.tensor(item) for item in flat_transport_results]
    )
    flat_real_labels = torch.tensor(flat_real_labels, dtype=torch.long)

    transport_dataset = torch.utils.data.TensorDataset(
        flat_transport_results, flat_real_labels
    )
    transport_loader = torch.utils.data.DataLoader(
        transport_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for x, y in transport_loader:
        x, y = x.to(device), y.to(device)
        outputs = classifier(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network: {accuracy:.3f} %")
    return accuracy


# ================== Z =================
def get_Z_pushed_loader_stats(
    T,
    loader,
    ZC=1,
    Z_STD=0.1,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
    resnet=True,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)
    freeze(T)

    size = len(loader.dataset)
    pred_arr = []

    if verbose:
        display_id = display(
            f"Epoch 0/{n_epochs}: Processing batch 0/{len(loader)}", display_id=True
        )
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader):
                if verbose:
                    update_display(
                        f"Epoch {epoch+1}/{n_epochs}: Processing batch {step + 1}/{len(loader)}",
                        display_id=display_id.display_id,
                    )
                if not resnet:
                    Z = torch.randn(len(X), ZC, X.size(2), X.size(3)) * Z_STD
                    XZ = torch.cat([X, Z], dim=1)
                else:
                    Z = torch.randn(len(X), ZC, 1, 1) * Z_STD
                    XZ = (X, Z)
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if not resnet:
                        batch = (
                            T(XZ[start:end].type(torch.FloatTensor).to(device))
                            .add(1)
                            .mul(0.5)
                        )
                    else:
                        batch = (
                            T(
                                XZ[0][start:end].type(torch.FloatTensor).to(device),
                                XZ[1][start:end].type(torch.FloatTensor).to(device),
                            )
                            .add(1)
                            .mul(0.5)
                        )
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma
