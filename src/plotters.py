import gc

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tools import freeze, linked_push, linked_sde_push, sde_push


# ========= GNOT push =========
@torch.no_grad()
def plot_pushed_images(X, Y, T, gray=False):
    n_row, n_col = 3, int(X.shape[0])

    T_X = T(X)
    imgs = (
        torch.cat([X, T_X, Y])
        .to("cpu")
        .permute(0, 2, 3, 1)
        .mul(0.5)
        .add(0.5)
        .numpy()
        .clip(0, 1)
    )

    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5 * n_col, 1.5 * n_row), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if gray:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    axes[1, 0].set_ylabel("T(X)", fontsize=24)
    axes[2, 0].set_ylabel("Y", fontsize=24)
    fig.tight_layout(pad=0.001)
    return fig, axes


@torch.no_grad()
def plot_pushed_random_images(X_sampler, Y_sampler, T, plot_n_samples=10, gray=False):
    X = X_sampler.sample(plot_n_samples)
    Y = Y_sampler.sample(plot_n_samples)
    return plot_pushed_images(X, Y, T, gray)


@torch.no_grad()
def plot_pushed_random_paired_images(XY_sampler, T, plot_n_samples=10, gray=False):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_pushed_images(X, Y, T, gray)


@torch.no_grad()
def plot_pushed_random_class_images(XY_sampler, T, plot_n_samples=10, gray=False):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_pushed_images(X.flatten(0, 1), Y.flatten(0, 1), T, gray)


# ========= DNOT(link GNOT) push =========
@torch.no_grad()
def plot_linked_pushed_images(X, Y, Ts, gray=False, plot_trajectory=True):
    n_row = len(Ts) + 2 if plot_trajectory else 3
    n_col = int(X.shape[0])

    if plot_trajectory:
        tr_list = linked_push(Ts, X, return_type="trajectory")
        tr_list.append(Y)
        imgs = (
            torch.cat(tr_list)
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )
    else:
        T_X = linked_push(Ts, X, return_type="T_X")
        # print(f"[Debug] {T_X.shape=}")
        imgs = (
            torch.cat([X, T_X, Y])
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )

    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5 * n_col, 1.5 * n_row), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if gray:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    if plot_trajectory:
        axes[0, 0].set_ylabel("X", fontsize=24)
        for i in range(1, n_row - 1):
            axes[i, 0].set_ylabel(f"$T_{i}(X)$", fontsize=24)
        axes[(n_row - 1), 0].set_ylabel("Y", fontsize=24)
    else:
        axes[0, 0].set_ylabel("X", fontsize=24)
        axes[1, 0].set_ylabel("T(X)", fontsize=24)
        axes[2, 0].set_ylabel("Y", fontsize=24)
    fig.tight_layout(pad=0.001)
    return fig, axes


@torch.no_grad()
def plot_linked_pushed_random_images(
    X_sampler, Y_sampler, Ts, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X = X_sampler.sample(plot_n_samples)
    Y = Y_sampler.sample(plot_n_samples)
    return plot_linked_pushed_images(X, Y, Ts, gray, plot_trajectory)


@torch.no_grad()
def plot_linked_pushed_random_paired_images(
    XY_sampler, Ts, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_linked_pushed_images(X, Y, Ts, gray, plot_trajectory)


@torch.no_grad()
def plot_linked_pushed_random_class_images(
    XY_sampler, Ts, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_linked_pushed_images(
        X.flatten(0, 1), Y.flatten(0, 1), Ts, gray, plot_trajectory
    )


# ========= ENOT(SDE) push =========


# ==== need replace ====
def plot_fixed_sde_trajectories(X, Y, T, n_steps_to_show=10, n_steps=10, gray=False):
    freeze(T)

    trajectory_index_step_size = n_steps // n_steps_to_show
    with torch.no_grad():
        trajectory = T(X)[0]
        T_X = torch.cat(
            (trajectory[:, ::trajectory_index_step_size], trajectory[:, -1][:, None]),
            dim=1,
        )
        T_X = torch.transpose(T_X, 0, 1)[1:]

        c, h, w = X.shape[1:]
        imgs = (
            torch.cat([X[None, :], T_X, Y[None, :]])
            .reshape(-1, c, h, w)
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )

    rows_with_steps = np.arange(n_steps + 1)[::trajectory_index_step_size].shape[0]
    fig, axes = plt.subplots(2 + rows_with_steps, 10, figsize=(15, 20), dpi=150)

    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    for i in range(rows_with_steps):
        axes[i + 1, 0].set_ylabel(
            f"T(X)_{(i+1)*trajectory_index_step_size}", fontsize=24
        )
    axes[-2, 0].set_ylabel("T(X)", fontsize=24)
    axes[-1, 0].set_ylabel("Y", fontsize=24)

    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache()
    gc.collect()
    return fig, axes


def plot_random_sde_trajectories(
    X_sampler, Y_sampler, T, n_steps_to_show=10, n_steps=10, gray=False
):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)

    return plot_fixed_sde_trajectories(X, Y, T, n_steps_to_show, n_steps, gray)


def plot_several_fixed_sde_trajectories(
    X, Y, T, n_steps_to_show=10, n_steps=10, gray=False
):
    freeze(T)

    n_trajectories = 3
    X = X[:4]
    Y = Y[:4]

    trajectory_index_step_size = n_steps // n_steps_to_show
    with torch.no_grad():
        trajectory = []
        for i in range(n_trajectories):
            trajectory.append(T(X)[0])

        trajectory = torch.stack(trajectory, dim=1)
        trajectory = trajectory.reshape(
            trajectory.shape[0] * trajectory.shape[1],
            trajectory.shape[2],
            trajectory.shape[3],
            trajectory.shape[4],
            trajectory.shape[5],
        )

        T_X = torch.cat(
            (trajectory[:, ::trajectory_index_step_size], trajectory[:, -1][:, None]),
            dim=1,
        )
        T_X = torch.transpose(T_X, 0, 1)[1:]

        c, h, w = X.shape[1:]
        imgs = (
            torch.cat(
                [
                    X.repeat_interleave(3, dim=0)[None, :],
                    T_X,
                    Y.repeat_interleave(3, dim=0)[None, :],
                ]
            )
            .reshape(-1, c, h, w)
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )

    rows_with_steps = np.arange(n_steps + 1)[::trajectory_index_step_size].shape[0]
    fig, axes = plt.subplots(2 + rows_with_steps, 12, figsize=(15, 20), dpi=150)

    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    for i in range(rows_with_steps):
        axes[i + 1, 0].set_ylabel(
            f"T(X)_{(i+1)*trajectory_index_step_size}", fontsize=24
        )
    axes[-2, 0].set_ylabel("T(X)", fontsize=24)
    axes[-1, 0].set_ylabel("Y", fontsize=24)

    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache()
    gc.collect()
    return fig, axes


def plot_several_random_sde_trajectories(
    X_sampler, Y_sampler, T, n_steps_to_show=10, n_steps=10, gray=False
):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)

    return plot_several_fixed_sde_trajectories(X, Y, T, n_steps_to_show, n_steps, gray)


# ==============================


# TODO: ENOT with trajectory
# @torch.no_grad()
# def plot_sde_pushed_images(X, Y, SDE, gray=False, plot_trajectory=True):
#     pass
@torch.no_grad()
def plot_sde_pushed_images(X, Y, SDE, gray=False):
    n_row, n_col = 3, int(X.shape[0])

    T_X = sde_push(SDE, X, return_type="XN")
    imgs = (
        torch.cat([X, T_X, Y])
        .to("cpu")
        .permute(0, 2, 3, 1)
        .mul(0.5)
        .add(0.5)
        .numpy()
        .clip(0, 1)
    )

    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5 * n_col, 1.5 * n_row), dpi=150)

    for i, ax in enumerate(axes.flatten()):
        if gray:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    axes[1, 0].set_ylabel("T(X)", fontsize=24)
    axes[2, 0].set_ylabel("Y", fontsize=24)
    fig.tight_layout(pad=0.001)
    return fig, axes


@torch.no_grad()
def plot_sde_pushed_random_images(
    X_sampler, Y_sampler, SDE, plot_n_samples=10, gray=False
):
    X = X_sampler.sample(plot_n_samples)
    Y = Y_sampler.sample(plot_n_samples)
    return plot_sde_pushed_images(X, Y, SDE, gray)


@torch.no_grad()
def plot_sde_pushed_random_paired_images(
    XY_sampler, SDE, plot_n_samples=10, gray=False
):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_sde_pushed_images(X, Y, SDE, gray=gray)


@torch.no_grad()
def plot_sde_pushed_random_class_images(XY_sampler, SDE, plot_n_samples=10, gray=False):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_sde_pushed_images(X.flatten(0, 1), Y.flatten(0, 1), SDE, gray)


# ======================= DENOT(link SDE) push ====================
@torch.no_grad()
def plot_linked_sde_pushed_images(X, Y, SDEs, gray=False, plot_trajectory=True):
    n_row = len(SDEs) + 2 if plot_trajectory else 3
    n_col = int(X.shape[0])

    if plot_trajectory:
        tr_list = linked_sde_push(SDEs, X, return_type="trajectory")
        tr_list.append(Y)
        imgs = (
            torch.cat(tr_list)
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )
    else:
        T_X = linked_sde_push(SDEs, X, return_type="XN")
        # print(f"[Debug] {T_X.shape=}")
        imgs = (
            torch.cat([X, T_X, Y])
            .to("cpu")
            .permute(0, 2, 3, 1)
            .mul(0.5)
            .add(0.5)
            .numpy()
            .clip(0, 1)
        )

    fig, axes = plt.subplots(n_row, n_col, figsize=(1.5 * n_col, 1.5 * n_row), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if gray:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    if plot_trajectory:
        axes[0, 0].set_ylabel("X", fontsize=24)
        for i in range(1, n_row - 1):
            axes[i, 0].set_ylabel(f"$T_{i}(X)$", fontsize=24)
        axes[(n_row - 1), 0].set_ylabel("Y", fontsize=24)
    else:
        axes[0, 0].set_ylabel("X", fontsize=24)
        axes[1, 0].set_ylabel("T(X)", fontsize=24)
        axes[2, 0].set_ylabel("Y", fontsize=24)
    fig.tight_layout(pad=0.001)
    return fig, axes


@torch.no_grad()
def plot_linked_sde_pushed_random_images(
    X_sampler, Y_sampler, SDEs, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X = X_sampler.sample(plot_n_samples)
    Y = Y_sampler.sample(plot_n_samples)
    return plot_linked_sde_pushed_images(X, Y, SDEs, gray, plot_trajectory)


@torch.no_grad()
def plot_linked_sde_pushed_random_paired_images(
    XY_sampler, SDEs, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_linked_sde_pushed_images(X, Y, SDEs, gray, plot_trajectory)


@torch.no_grad()
def plot_linked_sde_pushed_random_class_images(
    XY_sampler, SDEs, plot_n_samples=10, gray=False, plot_trajectory=True
):
    X, Y = XY_sampler.sample(plot_n_samples)
    return plot_linked_sde_pushed_images(
        X.flatten(0, 1), Y.flatten(0, 1), SDEs, gray, plot_trajectory
    )


# ======================= TODO: with Z ====================
def plot_Z_images(XZ, Y, T, resnet=False, gray=False):
    freeze(T)
    with torch.no_grad():
        if not resnet:
            T_XZ = (
                T(XZ.flatten(start_dim=0, end_dim=1))
                .permute(1, 2, 3, 0)
                .reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4)
                .permute(4, 3, 0, 1, 2)
                .flatten(start_dim=0, end_dim=1)
            )
        else:
            T_XZ = (
                T(
                    *(
                        XZ[0].flatten(start_dim=0, end_dim=1),
                        XZ[1].flatten(start_dim=0, end_dim=1),
                    )
                )
                .permute(1, 2, 3, 0)
                .reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4)
                .permute(4, 3, 0, 1, 2)
                .flatten(start_dim=0, end_dim=1)
            )
        if not resnet:
            imgs = (
                torch.cat([XZ[:, 0, : Y.shape[1]], T_XZ, Y])
                .to("cpu")
                .permute(0, 2, 3, 1)
                .mul(0.5)
                .add(0.5)
                .numpy()
                .clip(0, 1)
            )
        else:
            imgs = (
                torch.cat([XZ[0][:, 0], T_XZ, Y])
                .to("cpu")
                .permute(0, 2, 3, 1)
                .mul(0.5)
                .add(0.5)
                .numpy()
                .clip(0, 1)
            )

    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if gray:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])

    axes[0, 0].set_ylabel("X", fontsize=24)
    for i in range(4):
        axes[i + 1, 0].set_ylabel("T(X,Z)", fontsize=24)
    axes[-1, 0].set_ylabel("Y", fontsize=24)

    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache()
    gc.collect()
    return fig, axes


def plot_random_Z_images(X_sampler, ZC, Z_STD, Y_sampler, T, resnet=False, gray=False):
    X = X_sampler.sample(10)[:, None].repeat(1, 4, 1, 1, 1)
    with torch.no_grad():
        if not resnet:
            Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device="cuda") * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        else:
            Z = torch.randn(10, 4, ZC, 1, 1, device="cuda") * Z_STD
            XZ = (
                X,
                Z,
            )
    Y = Y_sampler.sample(10)
    return plot_Z_images(XZ, Y, T, resnet=resnet, gray=gray)


def plot_random_paired_Z_images(XY_sampler, ZC, Z_STD, T, resnet=False, gray=False):
    X, Y = XY_sampler.sample(10)
    X = X[:, None].repeat(1, 4, 1, 1, 1)
    with torch.no_grad():
        if not resnet:
            Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device="cuda") * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        else:
            Z = torch.randn(10, 4, ZC, 1, 1, device="cuda") * Z_STD
            XZ = (
                X,
                Z,
            )
    return plot_Z_images(XZ, Y, T, resnet=resnet, gray=gray)
