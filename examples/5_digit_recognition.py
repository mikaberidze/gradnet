"""Attractive-repulsive pixel network memorizing MNIST digits."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gradnet.gradnet import GradNet

DATA_ROOT = Path.home() / ".mnist-data"
MNIST_ARCHIVE = DATA_ROOT / "mnist.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
OUTPUT = Path(__file__).with_name("digit_relaxation.mp4")
IMG_SHAPE = (28, 28)
NUM_PIXELS = IMG_SHAPE[0] * IMG_SHAPE[1]


def load_patterns(digits: Iterable[int] = range(5), per_digit: int = 50) -> tuple[np.ndarray, np.ndarray]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not MNIST_ARCHIVE.exists():
        with urlopen(MNIST_URL) as response, MNIST_ARCHIVE.open("wb") as fp:
            fp.write(response.read())

    with np.load(MNIST_ARCHIVE) as archive:
        images = archive["x_train"].reshape(-1, NUM_PIXELS).astype(np.float32) / 255.0
        labels = archive["y_train"].astype(int)

    wanted = {int(d): 0 for d in digits}
    data: List[np.ndarray] = []
    picked: List[int] = []
    for image, label in zip(images, labels):
        if label in wanted and wanted[label] < per_digit:
            data.append(image)
            picked.append(label)
            wanted[label] += 1
        if all(count >= per_digit for count in wanted.values()):
            break
    return np.stack(data), np.array(picked)


def to_binary(images: np.ndarray) -> np.ndarray:
    return np.where(images > 0.5, 1.0, -1.0).astype(np.float32)


def train_weights(
    patterns: np.ndarray,
    epochs: int = 200,
    lr: float = 0.05,
    dt: float = 0.25,
    leak: float = 0.4,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_patterns = torch.from_numpy(patterns).to(device)

    mask = torch.ones((NUM_PIXELS, NUM_PIXELS), dtype=torch_patterns.dtype, device=device)
    mask.fill_diagonal_(0)
    base_adj = torch.zeros((NUM_PIXELS, NUM_PIXELS), dtype=torch_patterns.dtype, device=device)

    gradnet = GradNet(
        num_nodes=NUM_PIXELS,
        budget=float(NUM_PIXELS),
        mask=mask,
        adj0=base_adj,
        delta_sign="free",
        final_sign="free",
        undirected=True,
        use_budget_up=False,
        rand_init_weights=True,
        cost_matrix=torch.ones_like(mask),
        cost_aggr_norm=2,
    ).to(device)

    diag_mask = mask
    optimizer = torch.optim.Adam(gradnet.parameters(), lr=lr)
    gradnet.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        adjacency = gradnet()
        adjacency = 0.5 * (adjacency + adjacency.T)
        adjacency = adjacency * diag_mask

        update = torch_patterns + dt * (torch.matmul(torch_patterns, adjacency.T) - leak * torch_patterns)
        relaxed = torch.clamp(update, -1.0, 1.0)
        loss = F.mse_loss(relaxed, torch_patterns) + weight_decay * adjacency.pow(2).mean()
        loss.backward()
        optimizer.step()

    gradnet.eval()
    with torch.no_grad():
        adjacency = gradnet()
        adjacency = 0.5 * (adjacency + adjacency.T)
        adjacency = adjacency * diag_mask
    return adjacency.cpu().numpy()


def ensure_imageio_ffmpeg():
    try:
        import imageio_ffmpeg  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The imageio-ffmpeg package is required to export MP4 animations. "
            "Install it with `pip install imageio-ffmpeg`."
        ) from exc
    return imageio_ffmpeg


def colorize(frame: np.ndarray) -> np.ndarray:
    """Map a single frame in [-1, 1] to an RGB image."""

    clipped = np.clip(frame, -1.0, 1.0)
    positive = np.clip(clipped, 0.0, None)
    negative = np.clip(-clipped, 0.0, None)
    neutral = 1.0 - np.minimum(1.0, np.abs(clipped))

    red = 0.25 + 0.75 * positive
    green = 0.25 + 0.75 * neutral
    blue = 0.25 + 0.75 * negative
    rgb = np.stack([red, green, blue], axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def save_animation(trajectories: list[np.ndarray], path: Path, fps: int = 10) -> None:
    imageio_ffmpeg = ensure_imageio_ffmpeg()
    width = IMG_SHAPE[1]
    height = IMG_SHAPE[0]
    writer = imageio_ffmpeg.write_frames(
        str(path), size=(width, height), fps=fps, macro_block_size=1
    )
    writer.send(None)
    try:
        spacer = np.zeros(IMG_SHAPE, dtype=np.float32)
        for traj in trajectories:
            for frame in traj:
                writer.send(colorize(frame))
            for _ in range(3):
                writer.send(colorize(spacer))
    finally:
        writer.close()


def relax(weights: np.ndarray, state: np.ndarray, steps: int = 40, dt: float = 0.25, leak: float = 0.4) -> np.ndarray:
    frames = [state.reshape(IMG_SHAPE)]
    current = state.copy()
    for _ in range(steps - 1):
        drive = weights @ current - leak * current
        current = np.clip(current + dt * drive, -1.0, 1.0)
        frames.append(current.reshape(IMG_SHAPE))
    return np.stack(frames)


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    raw_images, labels = load_patterns(per_digit=1)
    binary_patterns = to_binary(raw_images)
    weights = train_weights(binary_patterns)

    exemplars = [binary_patterns[np.where(labels == digit)[0][0]] for digit in range(5)]
    noisy = [np.clip(pat + 0.4 * np.random.randn(NUM_PIXELS), -1.0, 1.0) for pat in exemplars]
    trajectories = [relax(weights, start) for start in noisy]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_animation(trajectories, OUTPUT)


if __name__ == "__main__":
    main()
