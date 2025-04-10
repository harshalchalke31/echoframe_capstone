import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from pathlib import Path

# ------------------- UTILS -------------------

def load_video(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    cap = cv2.VideoCapture(str(filename))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[i] = frame
    cap.release()
    return v

def get_optimum_set_of_frame_indexes(length, period, target_index, n_frames):
    candidates = [target_index]
    for i in range(length - 1):
        if i % 2 == 0:
            candidates.insert(0, candidates[0] - period)
        else:
            candidates.append(candidates[-1] + period)
    candidates = [i for i in candidates if i < n_frames]
    while len(candidates) < length:
        candidates.insert(0, candidates[0] - period)
    selected_frames = []
    for c in reversed(candidates):
        if c < 0:
            selected_frames.append(selected_frames[-1] + period)
        else:
            selected_frames.insert(0, c)
    return selected_frames

# ------------------- DATASET -------------------

class EchoNetDataset(VisionDataset):
    def __init__(self, root, split="train", length=16, period=2, transform=None):
        super().__init__(root)
        self.length = length
        self.period = period
        self.transform = transform
        self.root = Path(root)
        
        csv_path = self.root / "FileList.csv"
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["Split"].str.lower() == split.lower()]
        self.filenames = [self.root / "Videos" / f"{fn}.avi" for fn in self.data["FileName"]]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        video = load_video(self.filenames[idx])  # (T, H, W, 3)
        T, H, W, C = video.shape

        if T < self.length * self.period:
            padding = np.zeros((self.length * self.period - T, H, W, C), dtype=video.dtype)
            video = np.concatenate([video, padding], axis=0)
            T = video.shape[0]

        # Select center frame and make sure it is included
        center = T // 2
        frame_idxs = get_optimum_set_of_frame_indexes(self.length, self.period, center, T)
        video = video[frame_idxs]  # (length, H, W, 3)

        # Optional transform
        if self.transform:
            video = np.stack([self.transform(frame) for frame in video])

        # Transpose to [C, D, H, W]
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0   # Normalize here if needed

        return video  # [C, D, H, W]

# ------------------- DATALOADER -------------------

def preprocess_image(x):
    """Preprocessing logic to resize the image."""
    return cv2.resize(x, (112, 112))  # Replace lambda with a named function

def get_echonet_dataloader(data_path, batch_size=4, n_frames=16, period=2, split="train", num_workers=4):
    dataset = EchoNetDataset(
        root=data_path,
        split=split,
        length=n_frames,
        period=period,
        transform=preprocess_image  # Use the named function instead of lambda
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader

# ------------------- EXAMPLE -------------------

if __name__ == "__main__":
    data_path = r"C:\Projects\python\echoframe\data\EchoNet-Dynamic\EchoNet-Dynamic"
    loader = get_echonet_dataloader(data_path)

    for batch in loader:
        print("Batch shape:", batch.shape)  # Expecting [B, C, D, H, W]
        break
