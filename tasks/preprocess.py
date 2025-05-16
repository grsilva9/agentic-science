# tasks/preprocess.py

from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

from schemas import PreprocessInput, PreprocessOutput

def preprocess_task(input: PreprocessInput) -> PreprocessOutput:
    """
    1. Loads all .png images from input.raw_data_path (a folder).
    2. Resizes each to input.resize.
    3. Converts to torch.Tensor.
    4. Stacks into one tensor, shuffles and splits according to input.split_ratio.
    5. Wraps each split in a TensorDataset + DataLoader.
    6. Saves the two DataLoaders to disk.
    7. Returns the two paths.
    """
    # 1. Find & load all images
    img_dir: Path = input.raw_data_path
    tensors = []
    transform = T.ToTensor()
    for img_fp in img_dir.glob("*.png"):
        img = Image.open(img_fp).convert("L")
        img = img.resize(input.resize)
        tensors.append(transform(img))
    data = torch.stack(tensors)  # shape [N, C, H, W]

    # 2. Shuffle & split
    N = data.size(0)
    perm = torch.randperm(N)
    split_idx = int(N * input.split_ratio)
    train_data = data[perm[:split_idx]]
    test_data  = data[perm[split_idx:]]

    # 3. Build datasets & loaders
    train_ds = TensorDataset(train_data)
    test_ds  = TensorDataset(test_data)
    train_loader = DataLoader(train_ds, batch_size= input.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size= input.batch_size, shuffle=False)

    # 4. Save loaders
    out_dir = img_dir.parent / input.preprocessed_path
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_loader.pt"
    test_path  = out_dir / "test_loader.pt"
    torch.save(train_loader, str(train_path))
    torch.save(test_loader,  str(test_path))

    # 5. Return the two paths
    return PreprocessOutput(split_train_test=(train_path, test_path))
