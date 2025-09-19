"""
PyTorch Dataset class for loading image-text pairs.

This module defines the `ImageTextPairs` class, which is a custom
`torch.utils.data.Dataset`. It takes a pandas DataFrame as input and
is responsible for loading an image from a file path and returning it
along with its corresponding text description and a group ID for
contrastive learning.
"""
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset


class ImageTextPairs(Dataset):
    """
    A PyTorch Dataset to handle loading of image-text pairs from a DataFrame.

    This class reads a DataFrame containing file paths to images and their
    associated text. For each item, it opens the image, converts it to RGB,
    and returns the image, text, and a numerical group ID.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): A DataFrame with at least 'image_path' and 'text' columns.
                               It also uses 'canonical_dish' for grouping if available.
        """
        assert {"image_path", "text"}.issubset(df.columns)
        self.df = df.reset_index(drop=True)

        # Create unique numerical group IDs for contrastive loss
        if "canonical_dish" in self.df.columns:
            groups = self.df["canonical_dish"].astype("category").cat.codes.values
        else:
            # Fallback if canonical_dish is not present
            groups = pd.factorize(self.df['text'])[0]
        self.gids = groups

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, i: int) -> tuple:
        """
        Retrieves the i-th sample from the dataset.

        Args:
            i (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (PIL.Image, str, int) for the
                   image, text, and group ID respectively.
        """
        row = self.df.iloc[i]
        path = row["image_path"]
        text = str(row["text"])
        gid = int(self.gids[i])

        try:
            img = Image.open(path)
            # Ensure image is in RGB format and orientation is corrected
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {path}. Returning a placeholder. Error: {e}")
            img = Image.new('RGB', (224, 224), (255, 255, 255))

        return img, text, gid


def collate_pil_text_gid(batch):
    """Custom collate function for the DataLoader to handle PIL images."""
    imgs, texts, gids = zip(*batch)
    return list(imgs), list(texts), torch.as_tensor(gids, dtype=torch.long)
