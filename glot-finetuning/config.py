"""
This file contains the central configuration for the Glot-500 fine-tuning project.
All hyperparameters, model names, and file paths are defined here for easy access and modification.
"""

from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """
    Configuration class for the Glot fine-tuning pipeline.
    """
    # --- Model and Tokenizer ---
    base_model_name: str = "cis-lmu/glot500-base"

    # --- File Paths ---
    training_data_path: str = "/kaggle/input/training-data/training_data.json"
    output_dir: str = "./retriever_ckpt"

    # --- Masked Language Modeling (MLM) Phase ---
    do_mlm: bool = True
    mlm_epochs: int = 2
    mlm_batch_size: int = 1

    # --- Contrastive Training Phase ---
    contrastive_epochs: int = 6
    contrastive_batch_size: int = 8

    # --- General Training Parameters ---
    learning_rate: float = 2e-5
    max_seq_length: int = 500
    warmup_steps: int = 1000

    # --- Environment ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"