"""
Centralized configuration for the CLIP fine-tuning project.

This file contains a dataclass `Config` that holds all the necessary
parameters for data paths, model identifiers, and training hyperparameters.
Modifying this file is the primary way to adjust the training setup.
"""
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration class for the fine-tuning process.
    """
    # --- Data Paths ---
    # Path to the root directory containing the raw food image folders and JSON files.
    base_data_root: str = "/kaggle/input/food-dataset-nlp/images_passages"
    # Directory where processed data (docstore, training CSV) will be saved.
    processed_data_dir: str = "data/"
    # Directory to save the final fine-tuned model and its index.
    output_dir: str = "models/"

    # --- Model Identifiers ---
    # Hugging Face model ID for the pre-trained text encoder to be fine-tuned.
    mclip_text_model: str = "SajjadAyoubi/clip-fa-text"
    # Hugging Face model ID for the pre-trained vision encoder (weights are frozen).
    clip_vision_model: str = "SajjadAyoubi/clip-fa-vision"
    
    # --- Fine-Tuning Hyperparameters ---
    max_text_len: int = 512
    batch_size: int = 16
    epochs: int = 4
    lr_text: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    
    # --- System Configuration ---
    num_workers: int = 2
    seed: int = 42
