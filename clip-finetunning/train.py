"""
Main script to run the CLIP fine-tuning pipeline.

This script serves as the entry point for the entire process. It handles:
1. Parsing command-line arguments to override default configurations.
2. Running the initial data preparation step to create the docstore and training CSV.
3. Setting up the dataset and dataloader.
4. Initializing and running the `FaCLIPTextTrainer`.
5. Saving the final fine-tuned model.
6. Rebuilding the FAISS index for the newly trained text encoder.
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import faiss

from config import Config
from utils.data_prep import prep_dataset
from src.dataset import ImageTextPairs, collate_pil_text_gid
from src.encoders import FaTextEncoder
from src.trainer import FaCLIPTextTrainer
from torch.utils.data import DataLoader


def rebuild_text_index(model_dir: str, docstore_path: str, index_out_path: str, cfg: Config):
    """
    Creates a FAISS index from the passages in the docstore using the fine-tuned text encoder.

    Args:
        model_dir (str): Directory of the fine-tuned text encoder.
        docstore_path (str): Path to the docstore.parquet file.
        index_out_path (str): Path to save the new FAISS index.
        cfg (Config): The configuration object.
    """
    print("Rebuilding FAISS index with the fine-tuned model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FaTextEncoder(model_dir, device, cfg.max_text_len)
    df_docstore = pd.read_parquet(docstore_path)

    texts = df_docstore["passage_text"].astype(str).tolist()
    vectors = encoder.encode_numpy(texts, batch_size=cfg.batch_size * 2)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))

    os.makedirs(os.path.dirname(index_out_path), exist_ok=True)
    faiss.write_index(index, index_out_path)
    print(f"New FAISS index saved to {index_out_path}")


def run_finetuning(cfg: Config):
    """
    Main function to orchestrate the fine-tuning and index rebuilding process.

    Args:
        cfg (Config): The configuration object containing all parameters.
    """
    # Set seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 1. Prepare the dataset
    train_csv_path, docstore_path = prep_dataset(cfg.base_data_root, cfg.processed_data_dir)

    # 2. Set up DataLoader
    df_train = pd.read_csv(train_csv_path)
    train_dataset = ImageTextPairs(df_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_pil_text_gid
    )

    # 3. Initialize and run the trainer
    trainer = FaCLIPTextTrainer(
        cfg=cfg,
        text_model=cfg.mclip_text_model,
        vision_model=cfg.clip_vision_model
    )
    if cfg.epochs > 0:
        trainer.train_loop(train_loader)
    else:
        print("Epochs set to 0, skipping training loop.")

    # 4. Save the model
    output_model_dir = os.path.join(cfg.output_dir, "finetuned_mclip_text")
    trainer.save_text_model(output_model_dir)

    # 5. Rebuild the search index with the new model
    output_index_path = os.path.join(cfg.output_dir, "I_clip_text_fa.index")
    rebuild_text_index(output_model_dir, docstore_path, output_index_path, cfg)

    print("\nFine-tuning process finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a CLIP text encoder.")
    parser.add_argument("--data_root", type=str, help="Override path to the raw data root directory.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Override the training batch size.")
    parser.add_argument("--learning_rate", type=float, help="Override the learning rate.")
    parser.add_argument("--output_model_dir", type=str, help="Override the directory to save the final model.")

    args = parser.parse_args()

    # Initialize config and update with any command-line arguments
    config = Config()
    if args.data_root:
        config.base_data_root = args.data_root
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.lr_text = args.learning_rate
    if args.output_model_dir:
        config.output_dir = args.output_model_dir

    run_finetuning(config)