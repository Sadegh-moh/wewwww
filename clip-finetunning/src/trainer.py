"""
The main trainer class for fine-tuning the CLIP text encoder.

This module contains the `FaCLIPTextTrainer`, which orchestrates the entire
fine-tuning process. It initializes the text and vision encoders, sets up
the optimizer and learning rate scheduler, and implements the main training
loop, including the calculation of the multi-positive contrastive loss.
"""
import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .encoders import FaTextEncoder, FaVisionEncoder
from ..config import Config


class FaCLIPTextTrainer:
    """
    Orchestrates the fine-tuning of the text encoder using a contrastive loss.

    This trainer pairs a trainable text encoder with a frozen vision encoder.
    It uses a multi-positive contrastive loss, where multiple images of the same
    dish are considered positive pairs for a given text description.
    """

    def __init__(self, cfg: Config, text_model: str, vision_model: str):
        """
        Initializes the trainer, models, and optimizer.

        Args:
            cfg (Config): The global configuration object.
            text_model (str): HF model ID for the text encoder.
            vision_model (str): HF model ID for the vision encoder.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_enc = FaTextEncoder(text_model, self.device, cfg.max_text_len)
        self.text_enc.train()

        self.vision_enc = FaVisionEncoder(vision_model, self.device)

        # Learnable temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=self.device))

        self.optim = torch.optim.AdamW(
            list(self.text_enc.parameters()) + [self.logit_scale],
            lr=cfg.lr_text,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = None

    def _multi_pos_loss(self, logits: torch.Tensor, gids: torch.Tensor) -> torch.Tensor:
        """
        Calculates the multi-positive contrastive loss.

        This loss function considers all samples with the same group ID (gid)
        as positive pairs, and all others as negative pairs.

        Args:
            logits (torch.Tensor): A tensor of similarity scores (logits)
                                   between all pairs in the batch.
            gids (torch.Tensor): A tensor of group IDs for each sample.

        Returns:
            torch.Tensor: The computed loss value.
        """
        B = gids.size(0)
        # Create a mask where (i, j) is True if sample i and j are positive pairs
        pos_mask = (gids.view(B, 1) == gids.view(1, B))

        log_softmax = torch.log_softmax(logits, dim=1)

        # Use the mask to select the logits for positive pairs
        # The loss is the negative log-likelihood of correctly classifying all positive pairs.
        loss = -torch.logsumexp(log_softmax.where(pos_mask, -torch.inf), dim=1).mean()

        return loss

    def train_loop(self, dl: DataLoader):
        """
        The main training loop for the fine-tuning process.

        Iterates over the DataLoader for the specified number of epochs,
        computes the loss, and updates the model weights.

        Args:
            dl (DataLoader): The DataLoader providing training batches.
        """
        num_training_steps = self.cfg.epochs * len(dl)

        # Setup learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.cfg.warmup_steps:
                return float(step) / float(max(1, self.cfg.warmup_steps))
            progress = float(step - self.cfg.warmup_steps) / float(max(1, num_training_steps - self.cfg.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        print(f"Starting training for {self.cfg.epochs} epochs on {self.device}...")
        for epoch in range(1, self.cfg.epochs + 1):
            pbar = tqdm(dl, desc=f"Epoch {epoch}/{self.cfg.epochs}")
            for imgs, texts, gids in pbar:
                gids = gids.to(self.device)

                # Vision embeddings are computed with no gradients and in chunks for memory efficiency
                img_emb = self.vision_enc.encode_many_chunked(imgs, chunk_size=self.cfg.batch_size // 2)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    text_emb = self.text_enc._forward(texts)

                    scale = self.logit_scale.exp()
                    logits = scale * (text_emb @ img_emb.T)

                    # Symmetrical loss
                    loss = 0.5 * (self._multi_pos_loss(logits, gids) + self._multi_pos_loss(logits.T, gids))

                self.optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.text_enc.parameters(), 1.0)
                scaler.step(self.optim)
                scaler.update()
                self.scheduler.step()

                pbar.set_postfix(loss=loss.item(), lr=self.optim.param_groups[0]['lr'])

    def save_text_model(self, out_dir: str):
        """Saves the fine-tuned text encoder and its tokenizer to a directory."""
        print(f"Saving fine-tuned model to {out_dir}...")
        os.makedirs(out_dir, exist_ok=True)
        self.text_enc.model.save_pretrained(out_dir)
        self.text_enc.tok.save_pretrained(out_dir)
        print("Model saved successfully.")
