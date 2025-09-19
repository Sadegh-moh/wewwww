"""
Wrapper classes for CLIP vision and text models.

This module provides `FaTextEncoder` and `FaVisionEncoder`, which are
convenience wrappers around the Hugging Face Transformers library. They handle
the loading of pre-trained models and processors, and provide simple `encode`
methods to get embeddings for text and images.
"""
from typing import List

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModel,
)


class FaTextEncoder:
    """
    A wrapper for the CLIP-based text model from Hugging Face.

    This class handles tokenization, forward pass through the model, and
    L2 normalization of the output embeddings. It is the component whose
    weights will be updated during the fine-tuning process.
    """

    def __init__(self, model_name_or_path: str, device: torch.device, max_len: int):
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.max_len = max_len
        self.model.eval()

    def _forward(self, texts: List[str]) -> torch.Tensor:
        """Performs a forward pass and returns normalized embeddings."""
        toks = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(
            self.device)
        out = self.model(**toks)

        # Use pooler output, with a fallback to mean pooling
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            x = out.pooler_output
        else:
            last = out.last_hidden_state
            mask = toks.attention_mask.unsqueeze(-1)
            x = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        x = x / x.norm(p=2, dim=1, keepdim=True)
        return x

    @torch.no_grad()
    def encode_numpy(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encodes a list of texts into a numpy array of embeddings."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            x = self._forward(batch_texts).detach().cpu().numpy()
            vecs.append(x)
        return np.vstack(vecs).astype(np.float32)

    def train(self):
        self.model.train(True)

    def eval(self):
        self.model.train(False)

    def parameters(self):
        return self.model.parameters()


class FaVisionEncoder:
    """
    A wrapper for the CLIP vision model from Hugging Face.

    This class handles the preprocessing of images and computes their
    embeddings. The weights of this model are kept frozen during the
    fine-tuning process, serving as a stable target for the text encoder.
    """

    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.proc = CLIPImageProcessor.from_pretrained(model_name)
        self.model.eval()
        # Freeze the vision model's parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_many_chunked(self, imgs: List[Image.Image], chunk_size: int = 16) -> torch.Tensor:
        """Encodes a list of images into a tensor of embeddings, processing in chunks."""
        outs = []
        for i in range(0, len(imgs), chunk_size):
            sub_imgs = [ImageOps.exif_transpose(im).convert("RGB") for im in imgs[i:i + chunk_size]]
            batch = self.proc(images=sub_imgs, return_tensors="pt").to(self.device)
            out = self.model(**batch)

            # Use pooler output, with a fallback to CLS token
            v = out.pooler_output if hasattr(out,
                                             "pooler_output") and out.pooler_output is not None else out.last_hidden_state[
                                                                                                     :, 0]
            v = v / v.norm(p=2, dim=1, keepdim=True)
            outs.append(v)
        return torch.cat(outs, dim=0)
