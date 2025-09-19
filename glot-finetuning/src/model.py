"""
This file defines the Bi-Encoder model architecture used for retrieval.
It includes the core model class and the mean pooling utility function.
"""
import torch
from torch import nn
from transformers import AutoModel

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs mean pooling on token embeddings, correctly handling padding.

    It computes the average of token embeddings, weighted by their attention
    mask to exclude padding tokens from the calculation.

    Args:
        token_embeddings (torch.Tensor): The raw token embeddings from the model.
                                         Shape: (batch_size, seq_len, hidden_dim).
        attention_mask (torch.Tensor): The attention mask for the input tokens.
                                       Shape: (batch_size, seq_len).

    Returns:
        torch.Tensor: The pooled sentence embedding. Shape: (batch_size, hidden_dim).
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class BiEncoderRetriever(nn.Module):
    """
    A bi-encoder model that uses a shared transformer to encode both questions
    and passages into a dense vector space.

    This architecture is efficient for retrieval, as passage embeddings can be
    pre-computed and stored in an index for fast look-up.
    """
    def __init__(self, model_name: str):
        """
        Initializes the BiEncoderRetriever.

        Args:
            model_name (str): The name or path of a Hugging Face transformer model
                              to be used as the encoder (e.g., 'cis-lmu/glot500-base').
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute embeddings for a batch of texts.

        Args:
            input_ids (torch.Tensor): A batch of token IDs.
            attention_mask (torch.Tensor): The corresponding attention masks.

        Returns:
            torch.Tensor: A tensor of sentence embeddings, created by mean-pooling
                          the token embeddings from the last hidden layer.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
        return embedding
