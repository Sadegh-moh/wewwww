"""
This file defines the PyTorch Dataset classes used for loading and preparing
the data for both the Masked Language Modeling (MLM) and the contrastive
training phases.
"""
import random
import torch
from torch.utils.data import Dataset


class QAPairsDataset(Dataset):
    """
    A simple PyTorch Dataset that holds question-passage pairs.

    This dataset is used for the contrastive training phase, where the model
    learns to associate a question with its corresponding correct passage.
    """

    def __init__(self, data: list):
        """
        Initializes the dataset.

        Args:
            data (list): A list of dictionaries, where each dictionary
                         is expected to have 'question' and 'passage' keys.
        """
        self.data = data

    def __len__(self):
        """Returns the total number of pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieves a question-passage pair by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The dictionary containing the 'question' and 'passage'.
        """
        return self.data[idx]


class MLMDataset(Dataset):
    """
    A PyTorch Dataset for Masked Language Modeling (MLM).

    This dataset takes a list of texts, tokenizes them, and applies a dynamic
    masking strategy. It prepares the data with `input_ids` (with masks) and
    `labels` (with unmasked tokens ignored) for the MLM training objective.
    """

    def __init__(self, texts: list, tokenizer, max_length: int = 128, mask_prob: float = 0.15):
        """
        Initializes the MLM dataset.

        Args:
            texts (list): A list of strings (passages) to be used for MLM.
            tokenizer: The Hugging Face tokenizer instance.
            max_length (int): The maximum sequence length for tokenization.
            mask_prob (float): The probability of masking a token.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        """Returns the total number of texts in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves and processes a single text for MLM.

        This involves tokenizing the text and randomly masking some of the tokens.
        Tokens that are not masked are assigned a label of -100 to be ignored
        by the loss function.

        Args:
            idx (int): The index of the text to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids' and 'labels' as PyTorch tensors.
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

        input_ids = encoding['input_ids']
        labels = input_ids.copy()

        # Create a boolean mask for non-special tokens
        probability_matrix = torch.full(torch.Size(labels), self.mask_prob)
        special_tokens_mask = encoding['special_tokens_mask']
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels = torch.tensor(labels)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(torch.Size(labels), 0.8)).bool() & masked_indices
        input_ids = torch.tensor(input_ids)
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with a random word
        indices_random = torch.bernoulli(
            torch.full(torch.Size(labels), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), torch.Size(labels), dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            'input_ids': input_ids,
            'labels': labels
        }
