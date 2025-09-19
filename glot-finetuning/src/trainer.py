"""
This file contains the GlotTrainer class, which encapsulates the entire
fine-tuning logic, including both the optional Masked Language Modeling (MLM)
pre-training and the primary contrastive training phase.
"""
import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .dataset import MLMDataset, QAPairsDataset
from .model import BiEncoderRetriever


class GlotTrainer:
    """
    A trainer to handle the two-stage fine-tuning process for the Glot-500 model.

    This class manages:
    1. An optional Masked Language Modeling (MLM) phase to adapt the model to the corpus.
    2. A contrastive training phase on question-answer pairs to tune for retrieval.
    """

    def __init__(self, config, tokenizer):
        """
        Initializes the trainer with a given configuration.

        Args:
            config: A configuration object (e.g., a dataclass) containing all
                    hyperparameters and paths.
            tokenizer: An initialized Hugging Face tokenizer.
        """
        self.config = config
        self.tokenizer = tokenizer

    def _train_mlm(self, passages: list):
        """
        Executes the Masked Language Modeling (MLM) fine-tuning phase.

        This helps the model adapt to the domain-specific vocabulary and syntax
        of the provided passages before the main contrastive task.

        Args:
            passages (list): A list of text passages for MLM training.
        """
        print("--- Starting MLM Pre-training Phase ---")
        config = self.config

        mlm_dataset = MLMDataset(passages, self.tokenizer, max_length=config.max_seq_length)
        mlm_loader = DataLoader(mlm_dataset, batch_size=config.mlm_batch_size, shuffle=True,
                                collate_fn=self._collate_mlm)

        mlm_model = AutoModelForMaskedLM.from_pretrained(config.base_model_name).to(config.device)
        mlm_optimizer = AdamW(mlm_model.parameters(), lr=config.learning_rate)

        total_steps = len(mlm_loader) * config.mlm_epochs
        scheduler = get_linear_schedule_with_warmup(mlm_optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=total_steps)

        mlm_model.train()
        for epoch in range(1, config.mlm_epochs + 1):
            total_loss = 0
            for batch in tqdm(mlm_loader, desc=f"MLM Epoch {epoch}/{config.mlm_epochs}"):
                mlm_optimizer.zero_grad()
                outputs = mlm_model(
                    input_ids=batch['input_ids'].to(config.device),
                    attention_mask=batch['attention_mask'].to(config.device),
                    labels=batch['labels'].to(config.device)
                )
                loss = outputs.loss
                loss.backward()
                mlm_optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(mlm_loader)
            print(f"MLM Epoch {epoch}/{config.mlm_epochs} - Average Loss: {avg_loss:.4f}")

        # Save the MLM-adapted model
        mlm_output_path = os.path.join(config.output_dir, "mlm_finetuned")
        os.makedirs(mlm_output_path, exist_ok=True)
        mlm_model.save_pretrained(mlm_output_path)
        self.tokenizer.save_pretrained(mlm_output_path)
        print(f"MLM-finetuned model saved to {mlm_output_path}")

        # Clean up memory
        del mlm_model, mlm_optimizer, scheduler, mlm_loader
        torch.cuda.empty_cache()

    def _train_contrastive(self, qa_data: list):
        """
        Executes the contrastive fine-tuning phase on question-passage pairs.

        The model learns to produce similar embeddings for correct pairs and
        dissimilar embeddings for incorrect pairs using an InfoNCE loss.

        Args:
            qa_data (list): A list of dictionaries, each with 'question' and 'passage'.
        """
        print("\n--- Starting Contrastive Training Phase ---")
        config = self.config

        # The model for contrastive training should be the MLM-finetuned one if it exists
        model_path = config.base_model_name
        mlm_output_path = os.path.join(config.output_dir, "mlm_finetuned")
        if config.do_mlm and os.path.exists(mlm_output_path):
            print(f"Loading model from MLM checkpoint: {mlm_output_path}")
            model_path = mlm_output_path

        retriever = BiEncoderRetriever(model_path).to(config.device)
        optimizer = AdamW(retriever.parameters(), lr=config.learning_rate)

        qa_dataset = QAPairsDataset(qa_data)
        qa_loader = DataLoader(
            qa_dataset,
            batch_size=config.contrastive_batch_size,
            shuffle=True,
            collate_fn=lambda b: self._collate_qa(b, self.tokenizer, config.max_seq_length)
        )

        total_steps = len(qa_loader) * config.contrastive_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=total_steps)

        retriever.train()
        for epoch in range(1, config.contrastive_epochs + 1):
            total_loss = 0
            for batch in tqdm(qa_loader, desc=f"Contrastive Epoch {epoch}/{config.contrastive_epochs}"):
                optimizer.zero_grad()

                q_embeddings = retriever(batch['q_input_ids'].to(config.device),
                                         batch['q_attention_mask'].to(config.device))
                p_embeddings = retriever(batch['p_input_ids'].to(config.device),
                                         batch['p_attention_mask'].to(config.device))

                loss = self._info_nce_loss(q_embeddings, p_embeddings)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(qa_loader)
            print(f"Contrastive Epoch {epoch}/{config.contrastive_epochs} - Average Loss: {avg_loss:.4f}")

            # Save a checkpoint after each epoch
            epoch_dir = os.path.join(config.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            retriever.encoder.save_pretrained(epoch_dir)
            self.tokenizer.save_pretrained(epoch_dir)
            print(f"Checkpoint saved to {epoch_dir}")

    def train(self, qa_data: list):
        """
        Starts the full training pipeline.

        This method will first run the MLM phase (if enabled in the config)
        and then proceed with the contrastive training phase.

        Args:
            qa_data (list): The training data as a list of dictionaries.
        """
        if self.config.do_mlm:
            passages = [ex['passage'] for ex in qa_data]
            self._train_mlm(passages)

        self._train_contrastive(qa_data)
        print("\nTraining complete.")

    @staticmethod
    def _info_nce_loss(q_embeds: torch.Tensor, p_embeds: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        """
        Computes the InfoNCE contrastive loss.

        This loss encourages the similarity of positive (question, passage) pairs
        to be higher than the similarity of all other negative pairs in the batch.
        """
        q_norm = nn.functional.normalize(q_embeds, p=2, dim=1)
        p_norm = nn.functional.normalize(p_embeds, p=2, dim=1)
        scores = torch.matmul(q_norm, p_norm.t()) / temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.CrossEntropyLoss()(scores, labels)

    @staticmethod
    def _collate_qa(batch: list, tokenizer, max_length: int) -> dict:
        """Collate function for question-passage pairs."""
        questions = [ex['question'] for ex in batch]
        passages = [ex['passage'] for ex in batch]
        q_enc = tokenizer(questions, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        p_enc = tokenizer(passages, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return {
            'q_input_ids': q_enc.input_ids, 'q_attention_mask': q_enc.attention_mask,
            'p_input_ids': p_enc.input_ids, 'p_attention_mask': p_enc.attention_mask
        }

    @staticmethod
    def _collate_mlm(batch: list) -> dict:
        """Collate function for MLM data."""
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        input_ids_padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = (input_ids_padded != 0).long()
        return {'input_ids': input_ids_padded, 'attention_mask': attention_mask, 'labels': labels_padded}
