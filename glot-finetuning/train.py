"""
This is the main entry point for the Glot-500 fine-tuning project.

This script reads the configuration, loads the training data, initializes the
trainer, and starts the full training pipeline which includes optional MLM
pre-training followed by contrastive training.
"""
import json
from transformers import AutoTokenizer
from config import TrainingConfig
from src.trainer import GlotTrainer

def main():
    """
    Main function to orchestrate the fine-tuning process.
    """
    # 1. Load configuration
    config = TrainingConfig()
    print("--- Configuration ---")
    print(config)
    print("---------------------")

    # 2. Load Tokenizer
    print(f"Loading tokenizer: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # 3. Load Data
    print(f"Loading training data from: {config.training_data_path}")
    try:
        with open(config.training_data_path, 'r', encoding='utf-8-sig') as f:
            qa_data = json.load(f)
        print(f"Loaded {len(qa_data)} question-passage pairs.")
    except FileNotFoundError:
        print(f"Error: Training data not found at {config.training_data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse the JSON file at {config.training_data_path}")
        return

    # 4. Initialize and run the trainer
    trainer = GlotTrainer(config, tokenizer)
    trainer.train(qa_data)

if __name__ == "__main__":
    main()
