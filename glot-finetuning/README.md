# Glot-500 Fine-Tuning for Passage Retrieval

This project contains the code to fine-tune a Glot-500 model for passage retrieval using a two-stage process: optional Masked Language Modeling (MLM) followed by contrastive training on question-answer pairs. The goal is to produce a powerful text encoder that can map questions and relevant passages to similar representations in an embedding space.

## The Fine-Tuning Pipeline

The `train.py` script orchestrates the full fine-tuning pipeline:

1.  **Masked Language Modeling (Optional):** If enabled in the configuration, the model is first fine-tuned on the passages from the training data using an MLM objective. This helps the model adapt to the specific vocabulary and domain of the corpus before the main retrieval task.
2.  **Contrastive Training:** The model then undergoes contrastive training using question-passage pairs. It learns to produce similar vector embeddings for a question and its correct passage, while producing dissimilar embeddings for incorrect pairings. This is achieved using an InfoNCE loss function.

## Project Structure

-   `config.py`: A centralized file for all project configurations, including file paths, the base model name, and all hyperparameters for both MLM and contrastive training phases.
-   `train.py`: The main executable script that loads the data, initializes the trainer, and starts the fine-tuning process.
-   **`src/`**: Contains the core source code for the project.
    -   `dataset.py`: Defines the PyTorch `Dataset` classes (`MLMDataset` and `QAPairsDataset`) responsible for loading and preparing the data for each training phase.
    -   `model.py`: Defines the `BiEncoderRetriever` neural network architecture and the `mean_pooling` function used to create sentence embeddings.
    -   `trainer.py`: Contains the `GlotTrainer` class, which is the core of the project. It encapsulates all the logic for both the MLM and contrastive training loops, including model setup, optimization, and saving checkpoints.

## How to Run

1.  **Prepare Your Data**:
    -   Place your `training_data.json` file in a directory accessible to the project.

2.  **Install Dependencies**:
    ```bash
    pip install torch pandas transformers scikit-learn tqdm
    ```

3.  **Configure the Project**:
    -   Open `config.py`.
    -   Set the `training_data_path` variable to the correct path of your JSON file.
    -   Adjust training hyperparameters (e.g., `epochs`, `do_mlm`, `learning_rate`) as needed.

4.  **Run Training**:
    -   Execute the main training script from your terminal:
    ```bash
    python train.py
    ```


The script will first run the MLM pre-training phase (if enabled) and save the adapted model to `output_dir/mlm_finetuned`. It will then proceed with contrastive training, saving a checkpoint for each epoch in `output_dir/epoch_{n}`.


**The final fine-tuned model is available at:**
**The final fine-tuned model is available at:**
[https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main](https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main)

