# CLIP Fine-Tuning for Persian Food VQA

This project contains the code to fine-tune a CLIP-based text encoder on a dataset of Persian food images and their corresponding descriptions. The goal is to create a model that can effectively embed both images and text into a shared representation space for retrieval tasks.

## The Fine-Tuning Pipeline

When you execute `train.py`, it orchestrates a complete end-to-end pipeline with three main stages:

1.  **Data Preparation:** The pipeline first calls the functions in `utils/data_prep.py` to process your raw dataset. It scans all subdirectories, parses the JSON files, and pairs them with their corresponding images. It performs crucial cleaning steps, such as deduplicating identical passages, and then outputs two key files into the `data/` directory:
    - `docstore.parquet`: A clean, structured knowledge base of all unique passages.
    - `training_pairs.csv`: A file listing the direct image-to-text pairs used for training.

2.  **Model Training:** The script then initializes the `FaCLIPTextTrainer`. This trainer uses the `ImageTextPairs` dataset class from `src/dataset.py` to load the training pairs. It runs a contrastive learning loop, showing the model an image and its corresponding text description. The model's goal is to learn to pull the representations of matching pairs closer together in the embedding space while pushing non-matching pairs apart. This process adjusts the weights of the text encoder to better understand the specific domain of Persian cuisine.

3.  **Index Building:** After the final training epoch is complete, the script uses the newly fine-tuned text encoder to create a FAISS index from the passages in `docstore.parquet`. This index is a highly efficient data structure that allows for near-instantaneous similarity searches, making the fine-tuned model immediately ready for use in a retrieval system.

## File Descriptions

-   `config.py`: A centralized file for all configurations. It contains paths to data, model names from Hugging Face, and all training hyperparameters like learning rate, batch size, and number of epochs. This makes experimentation easy without changing the core code.
-   `train.py`: The main entry point and orchestrator for the entire pipeline. It reads the configuration, initiates the data preparation, sets up the trainer, runs the training loop, and saves the final model and FAISS index.

-   **`src/`**: This directory contains the core source code for the model and training logic.
    -   `dataset.py`: Defines a custom PyTorch `Dataset` class named `ImageTextPairs`. Its job is to efficiently load image-text pairs from the `training_pairs.csv` file, process images, and prepare them for the model during training.
    -   `encoders.py`: Contains Python classes that act as wrappers around the Hugging Face `CLIPVisionModel` and `AutoModel` (for text). These classes handle the logic for encoding images and text into vector representations (embeddings).
    -   `trainer.py`: Contains the `FaCLIPTextTrainer` class, which is the heart of the project. It manages the entire fine-tuning process, including setting up the optimizer, handling the training loop, calculating the contrastive loss, performing backpropagation, and saving the final model artifacts.

-   **`utils/`**: This directory contains utility scripts, primarily for data processing.
    -   `data_prep.py`: Contains all the functions necessary for the first stage of the pipeline. It handles finding, parsing, cleaning, and structuring the raw data into the `docstore.parquet` and the training CSV file.

## How to Run

1.  **Prepare Your Data**:
    - Place your raw food dataset (the folders containing images and `.json` files) into a single directory.

2.  **Install Dependencies**:
    ```bash
    pip install torch pandas transformers faiss-cpu pillow tqdm
    ```

3.  **Configure the Project**:
    - Open `config.py`.
    - Set the `base_data_root` variable to the path of the directory from Step 1.
    - (Optional) Adjust other hyperparameters like `epochs`, `batch_size`, etc., as needed.

4.  **Run Training**:
    - Execute the main training script from your terminal:
    ```bash
    python train.py
    ```
    - You can also override the settings from `config.py` using command-line arguments for quick experiments:
    ```bash
    python train.py --epochs 5 --learning_rate 2e-5 --output_dir "models/my_finetuned_clip"
    ```

This process will first scan and process your raw data to create a `docstore.parquet` and a training CSV file in the `data/` directory. It will then proceed with fine-tuning the model. The final model, tokenizer, and the corresponding FAISS index will be saved to the directory specified by `output_dir` in your configuration.


**The final fine-tuned model is available at:**
[https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main](https://huggingface.co/Arshiaizd/MCLIP_FA_FineTuned/tree/main)
