# ==============================================================================

# Part 1: Core Classes from the Original Script

# All the necessary helper classes for the RAG system are defined here.

# ==============================================================================

import os
import re
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image, ImageOps

# Hugging Face Transformers & Sentence-Transformers
from transformers import (CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModel)
from sentence_transformers import SentenceTransformer

# Google Generative AI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Gradio for Web UI
import gradio as gr

# --- CONFIGURATION CLASS ---
class Config:
    per_option_ctx: int = 5
    max_text_len: int = 512
    docstore_path: str = "indexes/docstore.parquet"
    glot_model_hf: str = "Arshiaizd/Glot500-FineTuned"
    mclip_text_model_hf: str = "Arshiaizd/MCLIP_FA_FineTuned"
    clip_vision_model: str = "SajjadAyoubi/clip-fa-vision"
    glot_index_out: str = "indexes/I_glot_text_fa.index"
    clip_index_out: str = "indexes/I_clip_text_fa.index"

# --- UTILITY CLASS ---
class Utils:
    @staticmethod
    def build_context_block(hits: List[Tuple[int, float]], docstore: pd.DataFrame, count: int, max_chars=350) -> str:
        if not hits or docstore.empty:
            return "No relevant documents found."
        lines = []
        # Ensure we don't try to access indices that are out of bounds
        valid_hits = [h for h in hits if h[0] < len(docstore)]
        for i, score in valid_hits[:count]:
            row = docstore.iloc[i]
            # Ensure 'passage_text' and 'id' columns exist
            txt = str(row.get("passage_text", "Text not available"))
            doc_id = row.get("id", "N/A")
            txt = (txt[:max_chars] + "…") if len(txt) > max_chars else txt
            lines.append(f"- [doc:{doc_id}] {txt}")
        return "\n".join(lines)

# --- ENCODER CLASSES ---
class Glot500Encoder:
    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = SentenceTransformer(model_id, device=str(self.device))
        print(f"Glot-500 model '{model_id}' loaded successfully.")

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.st_model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

class FaTextEncoder:
    def __init__(self, model_id: str, device: torch.device, max_len: int):
        self.device, self.max_len = device, max_len
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        print(f"FaCLIP text model '{model_id}' loaded successfully.")

    @torch.no_grad()
    def encode_numpy(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            toks = self.tok(
                texts[i:i+batch_size], padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            out = self.model(**toks)
            x = (
                out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None
                else (out.last_hidden_state * toks.attention_mask.unsqueeze(-1)).sum(1)
                     / toks.attention_mask.sum(1).clamp(min=1)
            )
            x_norm = x / x.norm(p=2, dim=1, keepdim=True)
            vecs.append(x_norm.detach().cpu().numpy())
        return np.vstack(vecs).astype(np.float32)

class FaVisionEncoder:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.exif_transpose(img).convert("RGB")
        batch = self.proc(images=img, return_tensors="pt").to(self.device)
        out = self.model(**batch)
        v = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
        v_norm = v / v.norm(p=2, dim=1, keepdim=True)
        return v_norm[0].detach().cpu().numpy().astype(np.float32)

# --- RETRIEVER CLASSES ---
class BaseRetriever:
    def __init__(self, docstore: pd.DataFrame, index_path: str):
        self.docstore, self.index_path = docstore.reset_index(drop=True), index_path
        if os.path.isfile(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Make sure it's uploaded to your Space.")

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        D, I = self.index.search(query_vec[None, :].astype(np.float32), k)
        return list(zip(I[0].tolist(), D[0].tolist()))

class Glot500Retriever(BaseRetriever):
    def __init__(self, encoder: Glot500Encoder, docstore: pd.DataFrame, index_path: str):
        super().__init__(docstore, index_path)
        self.encoder = encoder

    def topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        qv = self.encoder.encode([query], batch_size=1)[0]
        return self.search(qv, k)

class TextIndexRetriever(BaseRetriever):
    def __init__(self, text_encoder: FaTextEncoder, docstore: pd.DataFrame, index_path: str):
        super().__init__(docstore, index_path)
        self.encoder = text_encoder

# --- GENERATION AND SYSTEM CLASSES ---
class VLM_GenAI:
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.1, max_output_tokens: int = 1024):
        if not api_key or "YOUR" in api_key:
            raise ValueError("Gemini API Key is missing or is a placeholder. Please add it to your Hugging Face Space secrets.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

class RAGSystem:
    def __init__(self, cfg: Config):
        self.docstore = pd.read_parquet(cfg.docstore_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glot_enc = Glot500Encoder(cfg.glot_model_hf)
        self.glot_ret = Glot500Retriever(self.glot_enc, self.docstore, cfg.glot_index_out)

        txt_enc = FaTextEncoder(cfg.mclip_text_model_hf, device, cfg.max_text_len)
        self.mclip_ret = TextIndexRetriever(txt_enc, self.docstore, cfg.clip_index_out)
        self.vision = FaVisionEncoder(cfg.clip_vision_model, device)

# ==============================================================================

# Part 2: Gradio Web Application

# ==============================================================================

# --- 1. LOAD MODELS AND INDEXES (This runs only once when the app starts) ---
print("Initializing configuration...")
cfg = Config()
print("Loading RAG system (models, encoders, and retrievers)...")
rag_system = RAGSystem(cfg)
print("Initializing Gemini model...")
api_key = os.environ.get("GEMINI_API_KEY")
vlm = VLM_GenAI(api_key, model_name="models/gemini-1.5-flash")
print("System ready.")

# --- 2. DEFINE THE FUNCTION TO HANDLE USER INPUT ---
def run_rag_query(question_text: str, question_image: Optional[Image.Image]) -> Tuple[str, str]:
    if not question_text.strip():
        return "Please ask a question.", ""
    context_block = ""
    # Decide which retriever to use based on input
    if question_image:
        print("Performing multimodal retrieval...")
        img_vec = rag_system.vision.encode(question_image)
        hits = rag_system.mclip_ret.search(img_vec, k=cfg.per_option_ctx)
        context_block = Utils.build_context_block(hits, rag_system.docstore, cfg.per_option_ctx)
    else:
        print("Performing text retrieval...")
        hits = rag_system.glot_ret.topk(question_text, k=cfg.per_option_ctx)
        context_block = Utils.build_context_block(hits, rag_system.docstore, cfg.per_option_ctx)

    # --- Augment and Generate ---
    print("Generating response...")
    if question_image:
        prompt = f"با توجه به تصویر و اسناد زیر، به سوال پاسخ دهید.\n\nاسناد:\n{context_block}\n\nسوال: {question_text}"
    else:
        prompt = f"با توجه به اسناد زیر، به سوال پاسخ دهید.\n\nاسناد:\n{context_block}\n\nسوال: {question_text}"

    content_parts = [question_image, prompt] if question_image else [prompt]

    try:
        resp = vlm.model.generate_content(
            content_parts,
            generation_config=vlm.generation_config,
            safety_settings=vlm.safety_settings
        )
        answer = resp.text
    except Exception as e:
        answer = f"Error during generation: {e}"
        print(answer)
    return answer, context_block

# --- 3. CREATE THE GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), title="Persian Culinary RAG") as demo:
    gr.Markdown("# 🍲 Persian Culinary RAG Demo")
    gr.Markdown("Ask a question about Iranian food, with or without an image, to see the RAG system in action.")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload an Image (Optional)")
            text_input = gr.Textbox(label="Ask your question in Persian", placeholder="...مثلا: در مورد قورمه سبزی توضیح بده")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column(scale=2):
            output_answer = gr.Textbox(label="Answer from Model", lines=8, interactive=False)
            output_context = gr.Textbox(label="Retrieved Context (What the model used to answer)", lines=12, interactive=False)

    gr.Examples(
        examples=[
            ["در مورد حلوا توضیح بده", None],
            ["مواد لازم برای تهیه آش رشته چیست؟", None],
        ],
        inputs=[text_input, image_input]
    )

    submit_button.click(
        fn=run_rag_query,
        inputs=[text_input, image_input],
        outputs=[output_answer, output_context]
    )


demo.launch()
