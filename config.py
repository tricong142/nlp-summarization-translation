

import os
import logging
import torch

def _auto_device() -> str:
    """Tự động nhận diện thiết bị chạy (GPU/CPU)"""
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class PipelineConfig:
    def __init__(self):
        # ── 0. Cấu hình Hệ thống & Dữ liệu ────────────────────────────
        self.DEVICE = _auto_device()
        # Đã cập nhật đúng đường dẫn Google Drive của bạn
        self.DATA_DIR = "/content/drive/MyDrive/Test Sumaziton" 
        
        self.MAX_TRAIN_SAMPLES = 20  # Dùng 2000 mẫu để train embedding + classifier
        self.MAX_VAL_SAMPLES   = 2   
        self.MAX_TEST_SAMPLES  = 5 
        self.DEMO_SAMPLES      = 3

        # ── 1. Cấu hình Truy xuất Thông tin (Retrieval) ──────────
        self.BM25_PASSAGE_SIZE = 4      
        self.BM25_TOP_K        = 10      
        self.BM25_OVERLAP      = 1      
        self.HYBRID_ALPHA      = 0.5
        self.TUNE_ALPHA        = False
        self.ALPHA_GRID        = [0.2, 0.3, 0.4]
        self.USE_MMR           = True
        self.MMR_LAMBDA        = 0.80
        self.QUERY_SENTENCE_COUNT = 3

        # ── 2. Cấu hình SBERT & Cache (Speed Optimization) ───────
        self.SBERT_MODEL       = "all-MiniLM-L6-v2" 
        self.SBERT_BATCH_SIZE  = 256                
        self.SBERT_CACHE_DIR   = "cache/sbert"
        self.SBERT_PRECOMPUTE  = True               
        self.SBERT_EMBED_CACHE_DIR = "cache/sbert_embeddings"

        # ── 3. Cấu hình Cross-Encoder (Xếp hạng lại) ─────────────
        self.USE_CROSS_ENCODER   = False
        self.CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.CROSS_ENCODER_TOP_K = 8

        # ── 4. Cấu hình Word Embeddings (FastText) ───────────────
        self.EMBEDDING_TYPE      = "fasttext"
        self.EMBEDDING_DIM       = 100
        self.EMBEDDING_WINDOW    = 5
        self.EMBEDDING_EPOCHS    = 3                
        self.EMBEDDING_MIN_COUNT = 1
        self.CACHE_EMBEDDINGS    = True
        self.EMBEDDING_CACHE_DIR = "cache/embeddings"

        # ── 5. Cấu hình Sinh Tóm tắt (BART Summarization) ────
        self.BART_MODEL           = "sshleifer/distilbart-cnn-12-6"
        self.BART_MAX_LENGTH      = 400
        self.BART_MIN_LENGTH      = 150             
        self.BART_NUM_BEAMS       = 8
        self.BART_NO_REPEAT_NGRAM = 4
        self.BART_LENGTH_PENALTY  = 1.0
        self.BART_BATCH_SIZE      = 2
        
        self.DYNAMIC_LENGTH       = True
        self.DYNAMIC_LENGTH_RATIO = 0.30
        self.DYNAMIC_MIN          = 150
        self.DYNAMIC_MAX          = 450

        # ── 6. Cấu hình Huấn luyện Tóm tắt (BART Fine-tuning) ────
        self.RUN_FINETUNING           = True        
        self.FINETUNED_MODEL_DIR      = "cache/finetuned_bart"
        self.FINETUNE_EPOCHS          = 4
        self.FINETUNE_LR              = 3e-5
        self.FINETUNE_WARMUP_STEPS    = 500
        self.FINETUNE_MAX_INPUT_LEN   = 512
        self.FINETUNE_MAX_TARGET_LEN  = 180
        
        self.FINETUNE_BATCH_SIZE      = 4           
        self.FINETUNE_GRAD_ACCUM      = 4           
        self.FINETUNE_SAVE_STEPS      = 200
        self.FINETUNE_EVAL_STEPS      = 200
        self.FINETUNE_FP16            = True        
        self.FINETUNE_MAX_SAMPLES     = 10000       
        self.FINETUNE_LABEL_SMOOTHING = 0.1         

        # ── 7. Cấu hình Baseline & Dịch máy & Đánh giá ───────────────
        self.RUN_LEAD3        = True
        self.LEAD_N           = 3
        self.NMT_MODEL        = "Helsinki-NLP/opus-mt-en-vi"
        self.CLASSIFIER_MODEL = "svm"
        self.EVAL_METEOR      = True
        self.EVAL_BERTSCORE   = True
        
    def summary(self) -> str:
        return (
            f"--- CONFIGURATION SUMMARY ---\n"
            f"Data Dir     : {self.DATA_DIR}\n"
            f"Device       : {self.DEVICE}\n"
            f"BART Model   : {self.BART_MODEL}\n"
            f"Fine-tuning  : {'ON' if self.RUN_FINETUNING else 'OFF'}\n"
            f"-----------------------------"
        )