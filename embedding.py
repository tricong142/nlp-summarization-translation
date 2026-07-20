"""
embedding.py ─ Bản Full 100%
==================================================
Mô hình biểu diễn từ (Word Embeddings) sử dụng FastText/Word2Vec.
- Sử dụng character n-grams để xử lý OOV (Out-of-vocabulary).
- Tích hợp hệ thống Disk Caching để nạp nhanh chóng.
"""
import os
import logging
from typing import List, Dict

try:
    from gensim.models import FastText, Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

logging.getLogger("gensim").setLevel(logging.ERROR)

class WordEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_path = os.path.join(
            config.EMBEDDING_CACHE_DIR, 
            f"{config.EMBEDDING_TYPE}_d{config.EMBEDDING_DIM}_w{config.EMBEDDING_WINDOW}.bin"
        )
        os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)

    def _prepare_sentences(self, dataset: List[Dict]) -> List[List[str]]:
        """Tiền xử lý văn bản thô thành danh sách các từ (tokens)"""
        if not GENSIM_AVAILABLE: return []
        sentences = []
        for sample in dataset:
            doc = sample.get("document", "")
            if doc:
                # Phân tách thành token cơ bản, chuyển lowercase
                sentences.append(simple_preprocess(doc, deacc=False))
        return sentences

    def train(self, dataset: List[Dict]) -> None:
        """Huấn luyện hoặc nạp mô hình từ đĩa (Cache)"""
        if not GENSIM_AVAILABLE:
            print("  ⚠ LỖI: Chưa cài đặt gensim. Hãy chạy: pip install gensim")
            return

        # 1. Kiểm tra Disk Cache
        if self.config.CACHE_EMBEDDINGS and os.path.exists(self.model_path):
            print(f"  [EMBEDDING] Nạp mô hình {self.config.EMBEDDING_TYPE.upper()} từ Disk Cache...")
            if self.config.EMBEDDING_TYPE.lower() == "fasttext":
                self.model = FastText.load(self.model_path)
            else:
                self.model = Word2Vec.load(self.model_path)
            return

        # 2. Huấn luyện mô hình từ đầu (From scratch)
        print(f"  [EMBEDDING] Đang huấn luyện {self.config.EMBEDDING_TYPE.upper()} trên {len(dataset)} tài liệu...")
        sentences = self._prepare_sentences(dataset)
        
        if not sentences:
            print("  [EMBEDDING] Dữ liệu rỗng, bỏ qua huấn luyện.")
            return

        if self.config.EMBEDDING_TYPE.lower() == "fasttext":
            self.model = FastText(
                sentences=sentences,
                vector_size=self.config.EMBEDDING_DIM,
                window=self.config.EMBEDDING_WINDOW,
                min_count=self.config.EMBEDDING_MIN_COUNT,
                epochs=self.config.EMBEDDING_EPOCHS,
                workers=4,
                sg=1 # Dùng kiến trúc Skip-gram
            )
        else:
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.config.EMBEDDING_DIM,
                window=self.config.EMBEDDING_WINDOW,
                min_count=self.config.EMBEDDING_MIN_COUNT,
                epochs=self.config.EMBEDDING_EPOCHS,
                workers=4,
                sg=1
            )
            
        # 3. Lưu mô hình xuống đĩa
        if self.config.CACHE_EMBEDDINGS:
            self.model.save(self.model_path)
            print(f"  [EMBEDDING] ✓ Đã lưu mô hình tại: {self.model_path}")

    def get_vector(self, word: str) -> list:
        """Truy xuất biểu diễn vector của một từ"""
        if self.model is None: return [0.0] * self.config.EMBEDDING_DIM
        try:
            return self.model.wv[word].tolist()
        except KeyError:
            # FastText tự động xử lý OOV bằng n-grams, Word2Vec sẽ ném KeyError
            return [0.0] * self.config.EMBEDDING_DIM


