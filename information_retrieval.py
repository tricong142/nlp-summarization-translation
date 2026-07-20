import os
import re
import hashlib
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Import BM25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("  ⚠ LỖI: Chưa cài rank_bm25. Chạy: pip install rank_bm25")

# ── Module-level cache (memory) ──────────────────────────────── #
_SBERT_MODEL_CACHE: Dict = {}

def _load_sbert(model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache/sbert"):
    if model_name in _SBERT_MODEL_CACHE:
        return _SBERT_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  [IR] Đang tải mô hình SBERT (Bi-encoder): {model_name}...")
        m = SentenceTransformer(model_name, cache_folder=cache_dir)
        _SBERT_MODEL_CACHE[model_name] = m
        print(f"  [IR] ✓ Tải thành công SBERT")
        return m
    except ImportError:
        print("  ⚠ LỖI: sentence-transformers chưa được cài đặt → TF-IDF fallback")
        return None
    except Exception as e:
        print(f"  ⚠ SBERT gặp lỗi: {e} → TF-IDF fallback")
        return None

def _tfidf_sim(query: str, passages: List[str]) -> List[float]:
    """Fallback dùng TF-IDF nếu SBERT không khả dụng"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = TfidfVectorizer(max_features=5000).fit_transform([query] + passages)
        return cosine_similarity(vecs[0:1], vecs[1:]).flatten().tolist()
    except Exception:
        return [0.0] * len(passages)

def _norm(scores: List[float]) -> List[float]:
    """Chuẩn hóa điểm số về khoảng [8] trước khi Hybrid Search"""
    if not scores: return scores
    mn, mx = min(scores), max(scores)
    if mx == mn: return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

_RE_NL = re.compile(r"\bNEWLINE_CHAR\b")

# ── Corpus Embeddings Cache ───────────────────────────────────── #
class CorpusEmbeddingCache:
    """Quản lý Disk Cache để tránh phải tính toán lại Embeddings"""
    def __init__(self, sbert_model, cache_dir: str = "cache/sbert_embeddings", batch_size: int = 256):
        self.model      = sbert_model
        self.cache_dir  = cache_dir
        self.batch_size = batch_size
        self._embeddings: Dict[str, np.ndarray] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]

    def _cache_path(self, corpus_hash: str) -> str:
        return os.path.join(self.cache_dir, f"corpus_{corpus_hash}.npy")

    def _key_path(self, corpus_hash: str) -> str:
        return os.path.join(self.cache_dir, f"corpus_{corpus_hash}_keys.txt")

    def encode_corpus(self, texts: List[str], corpus_label: str = "default") -> np.ndarray:
        if self.model is None: return None
        
        corpus_hash = self._text_hash(corpus_label + str(len(texts)))
        cache_path  = self._cache_path(corpus_hash)
        key_path    = self._key_path(corpus_hash)

        if os.path.exists(cache_path) and os.path.exists(key_path):
            try:
                embs = np.load(cache_path)
                with open(key_path, encoding="utf-8") as f:
                    keys = [line.rstrip("\n") for line in f]
                if len(keys) == len(texts):
                    for k, e in zip(keys, embs):
                        self._embeddings[k] = e
                    print(f"  [CACHE] ✓ Tải thành công {len(texts)} đoạn văn từ ổ cứng.")
                    return embs
            except Exception:
                pass 

        print(f"  [CACHE] Đang mã hóa {len(texts)} đoạn văn (batch={self.batch_size})...")
        t_start = __import__("time").time()
        embs = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        )
        elapsed = __import__("time").time() - t_start
        print(f"  [CACHE] ✓ Mã hóa hoàn tất trong {elapsed:.1f}s")

        try:
            np.save(cache_path, embs)
            with open(key_path, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t.replace('\n', ' ') + "\n")
        except Exception as e:
            print(f"  ⚠ Lỗi khi lưu Cache: {e}")

        for t, e in zip(texts, embs):
            self._embeddings[t] = e
        return embs

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if text in self._embeddings:
            return self._embeddings[text]
        if self.model is None: return None
        emb = self.model.encode([text], batch_size=1, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.flatten() 
        self._embeddings[text] = emb
        return emb

# ── BM25 Retriever ────────────────────────────────────────────── #
class BM25Retriever:
    """Hệ thống tìm kiếm thưa (Sparse Retrieval) dựa trên thuật toán BM25"""
    def __init__(self, passage_size: int = 4, top_k: int = 24, overlap: int = 1):
        self.passage_size = passage_size
        self.top_k        = top_k
        self.overlap      = overlap

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", _RE_NL.sub(" ", text)).strip()

    def split_into_passages(self, text: str) -> List[str]:
        text = self._clean(text)
        try:
            import nltk
            sents = nltk.sent_tokenize(text)
        except Exception:
            sents = [s.strip() for s in text.split(".") if s.strip()]
            
        if not sents: return [text[:600]]
        
        step = max(1, self.passage_size - self.overlap)
        return [" ".join(sents[i: i + self.passage_size]) 
                for i in range(0, len(sents), step)] or [text[:600]]

    def retrieve(self, query: str, documents: List[str], return_scores: bool = False):
        all_p = []
        doc_map = []
        for doc_idx, doc in enumerate(documents):
            passages = self.split_into_passages(doc)
            all_p.extend(passages)
            doc_map.extend([doc_idx] * len(passages))
            
        if not all_p or BM25Okapi is None: 
            return ([], []) if return_scores else []
            
        bm25 = BM25Okapi([p.lower().split() for p in all_p])
        scores = bm25.get_scores(query.lower().split())
        
        # Đã thêm .copy() để fix lỗi "negative stride" của PyTorch/NumPy
        idx = np.argsort(scores)[::-1].copy()
        
        passages, raw = [], []
        seen = {}
        max_per = 2
        for i in idx:
            if len(passages) >= self.top_k: break
            d = doc_map[i]
            if seen.get(d, 0) < max_per or len(seen) < 2:
                passages.append(all_p[i])
                raw.append(float(scores[i]))
                seen[d] = seen.get(d, 0) + 1
                
        return (passages, raw) if return_scores else passages

# ── MMR (Maximal Marginal Relevance) ──────────────────────────── #
def mmr_rerank(q_emb, p_embs, passages, scores, top_k, lam=0.80):
    """
    Thuật toán chống lặp: Cân bằng giữa độ liên quan và sự đa dạng.
    Đã fix lỗi kiểu dữ liệu (ValueError) trong hàm list.remove().
    """
    if len(passages) <= top_k or len(p_embs) == 0:
        return passages[:top_k], scores[:top_k]
        
    from sentence_transformers import util
    
    q_tensor = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0)
    p_tensor = torch.tensor(np.array(p_embs), dtype=torch.float32)
    
    sim_to_query = util.cos_sim(q_tensor, p_tensor).squeeze(0).tolist()
    
    # Ép kiểu int tường minh để đảm bảo hàm remove() hoạt động đúng
    best_initial_idx = int(np.argmax(sim_to_query))
    sel_idx = [best_initial_idx]
    
    rem = list(range(len(passages)))
    rem.remove(best_initial_idx)
    
    sel_embs = [p_embs[best_initial_idx]]
    
    while len(sel_idx) < top_k and len(rem) > 0:
        best_s = -9999.0
        best_i = -1
        
        sel_tensor = torch.tensor(np.array(sel_embs), dtype=torch.float32)
        
        for i in rem:
            cand_tensor = torch.tensor(p_embs[i], dtype=torch.float32).unsqueeze(0)
            sim_to_sel = util.cos_sim(cand_tensor, sel_tensor)
            max_sim_to_sel = torch.max(sim_to_sel).item()
            
            s = lam * sim_to_query[i] - (1 - lam) * max_sim_to_sel
            if s > best_s:
                best_s = s
                best_i = int(i) # Ép kiểu int
                
        sel_idx.append(best_i)
        sel_embs.append(p_embs[best_i])
        rem.remove(best_i)
        
    return [passages[i] for i in sel_idx], [scores[i] for i in sel_idx]

# ── Smart input builder ───────────────────────────────────────── #
def build_bart_input(document: str, retrieval_context: str, lead_n: int = 3, max_words: int = 800) -> str:
    doc_clean = re.sub(r"\bNEWLINE_CHAR\b", " ", document)
    try:
        import nltk
        sents = nltk.sent_tokenize(doc_clean)
    except Exception:
        sents = [s.strip() for s in doc_clean.split(".") if s.strip()]
        
    lead = " ".join(sents[:lead_n]) if sents else doc_clean[:300]
    combined = lead + " " + retrieval_context
    combined = re.sub(r"\s+", " ", combined).strip()
    
    words = combined.split()
    if len(words) > max_words:
        combined = " ".join(words[:max_words])
        last_p = max(combined.rfind(". "), combined.rfind("! "), combined.rfind("? "))
        if last_p > max_words * 4 * 0.7:
            combined = combined[:last_p + 1]
    return combined.strip()

# ── HybridRetriever ───────────────────────────────────────────── #
class HybridRetriever:
    """
    Kết hợp Tìm kiếm Tuyến tính (BM25) và Tìm kiếm Dày đặc (Dense Retrieval).
    """
    def __init__(self, config):
        self.config = config
        
        # Đã fix lỗi TypeError khi truyền tham số từ đối tượng Config
        self.top_k              = int(getattr(config, 'BM25_TOP_K', 8))
        self.alpha              = float(getattr(config, 'HYBRID_ALPHA', 0.3))
        self.use_mmr            = bool(getattr(config, 'USE_MMR', True))
        self.mmr_lambda         = float(getattr(config, 'MMR_LAMBDA', 0.8))
        self.precompute         = bool(getattr(config, 'SBERT_PRECOMPUTE', True))

        passage_size = int(getattr(config, 'BM25_PASSAGE_SIZE', 4))
        overlap      = int(getattr(config, 'BM25_OVERLAP', 1))
        
        self.bm25 = BM25Retriever(passage_size=passage_size, top_k=self.top_k * 3, overlap=overlap)
        
        self._sbert_raw = _load_sbert(getattr(config, 'SBERT_MODEL', "all-MiniLM-L6-v2"), 
                                      getattr(config, 'SBERT_CACHE_DIR', "cache/sbert"))
                                      
        self._embed_cache = CorpusEmbeddingCache(
            sbert_model=self._sbert_raw, 
            cache_dir=getattr(config, 'SBERT_EMBED_CACHE_DIR', "cache/sbert_embeddings"), 
            batch_size=int(getattr(config, 'SBERT_BATCH_SIZE', 256))
        )

    def precompute_corpus(self, corpus_by_topic: Dict[str, List[str]]) -> None:
        if not self.precompute or self._sbert_raw is None:
            return
        
        all_docs = []
        for docs in corpus_by_topic.values():
            all_docs.extend(docs)
        unique_docs = list(dict.fromkeys(all_docs))
        
        all_passages = []
        for doc in unique_docs:
            all_passages.extend(self.bm25.split_into_passages(doc))
            
        unique_passages = list(dict.fromkeys(all_passages))
        corpus_label = f"corpus_{len(unique_passages)}"
        self._embed_cache.encode_corpus(unique_passages, corpus_label)

    def _get_sbert_emb(self, text: str) -> Optional[np.ndarray]:
        return self._embed_cache.get_embedding(text)

    def retrieve(self, query: str, documents: List[str]) -> Tuple[List[str], List[float]]:
        passages, bm25_raw = self.bm25.retrieve(query, documents, return_scores=True)
        if not passages: return [], []
        
        # 1. Điểm BM25 (Sparse)
        bm25_norm = _norm(bm25_raw)

        if self._sbert_raw is not None:
            q_emb = self._get_sbert_emb(query)
            if q_emb is not None:
                p_embs = [self._get_sbert_emb(p) for p in passages]
                q_emb = q_emb.flatten()
                p_embs = [pe.flatten() if pe is not None else np.zeros(q_emb.shape) for pe in p_embs]
                
                # 2. Điểm SBERT (Dense)
                dense = _norm([float(np.dot(q_emb, pe)) for pe in p_embs])
            else:
                dense, q_emb, p_embs = _norm(_tfidf_sim(query, passages)), np.zeros(1), [np.zeros(1)] * len(passages)
        else:
            dense, q_emb, p_embs = _norm(_tfidf_sim(query, passages)), np.zeros(1), [np.zeros(1)] * len(passages)

        # 3. Kết hợp tuyến tính (Linear Combination / Hybrid Search)
        hybrid = [self.alpha * b + (1 - self.alpha) * d for b, d in zip(bm25_norm, dense)]

        # 4. Sắp xếp lại ứng viên (Reranking)
        if self.use_mmr and self._sbert_raw is not None and len(p_embs) > 1:
            passages, hybrid = mmr_rerank(q_emb, p_embs, passages, hybrid, self.top_k, self.mmr_lambda)
        else:
            ranked = sorted(zip(passages, hybrid), key=lambda x: x[8], reverse=True)
            passages = [p for p, _ in ranked[:self.top_k]]
            hybrid = [s for _, s in ranked[:self.top_k]]
            
        return passages, hybrid

    def retrieve_from_sample(self, sample: Dict, corpus_by_topic: Optional[Dict[str, List[str]]] = None, query_n_sents: int = 3) -> Dict:
        document = sample.get("document", "")
        topic    = sample.get("topic", "general")
        
        clean_doc = re.sub(r"\s+", " ", document).strip()
        sents = clean_doc.split('.')
        query = " ".join(sents[:query_n_sents])
        
        candidates = corpus_by_topic.get(topic, [document]) if corpus_by_topic and topic else [document]
        passages, scores = self.retrieve(query, candidates)
        ctx = " ".join(passages)
        
        smart_input = build_bart_input(document=document, retrieval_context=ctx, lead_n=3, max_words=800)
        ns = dict(sample)
        ns["retrieval_query"]    = query[:250]
        ns["retrieved_passages"] = passages
        ns["retrieval_scores"]   = scores
        ns["retrieval_context"]  = smart_input
        return ns

    def process_dataset(self, dataset: List[Dict], corpus_by_topic: Optional[Dict[str, List[str]]] = None, query_n_sents: Optional[int] = None, verbose: bool = True) -> List[Dict]:
        import time
        if query_n_sents is None:
            query_n_sents = int(getattr(self.config, 'QUERY_SENTENCE_COUNT', 3))
            
        print(f"  [IR] Bắt đầu Hybrid Retrieval + MMR trên {len(dataset)} mẫu...")
        
        if corpus_by_topic and self.precompute:
            self.precompute_corpus(corpus_by_topic)
            
        t0 = time.time()
        processed = []
        for i, s in enumerate(dataset):
            processed.append(self.retrieve_from_sample(s, corpus_by_topic, query_n_sents))
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(dataset)} ({rate:.1f} mẫu/s)")
                
        elapsed = time.time() - t0
        print(f"  [IR] ✓ Hoàn tất trong {elapsed:.1f}s")
        return processed

def tune_alpha(val_dataset, alpha_grid, config) -> float:
    print(f"  [IR] Alpha Tuning Grid: {alpha_grid}")
    return float(getattr(config, 'HYBRID_ALPHA', 0.3))