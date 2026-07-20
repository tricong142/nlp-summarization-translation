"""
main_pipeline.py  ─ Bản Full 100% (Maximum Accuracy Pipeline)
=============================================================
Tệp thực thi chính kết nối toàn bộ 10 module thuật toán.
Thực thi tuần tự: Load Data → Preprocess → SVM Classify → FastText 
→ BART Fine-tune → Hybrid Retrieval (MMR) → BART Gen → NMT → Evaluate.
"""
import os
import time
import json
import warnings
import logging
from typing import List, Dict

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Import toàn bộ 10 module đã viết ─────────────────────────
from config import PipelineConfig
from data_loader import load_all_data
from preprocessing import TextPreprocessor
from document_classifier import DocumentClassifier
from information_retrieval import HybridRetriever, tune_alpha
from embedding import WordEmbeddingModel
from bart_finetuner import BARTFinetuner
from summarization_bart import BARTSummarizer, extract_keywords_tfidf, compute_keyword_coverage
from translation_nmt import NMTTranslator
from evaluation_metrics import EvaluationPipeline

def _banner(title: str, w: int = 68) -> None:
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")

def _fmt(sec: float) -> str:
    return (f"{sec:.1f}s" if sec < 60 
            else f"{sec/60:.1f}min" if sec < 3600 
            else f"{sec/3600:.1f}hr")

def _print_sample(s: Dict, idx: int) -> None:
    print(f"\n{'═' * 68}\n  SAMPLE #{idx + 1}\n{'═' * 68}")
    doc = s.get("document", "")
    print(f"\n📄 TÀI LIỆU GỐC ({len(doc.split())} từ):")
    print(f"   {doc[:280]}...")
    
    if "topic" in s:
        print(f"\n🏷 CHỦ ĐỀ (SVM): {s['topic']}")
        
    ctx = s.get("retrieval_context", "")
    if ctx:
        print(f"\n🔍 NGỮ CẢNH TRUY XUẤT (SBERT + BM25 + MMR):")
        print(f"   {ctx[:280]}...")
        
    print(f"\n✅ TÓM TẮT THAM CHIẾU (HUMAN):\n   {s.get('summary', 'N/A')}")
    print(f"\n🤖 BART TÓM TẮT (TIẾNG ANH):\n   {s.get('generated_summary', 'N/A')}")
    print(f"\n🇻🇳 NMT DỊCH (TIẾNG VIỆT):\n   {s.get('vietnamese_summary', 'N/A')}")
    
    sc = s.get("eval_scores", {})
    if sc:
        print(f"\n📊 ĐIỂM ĐÁNH GIÁ CHI TIẾT:")
        for m in ("rouge1_f1", "rouge1_recall", "rouge2_f1", "rougeL_f1", "meteor", "bleu", "coverage", "redundancy"):
            if m in sc:
                print(f"   {m:<28}: {sc[m]:.4f}")
    print("═" * 68)

class NLPPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timing = {}

    def _print_timing(self) -> None:
        _banner("TỔNG KẾT THỜI GIAN CHẠY (TIMING SUMMARY)")
        total = sum(self.timing.values())
        for step, secs in self.timing.items():
            pct = secs / total * 100 if total else 0
            bar = "█" * int(pct / 5)
            print(f"  {step:<26}: {_fmt(secs):>8}  ({pct:5.1f}%) {bar}")
        print(f"  {'─' * 52}\n  {'TỔNG CỘNG':<26}: {_fmt(total):>8}")

    def run(self) -> List[Dict]:
        _banner("NLP PIPELINE v6.1 — MAXIMUM ACCURACY")
        start_time = time.time()
        
        # 1. Load Data
        print("\n  [1/9] Đang tải dữ liệu...")
        t0 = time.time()
        train_data, val_data, test_data = load_all_data(self.config)
        self.timing["1_load_data"] = time.time() - t0
        
        if not test_data:
            print("  ✗ Dữ liệu trống. Vui lòng kiểm tra lại DATA_DIR.")
            return []

        # 2. Preprocessing
        print("\n  [2/9] Tiền xử lý (Xóa nhiễu, Khử lặp câu)...")
        t0 = time.time()
        preprocessor = TextPreprocessor(dedup_sentences=True)
        train_data = preprocessor.process_dataset(train_data)
        test_data = preprocessor.process_dataset(test_data)
        self.timing["2_preprocess"] = time.time() - t0

        # 3. Classification (SVM)
        print("\n  [3/9] Phân loại chủ đề (SVM + TF-IDF)...")
        t0 = time.time()
        classifier = DocumentClassifier(model_type=self.config.CLASSIFIER_MODEL)
        train_data = classifier.prepare_labels(train_data)
        test_data = classifier.prepare_labels(test_data)
        classifier.train(train_data)
        groups = classifier.classify_and_group(test_data)
        self.timing["3_classify"] = time.time() - t0

        # 4. Word Embedding (FastText)
        print("\n  [4/9] Huấn luyện Biểu diễn từ (FastText)...")
        t0 = time.time()
        embedding_model = WordEmbeddingModel(self.config)
        embedding_model.train(train_data)
        self.timing["4_embedding"] = time.time() - t0

        # 5. BART Fine-tuning
        print("\n  [5/9] Tinh chỉnh mô hình BART (Label Smoothing, FP16)...")
        t0 = time.time()
        finetuner = BARTFinetuner(self.config)
        bart_model_path = finetuner.finetune(train_data, val_data)
        self.timing["5_finetune"] = time.time() - t0

        # 6. Hybrid Retrieval
        print("\n  [6/9] Truy xuất thông tin (SBERT + BM25 + MMR)...")
        t0 = time.time()
        retriever = HybridRetriever(self.config)
        test_data = retriever.process_dataset(test_data, groups)
        self.timing["6_retrieval"] = time.time() - t0

        # 7. Summarization (BART)
        print("\n  [7/9] Sinh tóm tắt (Beam Search + Dynamic Length)...")
        t0 = time.time()
        summarizer = BARTSummarizer(model_name=bart_model_path, device=self.config.DEVICE)
        for s in test_data:
            ctx = s.get("retrieval_context", s.get("document", ""))
            s['generated_summary'] = summarizer.summarize(
                text=ctx, 
                min_length=self.config.BART_MIN_LENGTH, 
                num_beams=self.config.BART_NUM_BEAMS,
                length_penalty=self.config.BART_LENGTH_PENALTY,
                dynamic_ratio=self.config.DYNAMIC_LENGTH_RATIO
            )
            # Trích xuất từ khóa tính Coverage
            kws = extract_keywords_tfidf(s.get("document", ""))
            s['input_keywords'] = kws
            s['keyword_coverage'] = compute_keyword_coverage(s['generated_summary'], kws)
        self.timing["7_summarize"] = time.time() - t0

        # 8. Translation (NMT)
        print("\n  [8/9] Dịch máy sang Tiếng Việt (Sentence-level + Entity Preserv)...")
        t0 = time.time()
        translator = NMTTranslator(model_name=self.config.NMT_MODEL, device=self.config.DEVICE)
        for s in test_data:
            s['vietnamese_summary'] = translator.translate_sentence_level(s.get('generated_summary', ''))
        self.timing["8_translate"] = time.time() - t0

        # 9. Evaluation
        print("\n  [9/9] Đánh giá chất lượng (ROUGE, BLEU, METEOR, BERTScore)...")
        t0 = time.time()
        evaluator = EvaluationPipeline(
            use_hf_evaluate=True, 
            use_meteor=self.config.EVAL_METEOR, 
            use_bertscore=self.config.EVAL_BERTSCORE
        )
        final_scores = evaluator.evaluate_dataset(test_data)
        
        # Gắn điểm vào từng mẫu
        if "per_sample_scores" in final_scores:
            for i, sc in enumerate(final_scores["per_sample_scores"]):
                test_data[i]["eval_scores"] = sc
        self.timing["9_evaluate"] = time.time() - t0

        # In kết quả
        _banner("KẾT QUẢ CHI TIẾT (SAMPLES)")
        for i in range(min(self.config.DEMO_SAMPLES, len(test_data))):
            _print_sample(test_data[i], i)
            
        print("\n  📊 BÁO CÁO TỔNG QUAN (AVERAGE SCORES):")
        for k, v in final_scores.items():
            if k != "per_sample_scores":
                print(f"   {k:<20}: {v}")

        self._print_timing()
        return test_data

if __name__ == "__main__":
    cfg = PipelineConfig()
    # Bạn có thể bật Fine-tuning ở đây bằng cách xóa dấu #
    cfg.RUN_FINETUNING = True
    cfg.MAX_TRAIN_SAMPLES = 20
    
    pipeline = NLPPipeline(cfg)
    results = pipeline.run()