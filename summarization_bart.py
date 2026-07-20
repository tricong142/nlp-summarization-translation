import re
import warnings
import logging
import torch
from typing import List, Dict, Optional
from collections import Counter

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline,
)

# ── Helpers ───────────────────────────────────────────────────── #

def _strip_noise(text: str) -> str:
    # Fix an toàn: Chặn lỗi nếu text lọt vào là dạng list
    if isinstance(text, list):
        text = text
    text = str(text)
    text = re.sub(r"\bNEWLINE_CHAR\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _dynamic_max_length(
    text:      str,
    ratio:     float = 0.30,
    min_l:     int   = 150,
    max_l:     int   = 450,
) -> int:
    """Scale max_length proportionally với input length."""
    word_count = len(text.split())
    target     = int(word_count * ratio)
    return max(min_l, min(target, max_l))

def _post_process_summary(text: str) -> str:
    """
    Fix common BART output artifacts:
    - Incomplete trailing sentence
    - Duplicate phrases từ beam search
    - Extra whitespace
    """
    # Fix an toàn tuyệt đối: Trích xuất string dù thư viện trả về cấu trúc list/dict
    if isinstance(text, list):
        text = text
        if isinstance(text, dict):
            text = text.get("summary_text", "")
            
    text = str(text).strip()
    if not text:
        return text
        
    # Remove trailing incomplete sentence nếu không kết thúc bằng dấu câu
    if text and text[-1] not in ".!?":
        last = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        if last > len(text) * 0.5:
            text = text[:last + 1]
            
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords_tfidf(text: str, top_k: int = 10) -> List[str]:
    """Top-k keywords theo TF-IDF frequency."""
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()
        
    words = [w for w in re.sub(r"[^\w\s]", "", text.lower()).split()
             if w not in sw and len(w) > 3]
    return [w for w, _ in Counter(words).most_common(top_k)]

def compute_keyword_coverage(summary: str, keywords: List[str]) -> float:
    """Fraction of keywords xuất hiện trong summary."""
    if not keywords:
        return 0.0
    sl = summary.lower()
    return round(sum(1 for kw in keywords if kw in sl) / len(keywords), 3)

# ── BARTSummarizer ────────────────────────────────────────────── #

class BARTSummarizer:
    """
    BART / DistilBART / PEGASUS summarizer.
    v6 key changes:
    - Default min_length=150 (fix ROUGE-Recall)
    - Default length_penalty=1.0 (no padding bias)
    - Default num_beams=8 (better quality)
    - Post-processing để fix artifacts
    """
    def __init__(
        self,
        model_name:   str           = "sshleifer/distilbart-cnn-12-6",
        device:       Optional[str] = None,
        use_pipeline: bool          = False,
    ):
        self.model_name   = model_name
        self.use_pipeline = use_pipeline
        self.device       = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"\nBARTSummarizer: {model_name}")
        print(f"  Device: {self.device}")
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU   : {gpu}  ({mem:.1f} GB VRAM)")
            if mem < 8 and "large" in model_name:
                print("  ⚠ Có thể OOM với bart-large. "
                      "Dùng distilbart-cnn-12-6 nếu cần.")
                      
        self._load_model()

    def _load_model(self) -> None:
        print(f"  Loading {self.model_name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.use_pipeline:
                self.summarizer_pipeline = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                )
                self.tokenizer = None
                self.model     = None
                
            elif "pegasus" in self.model_name.lower():
                from transformers import (
                    PegasusForConditionalGeneration,
                    PegasusTokenizer,
                )
                self.tokenizer = PegasusTokenizer.from_pretrained(
                    self.model_name)
                self.model     = PegasusForConditionalGeneration.from_pretrained(
                    self.model_name, ignore_mismatched_sizes=True)
                self.model.config.tie_word_embeddings = False
                self.model = self.model.to(self.device)
                self.model.eval()
                self.summarizer_pipeline = None
                
            else:
                self.tokenizer = BartTokenizer.from_pretrained(
                    self.model_name)
                self.model     = BartForConditionalGeneration.from_pretrained(
                    self.model_name, ignore_mismatched_sizes=True)
                self.model.config.tie_word_embeddings = False
                self.model = self.model.to(self.device)
                self.model.eval()
                self.summarizer_pipeline = None
                
        print("  ✓ Model ready")

    def summarize(
        self,
        text:                 str,
        max_length:           int   = 400,
        min_length:           int   = 150,
        num_beams:            int   = 8,
        length_penalty:       float = 1.0,
        early_stopping:       bool  = True,
        no_repeat_ngram_size: int   = 4,
        do_sample:            bool  = False,
        top_p:                float = 0.92,
        top_k:                int   = 50,
        temperature:          float = 1.0,
        dynamic_length:       bool  = True,
        dynamic_ratio:        float = 0.30,
        dynamic_min:          int   = 150,
        dynamic_max:          int   = 450,
    ) -> str:
        """Generate summary với các tham số đã được tối ưu v6."""
        text = _strip_noise(text)
        if not text:
            return ""
            
        # Dynamic length scaling
        if dynamic_length:
            max_length = _dynamic_max_length(
                text, ratio=dynamic_ratio,
                min_l=dynamic_min, max_l=dynamic_max,
            )
            min_length = min(min_length, max_length - 15)
            
        if self.use_pipeline:
            result = self.summarizer_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                do_sample=do_sample,
            )
            if isinstance(result, list):
                return _post_process_summary(result["summary_text"])
            return _post_process_summary(result["summary_text"])
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids      = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            gen_kw: Dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                forced_bos_token_id=self.tokenizer.bos_token_id,
            )
            
            if do_sample:
                gen_kw.update(
                    do_sample=True, top_p=top_p,
                    top_k=top_k, temperature=temperature, num_beams=1,
                )
            else:
                gen_kw.update(
                    do_sample=False,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                )
                
            with torch.no_grad():
                summary_ids = self.model.generate(**gen_kw)
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ĐÃ FIX: Chỉ định đúng phần tử đầu tiên của lô (batch) summary_ids
                summary = self.tokenizer.decode(
                    summary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
        return _post_process_summary(summary)