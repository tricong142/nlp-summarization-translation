import re
import warnings
import logging
import torch
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Vietnamese post-processing ────────────────────────────────── #

def postprocess_vietnamese(
    text:                str,
    remove_repetitions:  bool = True,
    fix_punct:           bool = True,
) -> str:
    """Hậu xử lý văn bản tiếng Việt để làm mượt mà bản dịch"""
    if not text or not text.strip():
        return ""
        
    text = text.replace("[UNK]", "").replace("<unk>", "")
    
    if fix_punct:
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        
    if remove_repetitions:
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+\s+\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        
    text = re.sub(r"\s+", " ", text).strip()
    
    if text:
        text = text.upper() + text[1:]
        
    if text and text[-1] not in ".!?":
        text += "."
        
    return text

def split_into_sentences(text: str, max_chars: int = 400) -> List[str]:
    """Cắt câu thông minh để tránh lỗi dịch mất chữ của NMT"""
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        
    chunks = []
    current = []
    length = 0
    
    for word in sentences:
        if length + len(word) > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
            
    if current:
        chunks.append(" ".join(current))
        
    return [c for c in chunks if c.strip()]

# ── NMTTranslator ─────────────────────────────────────────────── #

class NMTTranslator:
    def __init__(
        self,
        model_name:   str           = "Helsinki-NLP/opus-mt-en-vi",
        device:       Optional[str] = None,
        use_pipeline: bool          = False,
    ):
        self.model_name   = model_name
        self.use_pipeline = use_pipeline
        self.is_t5_model  = any(x in model_name.lower() for x in ["t5", "vit5", "envit5"])
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"\n  [NMT] Đang tải mô hình Dịch máy (NMT): {self.model_name}")
        print(f"  [NMT] Device: {self.device}  Type: {'T5' if self.is_t5_model else 'Marian'}")
              
        self._load_model()

    def _load_model(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.is_t5_model:
                self._load_t5()
            else:
                self._load_marian()

    def _load_marian(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model     = MarianMTModel.from_pretrained(self.model_name)
            self.model.config.tie_word_embeddings = False
            self.model = self.model.to(self.device)
            self.model.eval()
            self.translate_fn = self._translate_marian
            print("  [NMT] ✓ Đã tải mô hình NMT thành công.")

    def _load_t5(self) -> None:
        try:
            # Thử tải T5
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
                self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.translate_fn = self._translate_t5
                print("  [NMT] ✓ Đã tải mô hình NMT thành công.")
        except Exception as e:
            # NẾU LỖI THƯ VIỆN, TỰ ĐỘNG CHUYỂN SANG MÔ HÌNH MARIAN CỦA HELSINKI
            print(f"  [NMT] ⚠ LỖI thư viện transformers với mô hình {self.model_name}: {e}")
            print("  [NMT] 🔄 Tự động chuyển sang mô hình MarianMT siêu ổn định (Helsinki-NLP/opus-mt-en-vi)...")
            self.model_name = "Helsinki-NLP/opus-mt-en-vi"
            self.is_t5_model = False
            self._load_marian()

    # ── Core translation ─────────────────────────────────────── #
    
    def translate(
        self,
        text:           str,
        max_length:     int   = 512,
        num_beams:      int   = 5,
        length_penalty: float = 1.0,
    ) -> str:
        if not text or not text.strip():
            return ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.translate_fn(text, max_length, num_beams, length_penalty)

    def _translate_marian(
        self, text: str, max_length: int, num_beams: int, length_penalty: float
    ) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        input_ids      = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
            
        trans = self.tokenizer.batch_decode(
            out, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return trans if trans else ""

    def _translate_t5(
        self, text: str, max_length: int, num_beams: int, length_penalty: float
    ) -> str:
        prefixed = f"translate English to Vietnamese: {text}"
        inputs   = self.tokenizer(
            prefixed, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
            )
            
        trans = self.tokenizer.batch_decode(
            out, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return trans if trans else ""

    # ── Sentence-level translation ────────────────────────────── #
    
    def translate_sentence_level(
        self,
        text:           str,
        max_chars:      int   = 400,
        num_beams:      int   = 5,
        max_length:     int   = 512,
        length_penalty: float = 1.0,
    ) -> str:
        # Entity Preservation
        text = re.sub(r'two-thirds', '2/3', text, flags=re.IGNORECASE)
        
        chunks = split_into_sentences(text, max_chars=max_chars)
        if not chunks:
            return postprocess_vietnamese(self.translate(text))
            
        translated_parts = []
        for chunk in chunks:
            vi = self.translate(
                chunk,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
            if ' '.join(vi).strip():
                translated_parts.append(' '.join(vi).strip())
                
        raw = " ".join(translated_parts)
        return postprocess_vietnamese(raw)

    # ── Batch translation ─────────────────────────────────────── #
    
    def translate_batch(
        self,
        texts:          List[str],
        batch_size:     int  = 8,
        sentence_level: bool = True,
        **kwargs,
    ) -> List[str]:
        total = len(texts)
        print(f"  [NMT] Đang dịch {total} đoạn tóm tắt sang Tiếng Việt...")
              
        translations = []
        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            if sentence_level:
                batch_trans = [
                    self.translate_sentence_level(t, **kwargs)
                    for t in batch
                ]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    inputs = self.tokenizer(
                        batch, return_tensors="pt",
                        padding=True, truncation=True, max_length=512,
                    )
                    input_ids      = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)
                    
                    with torch.no_grad():
                        out = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=kwargs.get("max_length", 512),
                            num_beams=kwargs.get("num_beams", 5),
                            early_stopping=True,
                        )
                        
                    batch_trans = [
                        postprocess_vietnamese(t)
                        for t in self.tokenizer.batch_decode(
                            out, skip_special_tokens=True)
                    ]
            translations.extend(batch_trans)
            
        return translations

    # ── Dataset API ───────────────────────────────────────────── #
    
    def process_dataset(
        self,
        dataset:        List[Dict],
        summary_key:    str  = "generated_summary",
        sentence_level: bool = True,
        **kwargs,
    ) -> List[Dict]:
        summaries  = [s.get(summary_key, s.get("generated_summary", ""))
                      for s in dataset]
        vietnamese = self.translate_batch(
            summaries, sentence_level=sentence_level, **kwargs)
            
        results = []
        for sample, vi in zip(dataset, vietnamese):
            ns = dict(sample)
            ns["vietnamese_summary"] = vi
            results.append(ns)
            
        return results