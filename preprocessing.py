import re
import nltk

# Đảm bảo NLTK data đã được tải để phục vụ việc tách câu chuẩn xác
for res in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

class TextPreprocessor:
    def __init__(self, dedup_sentences: bool = True):
        self.dedup_sentences = dedup_sentences
        
        # Các biểu thức chính quy (Regex) để lọc nhiễu theo đúng bản gốc
        self._RE_NEWLINE = re.compile(r"\bNEWLINE_CHAR\b")
        # Xóa các caption rác từ dữ liệu crawl (ví dụ: "enlarge this image toggle caption...")
        self._RE_CAPTION = re.compile(r"(?i)enlarge this image toggle caption.*?(photo credit|ap photo).*?\b")
        # Xóa ký tự phân cách pipe lạ
        self._RE_PIPE = re.compile(r"\|\|\|\|\|")
        # Chuẩn hóa khoảng trắng thừa
        self._RE_MULTIPLE_SPACES = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """Thực thi chuỗi thao tác làm sạch trên một chuỗi văn bản"""
        if not text:
            return ""
            
        # 1. Xóa các ký tự nhiễu đặc thù
        text = self._RE_NEWLINE.sub(" ", text)
        text = self._RE_CAPTION.sub(" ", text)
        text = self._RE_PIPE.sub(" ", text)

        # 2. Xóa các câu bị lặp lại liên tiếp (Sentence Deduplication)
        # Kỹ thuật này giúp mô hình sinh tóm tắt không bị lặp từ
        if self.dedup_sentences:
            try:
                sents = nltk.sent_tokenize(text)
                seen_sents = set()
                out_sents = []
                for s in sents:
                    s_lower = s.lower().strip()
                    # Chỉ giữ lại câu nếu nó chưa xuất hiện và có độ dài hợp lý
                    if s_lower not in seen_sents and len(s_lower) > 5:
                        seen_sents.add(s_lower)
                        out_sents.append(s)
                text = " ".join(out_sents)
            except Exception as e:
                # Fallback an toàn nếu nltk lỗi
                pass 

        # 3. Chuẩn hóa khoảng trắng cuối cùng
        text = self._RE_MULTIPLE_SPACES.sub(" ", text).strip()
        return text

    def process_dataset(self, dataset: list) -> list:
        """Áp dụng làm sạch cho toàn bộ tập dữ liệu (cả tài liệu gốc và bản tóm tắt)"""
        print(f"  [PREPROC] Đang làm sạch và khử lặp câu cho {len(dataset)} mẫu...")
        processed_data = []
        
        for sample in dataset:
            # Làm sạch tài liệu đầu vào (áp dụng chống lặp)
            processed_doc = self.clean_text(sample.get("document", ""))
            
            # Làm sạch bản tóm tắt tham chiếu (chỉ chuẩn hóa khoảng trắng)
            summ = sample.get("summary", "")
            processed_summ = self._RE_MULTIPLE_SPACES.sub(" ", summ).strip()

            if processed_doc and processed_summ:
                processed_data.append({
                    "document": processed_doc,
                    "summary": processed_summ,
                    "topic": sample.get("topic", "general") # Bảo lưu nhãn chủ đề nếu có
                })
                
        return processed_data