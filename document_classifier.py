import warnings
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")

# Từ khóa định tuyến chủ đề (Dựa trên bản gốc)
TOPIC_KEYWORDS = {
    "politics": ["treaty", "sanctions", "diplomacy", "refugee", "troops", "border", "ambassador"],
    "technology": ["algorithm", "software", "hardware", "internet", "network", "digital", "ai", "machine learning"],
    "business": ["market", "economy", "shares", "company", "investment", "trade", "profit"],
    "general": []
}

def assign_topic_label(text: str) -> str:
    """Keyword-based fallback khi dataset chưa có nhãn chủ đề."""
    text_lower = text.lower()
    scores = {
        t: sum(1 for kw in kws if kw in text_lower)
        for t, kws in TOPIC_KEYWORDS.items() if kws
    }
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)

class DocumentClassifier:
    def __init__(self, model_type: str = 'svm', class_weight: str = 'balanced'):
        self.model_type = model_type
        # Tối ưu hóa SVM cho văn bản thưa
        self.model = LinearSVC(class_weight=class_weight, random_state=42) if model_type == 'svm' else None
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        self.is_trained = False

    def prepare_labels(self, dataset: list) -> list:
        """Gán nhãn tự động cho các mẫu thiếu nhãn chủ đề"""
        print("  [SVM] Đang kiểm tra và chuẩn bị nhãn chủ đề (Labels)...")
        for s in dataset:
            if 'topic' not in s or not s['topic']:
                s['topic'] = assign_topic_label(s.get('document', ''))
        return dataset

    def train(self, dataset: list) -> None:
        """Huấn luyện mô hình SVM phân loại chủ đề"""
        if not self.model: return
        
        print("  [SVM] Trích xuất đặc trưng TF-IDF và huấn luyện mô hình...")
        texts = [s['document'] for s in dataset]
        labels = [s['topic'] for s in dataset]
        
        # Bỏ qua nếu chỉ có 1 class (ví dụ toàn là "general")
        if len(set(labels)) <= 1:
            print("  [SVM] Dữ liệu chỉ có 1 chủ đề, bỏ qua huấn luyện.")
            return

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        print("  [SVM] ✓ Huấn luyện hoàn tất.")

    def classify_and_group(self, dataset: list) -> dict:
        """Phân loại và nhóm các tài liệu theo chủ đề phục vụ Retrieval"""
        grouped = {}
        for s in dataset:
            doc = s.get('document', '')
            if self.is_trained:
                X_test = self.vectorizer.transform([doc])
                topic = self.model.predict(X_test)
            else:
                topic = s.get('topic', assign_topic_label(doc))
            
            s['predicted_topic'] = topic
            if topic not in grouped:
                grouped[topic] = []
            grouped[topic].append(doc)
            
        print(f"  [SVM] Đã nhóm tài liệu thành {len(grouped)} chủ đề.")
        return grouped
