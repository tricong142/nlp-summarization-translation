import os
import logging

def load_split(data_dir: str, split_name: str, max_samples: int) -> list:
    """
    Tải một tập dữ liệu cụ thể (train, val, hoặc test).
    Ưu tiên tải file đã được clean (.src.cleaned) nếu có.
    """
    src_path = os.path.join(data_dir, f"{split_name}.src.cleaned")
    if not os.path.exists(src_path):
        src_path = os.path.join(data_dir, f"{split_name}.src") # Fallback về file gốc
        
    tgt_path = os.path.join(data_dir, f"{split_name}.tgt")

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(
            f"⚠ LỖI: Không tìm thấy dữ liệu tại:\n"
            f"  - {src_path}\n"
            f"  - {tgt_path}\n"
            f"Vui lòng kiểm tra lại thư mục {data_dir}"
        )

    data = []
    with open(src_path, 'r', encoding='utf-8') as f_src, \
         open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        
        for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
            if i >= max_samples:
                break
                
            doc = src_line.strip()
            summ = tgt_line.strip()
            
            # Chỉ lấy các dòng có dữ liệu hợp lệ
            if doc and summ:
                data.append({
                    "document": doc, 
                    "summary": summ
                })
                
    return data

def load_all_data(config) -> tuple:
    """
    Tải toàn bộ Train, Val, Test dựa trên tham số trong config.py
    """
    print(f"  [DATA] Đang tìm kiếm dữ liệu tại: {config.DATA_DIR}")
    
    train_data = load_split(config.DATA_DIR, "train", config.MAX_TRAIN_SAMPLES)
    print(f"  [DATA] ✓ Đã tải {len(train_data)} mẫu Train.")

    val_data = load_split(config.DATA_DIR, "val", config.MAX_VAL_SAMPLES)
    print(f"  [DATA] ✓ Đã tải {len(val_data)} mẫu Validation.")

    test_data = load_split(config.DATA_DIR, "test", config.MAX_TEST_SAMPLES)
    print(f"  [DATA] ✓ Đã tải {len(test_data)} mẫu Test.")

    return train_data, val_data, test_data